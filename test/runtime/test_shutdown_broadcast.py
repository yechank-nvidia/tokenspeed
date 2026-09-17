# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""CPU-only real-class tests; only torch.distributed transport is simulated."""

from __future__ import annotations

import threading
from test.ci_system.ci_register import register_cuda_ci
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.engine import request_handler as handler_module
from tokenspeed.runtime.engine.event_loop import EventLoop
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.utils import common

register_cuda_ci(est_time=10, suite="runtime-1gpu")


class _BroadcastBus:
    """Two CPU ranks, ordered rendezvous; sync payload uses real broadcast_pyobj."""

    def __init__(self, *, ranks, fast_rank):
        self.ranks = ranks
        self.fast_rank = fast_rank
        self.condition = threading.Condition()
        self.next = {rank: 0 for rank in ranks}
        self.entries = {}
        self.headers = []
        self.ahead = threading.Event()

    def broadcast(self, tensor, src, group, async_op=False):
        assert tensor.device.type == "cpu"
        rank = group.rank
        with self.condition:
            index = self.next[rank]
            self.next[rank] += 1
            entry = self.entries.setdefault(
                index, {"tensors": {}, "async": async_op, "src": src}
            )
            assert entry["async"] == async_op and entry["src"] == src
            assert rank not in entry["tensors"]
            entry["tensors"][rank] = tensor
            if len(entry["tensors"]) == len(self.ranks):
                value = entry["tensors"][src].clone()
                for target in entry["tensors"].values():
                    assert target.shape == value.shape and target.dtype == value.dtype
                    target.copy_(value)
                if async_op:
                    self.headers.append(value.item())
                entry["done"] = True
                self.condition.notify_all()
            if index == 1 and async_op and rank == self.fast_rank:
                self.ahead.set()

        def wait():
            # Hold slow rank's first empty-header finish until fast rank has
            # already posted the next header. No sleeps or probabilistic race.
            if index == 0 and self.fast_rank is not None and rank != self.fast_rank:
                assert self.ahead.wait(5), "fast rank failed to pre-post next header"
            with self.condition:
                assert self.condition.wait_for(
                    lambda: entry.get("done", False), timeout=5
                ), f"unmatched collective index={index}, rank={rank}, counts={self.next}"
            return True

        work = SimpleNamespace(wait=wait)
        if async_op:
            return work
        work.wait()
        return None


def _run_ranks(ranks, callback):
    errors = []

    def run(rank):
        try:
            callback(rank)
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=run, args=(rank,), daemon=True) for rank in ranks
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=12)
    assert not any(
        thread.is_alive() for thread in threads
    ), "rank thread did not terminate"
    if errors:
        raise errors[0]


def _install_transport(monkeypatch, bus):
    monkeypatch.setattr(common.dist, "broadcast", bus.broadcast)
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(handler_module, "prepare_shm_features", lambda *_args: None)


@pytest.mark.parametrize("ranks", [(0, 1), (4, 7)])
@pytest.mark.parametrize(
    "payload,stop,expected,calls",
    [
        ([], False, [], 1),
        (["kept"], False, ["kept"], 3),
        (["not_admitted"], True, None, 1),
    ],
)
def test_header_values_and_source_owned_stop(
    monkeypatch, ranks, payload, stop, expected, calls
):
    bus = _BroadcastBus(ranks=ranks, fast_rank=None)
    _install_transport(monkeypatch, bus)
    results = {}

    def rank_body(rank):
        broadcaster = common.PipelinedPyobjBroadcaster(
            rank, SimpleNamespace(rank=rank), src=ranks[0]
        )
        # Non-source TERM alone cannot author the replicated stop header.
        broadcaster.start(
            payload if rank == ranks[0] else None,
            shutdown_requested=stop if rank == ranks[0] else True,
        )
        results[rank] = broadcaster.finish()
        assert not broadcaster.in_flight

    _run_ranks(ranks, rank_body)
    assert results == {rank: expected for rank in ranks}
    assert bus.headers == [-1 if stop else int(bool(payload))]
    assert bus.next == {rank: calls for rank in ranks}


def _handler(rank, size, payloads):
    handler = RequestHandler.__new__(RequestHandler)
    handler.attn_tp_size = size
    handler.attn_tp_rank = rank
    handler.attn_global_rank = rank
    handler.attn_tp_cpu_group = SimpleNamespace(rank=rank)
    handler.shutdown_received = False
    handler.req_broadcaster = (
        common.PipelinedPyobjBroadcaster(rank, handler.attn_tp_cpu_group, src=0)
        if size > 1
        else None
    )
    values = iter(payloads)
    handler._drain_reqs = lambda: next(values, []) if rank == 0 else None
    return handler


@pytest.mark.parametrize("fast_rank", [0, 1])
@pytest.mark.parametrize("follower_term", [False, True])
def test_one_ahead_pending_payload_is_preserved_then_stop_has_no_extra_collective(
    monkeypatch, fast_rank, follower_term
):
    bus = _BroadcastBus(ranks=(0, 1), fast_rank=fast_rank)
    _install_transport(monkeypatch, bus)
    payload = [{"rid": "already-broadcast-pending-request", "tokens": [3, 5, 8]}]
    handlers = {rank: _handler(rank, 2, [[], payload]) for rank in (0, 1)}
    outputs = {rank: [] for rank in (0, 1)}

    def rank_body(rank):
        handler = handlers[rank]
        # Source stops at round1, AFTER ordinary round1 header was prefetched.
        # Follower may have local TERM from round0, or never receive it at all.
        for round_index in range(3):
            local_term = round_index >= 1 if rank == 0 else follower_term
            outputs[rank].append(handler.recv_reqs(shutdown_requested=local_term))
            assert handler.shutdown_received is (round_index == 2)
            assert handler.req_broadcaster.in_flight is (round_index != 2)

    _run_ranks((0, 1), rank_body)
    assert outputs == {rank: [[], payload, []] for rank in (0, 1)}
    assert bus.ahead.is_set()
    assert bus.headers == [0, 1, -1]
    # Empty header + payload header/size/data + stop header, exactly on BOTH
    # ranks. No speculative header may be posted after the consumed sentinel.
    assert bus.next == {0: 5, 1: 5}
    assert all(
        len(entry["tensors"]) == 2 and entry["done"] for entry in bus.entries.values()
    )


def test_tp1_local_stop_latches_without_transport(monkeypatch):
    monkeypatch.setattr(
        common.dist,
        "broadcast",
        lambda *_args, **_kwargs: pytest.fail("TP1 collective"),
    )
    monkeypatch.setattr(handler_module, "prepare_shm_features", lambda *_args: None)
    handler = _handler(0, 1, [["ordinary"], ["not_admitted"]])
    assert handler.recv_reqs(shutdown_requested=False) == ["ordinary"]
    assert not handler.shutdown_received
    assert handler.recv_reqs(shutdown_requested=True) == []
    assert handler.shutdown_received


@pytest.mark.parametrize("local_term", [False, True])
@pytest.mark.parametrize("received_stop", [False, True])
def test_event_loop_exit_depends_on_replicated_latch_not_local_term(
    local_term, received_stop
):
    loop = SimpleNamespace(
        _shutdown_requested=lambda: local_term,
        request_handler=SimpleNamespace(shutdown_received=received_stop),
    )
    assert EventLoop._shutdown_complete(loop) is received_stop
