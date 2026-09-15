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

"""Explicit graph-owner release and data-plane shutdown contracts."""

from __future__ import annotations

import os
import queue
import sys
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.execution import forward_step as forward_step_module
from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
from tokenspeed.runtime.execution.device import DeviceHandle
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.execution.forward_thread import ForwardThread
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph


class _Resource:
    pass


class _Graph:
    def __init__(self, name, events, fails):
        self.name = name
        self.events = events
        self.fails = fails

    def replay(self):
        self.events.append("replay:" + self.name)

    def reset(self):
        self.events.append("reset:" + self.name)
        if self.fails:
            raise RuntimeError("reset failed: " + self.name)


def _breakable(graphs):
    capture = BreakableCapture.__new__(BreakableCapture)
    capture._graphs = list(graphs)
    capture.segments = [g.replay for g in graphs]
    capture._capturing = False
    capture._current_graph = None
    capture._handoff = {"value": _Resource()}
    capture.pool = _Resource()
    return capture


def _prefill(captures):
    prefill = PrefillGraph.__new__(PrefillGraph)
    prefill._captures = dict(enumerate(captures))
    prefill._outputs = {"value": _Resource()}
    prefill._ctx = _Resource()
    prefill._input_embeds_buf = _Resource()
    prefill._pool = _Resource()
    return prefill


def _decode(graphs, device_module):
    runner = ForwardStepRunner.__new__(ForwardStepRunner)
    runner.graphs = dict(enumerate(graphs))
    runner.output_buffers = {"value": _Resource()}
    runner._metadata_snapshots = {"value": _Resource()}
    runner.device_module = device_module
    return runner


def test_breakable_close_releases_handles_and_buffers_once():
    events = []
    capture = _breakable(
        [_Graph("first", events, False), _Graph("last", events, False)]
    )
    owners = [weakref.ref(t) for t in capture._graphs]
    handoff = weakref.ref(capture._handoff["value"])
    capture.close()
    capture.close()
    assert events == ["reset:last", "reset:first"]
    assert all(ref() is None for ref in owners)
    assert handoff() is None
    assert capture._graphs == [] and capture.segments == [] and capture._handoff == {}
    assert capture.pool is None and capture._current_graph is None


@pytest.mark.parametrize("active", ["segment", "eager_break", "other_capture"])
def test_breakable_close_rejects_active_capture(monkeypatch, active):
    events = []
    capture = _breakable([_Graph("g", events, False)])
    capture._capturing = active == "segment"
    current = (
        None
        if active == "segment"
        else capture if active == "eager_break" else _breakable([])
    )
    monkeypatch.setattr(BreakableCapture, "_active", current)
    with pytest.raises(RuntimeError, match="during capture"):
        capture.close()
    assert events == []
    assert len(capture._graphs) == 1


def test_breakable_reset_failure_retains_owners():
    events = []
    capture = _breakable([_Graph("first", events, False), _Graph("last", events, True)])
    with pytest.raises(RuntimeError, match="reset failed"):
        capture.close()
    assert events == ["reset:last"]
    assert len(capture._graphs) == 2 and len(capture.segments) == 2
    assert capture.pool is not None


def test_prefill_close_resets_buckets_and_drops_owned_state():
    events = []
    prefill = _prefill(
        [
            _breakable([_Graph("first", events, False)]),
            _breakable([_Graph("last", events, False)]),
        ]
    )
    outputs = weakref.ref(prefill._outputs["value"])
    inputs = weakref.ref(prefill._input_embeds_buf)
    prefill.close()
    prefill.close()
    assert events == ["reset:last", "reset:first"]
    assert prefill._captures == {} and prefill._outputs == {}
    assert (
        prefill._ctx is None
        and prefill._input_embeds_buf is None
        and prefill._pool is None
    )
    assert outputs() is None and inputs() is None


def test_decode_close_fences_resets_and_releases_pool(monkeypatch):
    events = []
    runner = _decode(
        [_Graph("first", events, False), _Graph("last", events, False)],
        SimpleNamespace(synchronize=lambda: events.append("sync")),
    )
    monkeypatch.setattr(forward_step_module, "global_graph_memory_pool", _Resource())
    outputs = weakref.ref(runner.output_buffers["value"])
    runner.close()
    assert events == ["sync", "reset:last", "reset:first", "sync"]
    assert (
        runner.graphs == {}
        and runner.output_buffers == {}
        and runner._metadata_snapshots == {}
    )
    assert forward_step_module.global_graph_memory_pool is None
    assert outputs() is None
    runner.close()
    assert events == ["sync", "reset:last", "reset:first", "sync", "sync", "sync"]


def test_decode_reset_failure_preserves_pool_and_owners(monkeypatch):
    events = []
    runner = _decode(
        [_Graph("bad", events, True)],
        SimpleNamespace(synchronize=lambda: events.append("sync")),
    )
    pool = _Resource()
    monkeypatch.setattr(forward_step_module, "global_graph_memory_pool", pool)
    with pytest.raises(RuntimeError, match="reset failed"):
        runner.close()
    assert events == ["sync", "reset:bad"]
    assert runner.graphs and runner.output_buffers and runner._metadata_snapshots
    assert forward_step_module.global_graph_memory_pool is pool


def _handle(events, failure, with_l2):
    thread = ForwardThread(torch.device("cpu"))
    owner = thread.run(threading.get_ident)

    def record(name):
        assert threading.get_ident() == owner
        events.append(name)
        if name == failure:
            raise RuntimeError("failed: " + name)

    executor = SimpleNamespace(
        forward_thread=thread,
        device_module=SimpleNamespace(synchronize=lambda: record("sync")),
        prefill_graph=SimpleNamespace(close=lambda: record("prefill")),
        forward_step=SimpleNamespace(close=lambda: record("decode")),
    )
    l2 = SimpleNamespace(shutdown=lambda: record("l2")) if with_l2 else None
    return DeviceHandle(executor, l2_cache_executor=l2, kv_transfer=None), thread


@pytest.mark.parametrize("with_l2", [False, True])
def test_device_close_follows_queued_work_on_forward_thread(with_l2):
    events = []
    handle, thread = _handle(events, None, with_l2)
    try:
        pending = thread.submit(lambda: events.append("forward"))
        handle.close()
        assert pending.done()
        assert events == ["forward"] + (["l2"] if with_l2 else []) + [
            "sync",
            "prefill",
            "decode",
        ]
        assert not thread._thread.is_alive()
        previous = events.copy()
        handle.close()
        assert events == previous
        with pytest.raises(RuntimeError, match="shut down"):
            thread.submit(lambda: None)
    finally:
        thread.shutdown()


@pytest.mark.parametrize("failure", ["l2", "sync", "prefill", "decode"])
def test_device_close_failure_stops_thread_and_remains_a_failure(failure):
    events = []
    handle, thread = _handle(events, failure, True)
    try:
        with pytest.raises(RuntimeError, match="failed: " + failure):
            handle.close()
        assert not thread._thread.is_alive()
        with pytest.raises(RuntimeError, match="previously failed"):
            handle.close()
        with pytest.raises(RuntimeError, match="shut down"):
            thread.submit(lambda: None)
    finally:
        thread.shutdown()


@pytest.mark.parametrize("queue", ["_l2_submissions", "_transfer_submissions"])
def test_device_close_surfaces_unpolled_submission_failure(queue):
    events = []
    handle, thread = _handle(events, None, False)

    def fail():
        raise ValueError("submission failed")

    getattr(handle, queue).append(thread.submit(fail))
    try:
        with pytest.raises(ValueError, match="submission failed"):
            handle.close()
        assert not thread._thread.is_alive()
        with pytest.raises(RuntimeError, match="previously failed"):
            handle.close()
    finally:
        thread.shutdown()


def test_forward_thread_rejects_submissions_during_shutdown():
    thread = ForwardThread(torch.device("cpu"))
    release = threading.Event()
    queued = threading.Event()
    first = thread.submit(release.wait)
    second = thread.submit(queued.set)
    shutdown_error = []

    def shutdown():
        try:
            thread.shutdown()
        except BaseException as exc:
            shutdown_error.append(exc)

    closer = threading.Thread(target=shutdown)
    closer.start()
    try:
        # Acquire the same lock to observe that the stop item was enqueued.
        for _ in range(1000):
            with thread._submission_lock:
                stopped = thread._shutdown_requested
            if stopped:
                break
            release.wait(0.001)
        assert stopped
        with pytest.raises(RuntimeError, match="shut down"):
            thread.submit(lambda: None)
        assert not queued.is_set()
    finally:
        release.set()
        closer.join(timeout=5)
        thread.shutdown()
    assert first.result() is True and second.result() is None
    assert queued.is_set() and not closer.is_alive() and not shutdown_error


def test_forward_thread_join_timeout_is_reported():
    thread = ForwardThread.__new__(ForwardThread)
    thread._queue = queue.SimpleQueue()
    thread._submission_lock = threading.Lock()
    thread._shutdown_requested = False
    waits = []
    thread._thread = SimpleNamespace(
        join=lambda timeout: waits.append(timeout), is_alive=lambda: True
    )
    with pytest.raises(RuntimeError, match="did not stop"):
        thread.shutdown()
    assert waits == [30]
    assert thread._queue.get_nowait() is None
    with pytest.raises(RuntimeError, match="shut down"):
        thread.submit(lambda: None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_breakable_capture_can_be_reused_after_close():
    x = torch.arange(16, device="cuda", dtype=torch.float32)
    capture = BreakableCapture(pool=None, stream=None)
    for _ in range(3):
        y = x * 2
        y.add_(1)
    torch.cuda.synchronize()
    for _ in range(2):
        with capture:
            y = x * 2
            capture.add_eager(lambda: y.add_(1))
            result = y * 3
        x.add_(1)
        capture.replay(valid_rows=None)
        torch.testing.assert_close(result, (x * 2 + 1) * 3)
        torch.cuda.synchronize()
        capture.close()
        assert capture.num_segments == 0
        assert not capture._graphs and not capture.segments
        assert capture.pool is None
