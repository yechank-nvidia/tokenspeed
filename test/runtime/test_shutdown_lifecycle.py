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

"""Owned shutdown contracts; CPU fakes, no model or distributed initialization."""

import asyncio
import signal
from datetime import timedelta
from test.ci_system.ci_register import register_cuda_ci
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tokenspeed.runtime.distributed import process_group_manager as groups_module
from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager
from tokenspeed.runtime.engine import async_llm as async_llm_module
from tokenspeed.runtime.engine.aio_rwlock import RWLock
from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.entrypoints import engine as engine_module
from tokenspeed.runtime.utils import process as process_module

# Full-runtime imports require the runtime environment, but tests do no GPU work.
register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.fixture
def owner():
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.scheduler_processes = (object(),)
    engine._sigterm_watchdog_task = None
    engine.gracefully_exit = False
    engine.rid_to_state = {}
    engine._generation_admit = asyncio.Event()
    engine.model_update_lock = RWLock()
    engine.auto_create_handle_loop = Mock()
    engine.input_processor = SimpleNamespace(validate_request=Mock())
    engine.log_requests = False
    return engine


def test_shutdown_joins_watchdog_and_in_tokenization_request(owner, monkeypatch):
    async def run():
        tokenizing, release = asyncio.Event(), asyncio.Event()
        trace = []

        async def watchdog():
            try:
                await asyncio.Event().wait()
            finally:
                assert not owner.gracefully_exit
                trace.append("watchdog_joined")

        async def tokenize(_obj):
            tokenizing.set()
            await release.wait()
            return object()

        async def responses(_obj):
            yield "done"

        def stop(processes, *, timeout_seconds):
            assert processes is owner.scheduler_processes
            assert timeout_seconds > 0
            assert trace == ["watchdog_joined", "request_finished"]
            trace.append("children_reaped")

        async def request():
            obj = SimpleNamespace(is_single=True, normalize_batch_and_arguments=Mock())
            assert [value async for value in owner.generate_request(obj)] == ["done"]
            trace.append("request_finished")

        owner._tokenize_one_request = tokenize
        owner._send_one_request = Mock()
        owner._wait_one_response = responses
        owner._generation_admit.set()
        owner._sigterm_watchdog_task = asyncio.create_task(watchdog())
        monkeypatch.setattr(async_llm_module, "stop_owned_processes", stop)
        generating = asyncio.create_task(request())
        await tokenizing.wait()
        stopping = asyncio.create_task(
            owner.shutdown_owned_schedulers(timeout_seconds=2)
        )
        while not owner.gracefully_exit:
            await asyncio.sleep(0)
        assert trace == ["watchdog_joined"]
        assert not stopping.done()
        release.set()
        await generating
        await stopping
        assert trace == ["watchdog_joined", "request_finished", "children_reaped"]

    asyncio.run(run())


def test_shutdown_wakes_paused_admission_into_rejection(owner, monkeypatch):
    async def run():
        request = asyncio.create_task(anext(owner.generate_request(object())))
        await asyncio.sleep(0)
        stop = Mock()
        monkeypatch.setattr(async_llm_module, "stop_owned_processes", stop)
        await owner.shutdown_owned_schedulers(timeout_seconds=1)
        with pytest.raises(RuntimeError, match="shutting down"):
            await request
        owner.input_processor.validate_request.assert_not_called()
        stop.assert_called_once()

    asyncio.run(run())


def test_first_request_after_shutdown_does_not_create_handle_loop(owner, monkeypatch):
    async def run():
        stop = Mock()
        monkeypatch.setattr(async_llm_module, "stop_owned_processes", stop)
        await owner.shutdown_owned_schedulers(timeout_seconds=1)
        with pytest.raises(RuntimeError, match="shutting down"):
            await anext(owner.generate_request(object()))
        owner.auto_create_handle_loop.assert_not_called()
        owner.input_processor.validate_request.assert_not_called()
        assert owner._sigterm_watchdog_task is None
        stop.assert_called_once()

    asyncio.run(run())


def test_request_waiting_for_writer_rejects_before_tokenization(owner):
    async def run():
        owner._generation_admit.set()
        owner._tokenize_one_request = Mock()
        obj = SimpleNamespace(normalize_batch_and_arguments=Mock())
        async with owner.model_update_lock.writer_lock:
            request = asyncio.create_task(anext(owner.generate_request(obj)))
            await asyncio.sleep(0)
            owner.gracefully_exit = True
        with pytest.raises(RuntimeError, match="shutting down"):
            await request
        owner._tokenize_one_request.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["drain_timeout", "child_cleanup"])
def test_shutdown_failure_propagates_and_attempts_owned_cleanup(
    owner, monkeypatch, failure
):
    async def run():
        stop = Mock()
        monkeypatch.setattr(async_llm_module, "stop_owned_processes", stop)
        if failure == "drain_timeout":
            owner.rid_to_state["active"] = object()
            with pytest.raises(TimeoutError):
                await owner.shutdown_owned_schedulers(timeout_seconds=0.001)
        else:
            stop.side_effect = RuntimeError("owned child failed")
            with pytest.raises(RuntimeError, match="owned child failed"):
                await owner.shutdown_owned_schedulers(timeout_seconds=1)
        stop.assert_called_once()
        assert stop.call_args.args == (owner.scheduler_processes,)

    asyncio.run(run())


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_async_shutdown_requires_positive_finite_timeout(owner, monkeypatch, timeout):
    stop = Mock()
    monkeypatch.setattr(async_llm_module, "stop_owned_processes", stop)
    with pytest.raises(ValueError, match="positive and finite"):
        asyncio.run(owner.shutdown_owned_schedulers(timeout_seconds=timeout))
    assert not owner.gracefully_exit
    stop.assert_not_called()


class _Child:
    def __init__(self, pid, exit_after_term, trace):
        self.pid, self.exit_after_term, self.trace = pid, exit_after_term, trace
        self.exitcode = None

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        self.trace.append(("term", self.pid))

    def join(self, *, timeout):
        self.trace.append(("join", self.pid))
        if self.exitcode is None:
            self.exitcode = self.exit_after_term


def test_owned_children_all_receive_term_before_wait(monkeypatch):
    trace = []
    children = [_Child(1, 0, trace), _Child(2, 0, trace)]
    kill = Mock()
    monkeypatch.setattr(process_module, "kill_process_tree", kill)
    process_module.stop_owned_processes(children, timeout_seconds=1)
    assert trace == [("term", 1), ("term", 2), ("join", 1), ("join", 2)]
    assert all(child.exitcode == 0 for child in children)
    kill.assert_not_called()


@pytest.mark.parametrize("exit_after_term", [-signal.SIGTERM, None])
def test_nonzero_or_forced_child_exit_is_never_success(monkeypatch, exit_after_term):
    child = _Child(1, exit_after_term, [])

    def kill(pid, *, include_parent):
        assert (pid, include_parent) == (child.pid, True)
        child.exitcode = -signal.SIGKILL

    monkeypatch.setattr(process_module, "kill_process_tree", kill)
    with pytest.raises(RuntimeError, match="not cleanly reaped|forced cleanup"):
        process_module.stop_owned_processes([child], timeout_seconds=0)
    assert child.trace[-1] == ("join", child.pid)
    assert not child.is_alive()


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_owned_cleanup_rejects_nonfinite_or_negative_timeout(timeout):
    child = _Child(1, 0, [])
    with pytest.raises(ValueError, match="nonnegative and finite"):
        process_module.stop_owned_processes([child], timeout_seconds=timeout)
    assert child.trace == []


@pytest.mark.parametrize(
    "outcome",
    [
        "term",
        "term_after_poll",
        "startup_stop",
        "bad_ready",
        "child_failure",
        "unexpected_exit",
        "cleanup_failure",
    ],
)
def test_follower_only_returns_marker_after_requested_clean_stop(monkeypatch, outcome):
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)
    original, handlers = {sig: object() for sig in signals}, {}
    child = SimpleNamespace(exitcode=0 if outcome == "unexpected_exit" else None)

    def poll(_timeout):
        if outcome == "startup_stop":
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return True

    def ready(*_args):
        if outcome not in ("unexpected_exit", "term_after_poll"):
            sig = signal.SIGUSR1 if outcome == "child_failure" else signal.SIGTERM
            with monkeypatch.context() as signal_patch:
                signal_patch.setattr(
                    engine_module.threading.Event,
                    "set",
                    lambda _self: pytest.fail("signal handler acquired an Event lock"),
                )
                handlers[sig](sig, None)
                handlers[sig](sig, None)

    def idle_poll(seconds):
        assert seconds == 0.05
        assert outcome == "term_after_poll"
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    stop = Mock(
        side_effect=(
            RuntimeError("cleanup failed") if outcome == "cleanup_failure" else None
        )
    )
    monkeypatch.setattr(engine_module.signal, "getsignal", original.__getitem__)
    monkeypatch.setattr(
        engine_module.signal,
        "signal",
        lambda sig, handler: handlers.__setitem__(sig, handler),
    )
    monkeypatch.setattr(engine_module, "launch_dummy_health_check_server", ready)
    monkeypatch.setattr(engine_module.time, "sleep", idle_poll)
    monkeypatch.setattr(engine_module, "stop_owned_processes", stop)
    reader = SimpleNamespace(
        poll=poll, recv=lambda: {"status": "bad" if outcome == "bad_ready" else "ready"}
    )
    args = SimpleNamespace(host="localhost", port=1, enable_metrics=False)
    if outcome in ("term", "term_after_poll"):
        assert engine_module._wait_for_follower_shutdown(args, [reader], [child]) == (
            None,
            None,
            {"follower_shutdown_complete": True},
        )
    else:
        with pytest.raises(RuntimeError):
            engine_module._wait_for_follower_shutdown(args, [reader], [child])
    stop.assert_called_once_with([child], timeout_seconds=30.0)
    assert handlers == original


@pytest.fixture
def distributed_owner(monkeypatch):
    trace = []
    groups = ProcessGroupManager()
    groups.register_process_group("nccl", (0, 1), "device_world")
    groups.register_process_group("nccl", (0,), "device_local")
    groups.register_process_group("gloo", (0, 1), "gloo_world")
    state = SimpleNamespace(initialized=True)

    def destroy(*args):
        if args:
            trace.append(("destroy", args[0]))
        else:
            trace.append(("destroy_all",))
            world.shutdown()
            state.initialized = False

    def barrier(*, group, timeout, wait_all_ranks):
        assert (group, timeout, wait_all_ranks) == (
            "gloo_world",
            timedelta(seconds=30),
            True,
        )
        assert groups._process_groups["gloo"][(0, 1)] == "gloo_world"
        trace.append(("barrier",))

    world = SimpleNamespace(
        shutdown=Mock(side_effect=lambda: trace.append(("world_shutdown",)))
    )
    fake = SimpleNamespace(
        is_initialized=lambda: state.initialized,
        get_world_size=lambda: 2,
        group=SimpleNamespace(WORLD=world),
        destroy_process_group=Mock(side_effect=destroy),
        monitored_barrier=Mock(side_effect=barrier),
    )
    monkeypatch.setattr(groups_module, "dist", fake)
    return groups, fake, state, trace


def test_ordered_groups_keep_cpu_barrier_until_all_device_groups_stop(
    distributed_owner,
):
    groups, _, _, trace = distributed_owner
    groups.close(timeout_seconds=30.0)
    assert trace == [
        ("destroy", "device_local"),
        ("destroy", "device_world"),
        ("world_shutdown",),
        ("barrier",),
        ("destroy_all",),
        ("world_shutdown",),
    ]
    assert groups._process_groups == {}
    groups.close(timeout_seconds=30.0)
    assert len(trace) == 6


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_ordered_groups_reject_invalid_timeout_before_mutation(
    distributed_owner, timeout
):
    groups, _, _, trace = distributed_owner
    with pytest.raises(ValueError, match="positive and finite"):
        groups.close(timeout_seconds=timeout)
    assert trace == []


@pytest.mark.parametrize("missing", ["timeout", "gloo", "shutdown"])
def test_ordered_groups_require_cleanup_prerequisites(distributed_owner, missing):
    groups, fake, _, trace = distributed_owner
    if missing == "timeout":
        with pytest.raises(TypeError):
            groups.close()
    else:
        if missing == "gloo":
            groups._process_groups["gloo"].clear()
        else:
            del fake.group.WORLD.shutdown
        with pytest.raises((KeyError, RuntimeError)):
            groups.close(timeout_seconds=30.0)
    assert trace == []


@pytest.mark.parametrize("stage", ["device", "world", "barrier", "final"])
def test_ordered_group_failure_propagates_without_clearing_cache(
    distributed_owner, stage
):
    groups, fake, _, trace = distributed_owner
    failure = RuntimeError(stage + " failed")
    if stage == "world":
        fake.group.WORLD.shutdown.side_effect = failure
    elif stage == "barrier":
        fake.monitored_barrier.side_effect = failure
    elif stage == "device":
        fake.destroy_process_group.side_effect = failure
    else:
        fake.destroy_process_group.side_effect = [None, None, failure]
    with pytest.raises(RuntimeError, match=stage + " failed"):
        groups.close(timeout_seconds=30.0)
    assert groups._process_groups
    if stage != "final":
        assert ("destroy_all",) not in trace


def test_uninitialized_groups_discard_stale_cached_objects(distributed_owner):
    groups, _, state, trace = distributed_owner
    state.initialized = False
    groups.close(timeout_seconds=30.0)
    assert groups._process_groups == {}
    assert trace == []
