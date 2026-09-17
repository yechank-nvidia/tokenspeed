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

"""Real graph-owner shutdown contracts with fake CUDA resources; no allocation."""

from functools import partial
from test.ci_system.ci_register import register_cuda_ci
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
from tokenspeed.runtime.multimodal.encoder_cudagraph import EncoderForwardStepRunner

register_cuda_ci(est_time=10, suite="runtime-1gpu")


OWNERS = (ForwardStepRunner, BreakableCapture, PrefillGraph, EncoderForwardStepRunner)


def _owner(cls, resources):
    owner = cls.__new__(cls)
    if cls is ForwardStepRunner:
        owner.graphs = dict(enumerate(resources))
        owner.output_buffers, owner._metadata_snapshots = {0: object()}, {0: object()}
        fields = ("graphs", "output_buffers", "_metadata_snapshots")
    elif cls is BreakableCapture:
        owner._graphs, owner.segments = list(resources), [Mock()]
        owner._capturing, owner._current_graph = False, None
        owner._handoff, owner.pool = {0: object()}, object()
        fields = ("_graphs", "segments", "_handoff", "pool")
    elif cls is PrefillGraph:
        owner._captures = dict(enumerate(resources))
        owner._outputs, owner._pool = {0: object()}, object()
        fields = ("_captures", "_outputs", "_pool")
    else:
        owner.budget_graphs = {
            i: SimpleNamespace(graph=g) for i, g in enumerate(resources)
        }
        fields = ("budget_graphs",)
    return owner, fields


@pytest.mark.parametrize("cls", OWNERS)
@pytest.mark.parametrize("count", [0, 2])
def test_graph_owners_release_in_order_and_close_idempotently(cls, count):
    trace = []
    resources = [Mock() for _ in range(count)]
    method = "close" if cls is PrefillGraph else "reset"
    for i, resource in enumerate(resources):
        getattr(resource, method).side_effect = partial(trace.append, i)
    owner, fields = _owner(cls, resources)
    owner.close()
    owner.close()
    assert trace == list(range(count))
    assert all(not getattr(owner, field) for field in fields)


@pytest.mark.parametrize("cls", OWNERS)
@pytest.mark.parametrize("failure_index", [0, 1])
def test_graph_owner_failure_propagates_without_dropping_ownership(cls, failure_index):
    resources = [Mock(), Mock(), Mock()]
    method = "close" if cls is PrefillGraph else "reset"
    failure = RuntimeError("graph release failed")
    getattr(resources[failure_index], method).side_effect = failure
    owner, fields = _owner(cls, resources)
    original = {field: getattr(owner, field) for field in fields}
    contents = {
        field: value.copy()
        for field, value in original.items()
        if isinstance(value, (dict, list))
    }
    with pytest.raises(RuntimeError) as caught:
        owner.close()
    assert caught.value is failure
    for index, resource in enumerate(resources):
        assert getattr(resource, method).call_count == int(index <= failure_index)
    assert all(getattr(owner, field) is value for field, value in original.items())
    assert all(getattr(owner, field) == value for field, value in contents.items())


def test_breakable_constructor_initializes_completed_graph_ownership():
    owner = BreakableCapture(pool=None, stream=Mock())
    assert owner._graphs == []
    owner.close()


@pytest.mark.parametrize("active_owner_is_self", [True, False])
def test_breakable_rejects_close_between_active_capture_segments(
    monkeypatch, active_owner_is_self
):
    graph = Mock()
    owner, _ = _owner(BreakableCapture, [graph])
    monkeypatch.setattr(
        BreakableCapture, "_active", owner if active_owner_is_self else object()
    )
    original_graphs, original_segments = owner._graphs, owner.segments
    with pytest.raises(RuntimeError, match="active"):
        owner.close()
    graph.reset.assert_not_called()
    assert owner._graphs is original_graphs and owner._graphs == [graph]
    assert owner.segments is original_segments and len(owner.segments) == 1


def test_breakable_capture_end_failure_keeps_active_graph_and_does_not_publish_segment():
    owner, _ = _owner(BreakableCapture, [])
    graph = Mock()
    failure = RuntimeError("capture end failed")
    graph.capture_end.side_effect = failure
    owner._current_graph, owner._capturing = graph, True
    original_segments = list(owner.segments)
    with pytest.raises(RuntimeError) as caught:
        owner._end_segment()
    assert caught.value is failure
    assert owner._current_graph is graph and owner._capturing
    assert owner._graphs == [] and owner.segments == original_segments
    graph.reset.assert_not_called()


def test_breakable_end_segment_records_graph_owner_and_active_close_rejects():
    owner, _ = _owner(BreakableCapture, [])
    graph = Mock()
    owner._current_graph, owner._capturing = graph, True
    with pytest.raises(RuntimeError):
        owner.close()
    graph.reset.assert_not_called()
    owner._end_segment()
    graph.capture_end.assert_called_once_with()
    assert owner._graphs == [graph]
    assert owner.segments[-1] == graph.replay
    assert owner._current_graph is None and not owner._capturing
    owner.close()
    graph.reset.assert_called_once_with()


def _executor():
    executor = ModelExecutor.__new__(ModelExecutor)
    trace = Mock()
    executor.device, executor.device_module = "cpu", trace.device
    executor.forward_step, executor.prefill_graph = trace.decode, trace.prefill
    executor.encoder_graph_wrappers = {"target": trace.target_encoder}
    executor.drafter = SimpleNamespace(
        draft_model_runner=SimpleNamespace(
            encoder_graph_wrappers={"draft": trace.draft_encoder}
        )
    )
    return executor, trace


def test_executor_synchronizes_before_closing_all_target_and_draft_graphs():
    executor, trace = _executor()
    executor.close()
    assert [call[0] for call in trace.mock_calls] == [
        "device.synchronize",
        "decode.close",
        "prefill.close",
        "target_encoder.close",
        "draft_encoder.close",
    ]


def test_executor_shared_target_and_draft_encoder_resets_graph_only_once():
    executor, trace = _executor()
    graph = Mock()
    wrapper, _ = _owner(EncoderForwardStepRunner, [graph])
    executor.encoder_graph_wrappers = {"target": wrapper}
    executor.drafter.draft_model_runner.encoder_graph_wrappers = {"draft": wrapper}
    executor.close()
    executor.close()
    graph.reset.assert_called_once_with()
    assert wrapper.budget_graphs == {}
    assert trace.device.synchronize.call_count == 2


@pytest.mark.parametrize(
    "draft", [None, SimpleNamespace(draft_model_runner=SimpleNamespace())]
)
def test_executor_without_draft_encoder_captures(draft):
    executor, trace = _executor()
    executor.drafter = draft
    executor.encoder_graph_wrappers = {}
    executor.close()
    assert [call[0] for call in trace.mock_calls] == [
        "device.synchronize",
        "decode.close",
        "prefill.close",
    ]


@pytest.mark.parametrize(
    "stage",
    [
        "device.synchronize",
        "decode.close",
        "prefill.close",
        "target_encoder.close",
        "draft_encoder.close",
    ],
)
def test_executor_failure_stops_before_later_owners(stage):
    executor, trace = _executor()
    parent, method = stage.split(".")
    failure = RuntimeError(stage)
    getattr(getattr(trace, parent), method).side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        executor.close()
    assert caught.value is failure
    stages = [
        "device.synchronize",
        "decode.close",
        "prefill.close",
        "target_encoder.close",
        "draft_encoder.close",
    ]
    assert [call[0] for call in trace.mock_calls] == stages[: stages.index(stage) + 1]
