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

"""CPU dispatch/order contracts; device math and graph lifetimes are separate."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

RUNTIME = Path(__file__).resolve().parents[3] / "python/tokenspeed/runtime"


def _load(namespace, relative, names, class_name=None, bases=()):
    path = RUNTIME / relative
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        original = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        methods = [
            copy.deepcopy(n)
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert {n.name for n in methods} == set(names)
        cls = copy.deepcopy(original)
        cls.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
        cls.keywords, cls.decorator_list, cls.body = [], [], methods
        selected = [cls]
    else:
        selected = [
            copy.deepcopy(n)
            for n in body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert {n.name for n in selected} == set(names)
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def dispatch():
    namespace = dict(
        torch=torch,
        try_gather_token_logprobs=Mock(),
        gather_token_logprobs_torch=Mock(),
    )
    _load(namespace, "sampling/utils.py", {"gather_token_logprobs"})
    return namespace


@pytest.mark.parametrize("accepted", [True, False])
def test_dispatch_preserves_inputs_and_selected_output(dispatch, accepted):
    logits = torch.randn(2, 7)
    tokens = torch.tensor([1, 5], dtype=torch.int64)
    output = torch.tensor([-1.0, -2.0])
    fast = dispatch["try_gather_token_logprobs"]
    fallback = dispatch["gather_token_logprobs_torch"]
    fast.return_value = output if accepted else None
    fallback.return_value = output
    before = logits.clone(), tokens.clone()
    assert dispatch["gather_token_logprobs"](logits, tokens) is output
    assert fast.call_args.args == (logits, tokens)
    if accepted:
        fallback.assert_not_called()
    else:
        assert fallback.call_args.args == (logits, tokens)
    assert torch.equal(logits, before[0]) and torch.equal(tokens, before[1])


def test_dispatch_retains_fresh_results_without_cache(dispatch):
    logits, tokens = torch.zeros(1, 7), torch.zeros(1, dtype=torch.int32)
    outputs = [torch.tensor([float(i)]) for i in range(2)]
    dispatch["try_gather_token_logprobs"].side_effect = outputs
    actual = [dispatch["gather_token_logprobs"](logits, tokens) for _ in range(2)]
    assert all(a is b for a, b in zip(actual, outputs))
    assert actual[0].data_ptr() != actual[1].data_ptr()
    dispatch["gather_token_logprobs_torch"].assert_not_called()


def test_execution_error_is_not_silently_retried(dispatch):
    dispatch["try_gather_token_logprobs"].side_effect = RuntimeError("kernel failure")
    with pytest.raises(RuntimeError, match="kernel failure"):
        dispatch["gather_token_logprobs"](torch.zeros(1, 7), torch.zeros(1))
    dispatch["gather_token_logprobs_torch"].assert_not_called()


@pytest.mark.parametrize("backend_name", ["flashinfer", "greedy"])
@pytest.mark.parametrize("method", ["sample", "verify"])
@pytest.mark.parametrize("enabled", [False, True])
def test_backend_calls_shared_helper_after_broadcast(backend_name, method, enabled):
    events = []
    namespace = dict(
        torch=torch,
        nvtx_range=lambda *a, **k: lambda fn: fn,
        SPECULATIVE_ACCEPT_THRESHOLD_SINGLE=1.0,
        SPECULATIVE_ACCEPT_THRESHOLD_ACC=1.0,
        _FUSED_TOPK_TOPP_AVAILABLE=False,
        pdl_enabled=lambda: True,
    )

    def scalars(indices, **kwargs):
        return (
            torch.ones(1),
            torch.ones(1, dtype=torch.int32),
            torch.ones(1),
            None,
            torch.zeros(1, dtype=torch.int64),
            None,
        )

    def argmax(logits, **kwargs):
        value = logits.argmax(-1)
        if kwargs.get("out") is not None:
            kwargs["out"].copy_(value)
            return kwargs["out"]
        return value

    def chain(**kwargs):
        predicts = kwargs["predicts"]
        target = kwargs.get("target_predict")
        if target is None:
            target = kwargs["target_probs"].argmax(-1)
        predicts.copy_(target.flatten().to(torch.int32))
        kwargs["accept_token_num"].zero_()
        kwargs["accept_index"].copy_(
            torch.arange(predicts.numel()).view_as(kwargs["accept_index"])
        )

    def logprobs(logits, tokens):
        events.append("logprobs")
        assert events[-2] == "broadcast"
        return torch.log_softmax(logits, -1).gather(-1, tokens.long()[:, None])[:, 0]

    namespace.update(
        gather_and_expand_scalars=scalars,
        softmax=lambda logits, **kw: logits.softmax(-1),
        top_k_renorm_prob=lambda probs, topks: probs,
        top_p_renorm_prob=lambda probs, topps, **kw: probs,
        top_k_top_p_sampling_from_probs=lambda probs, *a, **kw: probs.argmax(-1),
        chain_speculative_sampling_target_only=chain,
        _verify_chain_greedy=chain,
        sampling_argmax=argmax,
        gather_token_logprobs=logprobs,
    )
    class_name = (
        "FlashInferSamplingBackend"
        if backend_name == "flashinfer"
        else "GreedySamplingBackend"
    )
    _load(namespace, f"sampling/backends/{backend_name}.py", {method}, class_name)
    backend = namespace[class_name]()
    backend.config = NS(enable_output_logprobs=enabled)
    backend.maybe_broadcast = lambda *args: events.append("broadcast")
    backend._sample_token_buf = torch.empty(1, dtype=torch.int32)
    backend._predict_buf = torch.empty(1, dtype=torch.int32)
    backend._accept_index_buf = torch.empty(1, dtype=torch.int32)
    backend._accept_length_buf = torch.empty(1, dtype=torch.int32)
    backend._ones_buf = torch.ones(1, dtype=torch.int32)
    backend._coins_buf = torch.full((1, 1), 0.5)
    backend._final_coins_buf = torch.full((1,), 0.5)
    for attr in ("_temperature_pool", "_top_k_pool", "_top_p_pool", "_seed_pool"):
        setattr(backend, attr, torch.ones(1))
    logits = torch.tensor([[0.0, 3.0, 1.0]])
    output = NS(
        next_token_logits=logits, next_token_logprobs=None, logits_layout_plan=None
    )
    info = NS(
        vocab_mask=None,
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        valid_cache_lengths=None,
        batch_row_offset=0,
    )
    args = (
        (output, info)
        if method == "sample"
        else (output, info, torch.zeros((1, 1), dtype=torch.int32))
    )
    predicted, lengths = getattr(backend, method)(*args)
    assert predicted.tolist() == [1] and lengths.tolist() == [1]
    assert events == (["broadcast", "logprobs"] if enabled else ["broadcast"])
    assert (output.next_token_logprobs is not None) == enabled


def test_dp_logprobs_remain_before_output_gather():
    path = RUNTIME / "sampling/backends/flashinfer.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FlashInferSamplingBackend"
    )
    verify = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "verify"
    )
    calls = [n for n in ast.walk(verify) if isinstance(n, ast.Call)]
    helper = sorted(
        n.lineno
        for n in calls
        if isinstance(n.func, ast.Name) and n.func.id == "gather_token_logprobs"
    )
    gather = next(
        n.lineno
        for n in calls
        if isinstance(n.func, ast.Attribute) and n.func.attr == "gather_verify_outputs"
    )
    assert len(helper) == 2 and helper[0] < gather < helper[1]


def test_no_old_helper_calls_or_model_specific_dispatch():
    for name, expected in (("flashinfer", 3), ("greedy", 2)):
        tree = ast.parse((RUNTIME / f"sampling/backends/{name}.py").read_text())
        calls = [
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        assert calls.count("gather_token_logprobs") == expected
        assert "gather_token_logprobs_torch" not in calls
