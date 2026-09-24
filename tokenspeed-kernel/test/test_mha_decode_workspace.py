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

"""Decode-workspace contracts without importing optional GPU backends."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

MHA = Path(__file__).resolve().parents[1] / "python/tokenspeed_kernel/ops/attention/mha"


def _function(path: Path, name: str, namespace: dict):
    node = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        Mock(side_effect=AssertionError("Workspace host test attempted CUDA")),
    )
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def workspace_module():
    spec = importlib.util.spec_from_file_location(
        "mha_decode_workspace_host", MHA / "_workspace.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    return {
        "q": torch.empty((2, 4, 8), dtype=torch.bfloat16),
        "k_cache": torch.empty((2, 64, 2, 8), dtype=torch.bfloat16),
        "v_cache": torch.empty((2, 64, 2, 8), dtype=torch.bfloat16),
        "page_table": torch.tensor([[0], [1]], dtype=torch.int32),
        "cache_seqlens": torch.tensor([32, 64], dtype=torch.int32),
        "max_seqlen_k": 64,
        "max_seqlen_q": 1,
    }


def test_prepare_workspace_preserves_split_one_policy(workspace_module):
    prepare = workspace_module.prepare_mha_decode_workspace
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in inspect.signature(prepare).parameters.values()
    )
    workspace = prepare(8, torch.device("cpu"))
    assert workspace.dtype == torch.int32
    assert workspace.shape == (8,) and workspace.is_contiguous()
    assert workspace.tolist() == [1] * 8
    assert prepare(8, "cpu").data_ptr() != workspace.data_ptr()
    assert prepare(0, "cpu").shape == (0,)


@pytest.mark.parametrize("batch_size", [-1, True, 1.5])
def test_prepare_rejects_invalid_capacity(workspace_module, batch_size):
    with pytest.raises((ValueError, TypeError)):
        workspace_module.prepare_mha_decode_workspace(batch_size, "cpu")


@pytest.mark.parametrize("q_len", [1, 4])
def test_portable_wrapper_reuses_workspace_and_fallback_uses_same_policy(
    workspace_module, inputs, monkeypatch, q_len
):
    inputs["q"] = torch.empty((2 * q_len, 4, 8), dtype=torch.bfloat16)
    inputs["max_seqlen_q"] = q_len
    prepare = Mock(wraps=workspace_module.prepare_mha_decode_workspace)
    launched = Mock()
    namespace = {
        **vars(workspace_module),
        "torch": torch,
        "math": math,
        "prepare_mha_decode_workspace": prepare,
        "decode_attention_fwd": launched,
    }
    wrapper = _function(
        MHA / "_triton/decode.py",
        "_triton_mha_decode_with_kvcache_impl",
        namespace,
    )
    with pytest.raises(TypeError, match="decode_workspace"):
        wrapper(**inputs, block_n=32)
    workspace = workspace_module.prepare_mha_decode_workspace(5, "cpu")[:2]
    expected = workspace.clone()
    for method in ("cpu", "item", "tolist"):
        monkeypatch.setattr(
            torch.Tensor,
            method,
            Mock(side_effect=AssertionError("Forward read workspace values")),
        )
    for _ in range(2):
        assert (
            wrapper(**inputs, block_n=32, decode_workspace=workspace).shape
            == inputs["q"].shape
        )
    prepare.assert_not_called()
    assert all(call.args[8] is workspace for call in launched.call_args_list)
    assert torch.equal(workspace, expected)
    wrapper(**inputs, block_n=32, decode_workspace=None)
    wrapper(**inputs, block_n=32, decode_workspace=None)
    assert prepare.call_count == 2
    first, second = [call.args[8] for call in launched.call_args_list[-2:]]
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, expected) and torch.equal(second, expected)
    # The workspace path never perturbs the caller's explicit stage-1 tile.
    assert all(call.kwargs["block_n"] == 32 for call in launched.call_args_list)


@pytest.mark.parametrize(
    "invalid", ["dtype", "rank", "short", "long", "stride", "device", "not_tensor"]
)
def test_portable_wrapper_rejects_invalid_workspace_before_kernel(
    workspace_module, inputs, invalid
):
    choices = {
        "dtype": torch.ones(2, dtype=torch.int64),
        "rank": torch.ones((1, 2), dtype=torch.int32),
        "short": torch.ones(1, dtype=torch.int32),
        "long": torch.ones(3, dtype=torch.int32),
        "stride": torch.ones(4, dtype=torch.int32)[::2],
        "device": torch.empty(2, dtype=torch.int32, device="meta"),
        "not_tensor": object(),
    }
    launched = Mock()
    wrapper = _function(
        MHA / "_triton/decode.py",
        "_triton_mha_decode_with_kvcache_impl",
        {
            **vars(workspace_module),
            "torch": torch,
            "math": math,
            "decode_attention_fwd": launched,
        },
    )
    with pytest.raises((ValueError, TypeError), match="decode_workspace"):
        wrapper(**inputs, block_n=32, decode_workspace=choices[invalid])
    launched.assert_not_called()


def test_workspace_preparation_is_reexported_by_public_facade():
    tree = ast.parse((MHA / "__init__.py").read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module.endswith("._workspace")
        and any(name.name == "prepare_mha_decode_workspace" for name in node.names)
        for node in tree.body
    )
    exported = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    assert "prepare_mha_decode_workspace" in exported


def test_public_facade_requires_and_forwards_workspace_without_dispatch_change(inputs):
    selected = Mock(return_value=object())
    selected.name = "selected-decode"
    select = Mock(return_value=selected)
    namespace = {
        "torch": torch,
        "_blockscaled_signature_and_scales": lambda *args: ("signature", {}),
        "select_kernel": select,
        "ShapeCapture": SimpleNamespace(get=lambda: SimpleNamespace(record=Mock())),
        "kernel_scope": lambda *args, **kwargs: nullcontext(),
        "pdl_enabled": lambda: False,
    }
    facade = _function(MHA / "__init__.py", "mha_decode_with_kvcache", namespace)
    with pytest.raises(TypeError, match="decode_workspace"):
        facade(**inputs)
    workspace = object()
    assert facade(**inputs, decode_workspace=workspace) is selected.return_value
    assert selected.call_args.kwargs["decode_workspace"] is workspace
    before = select.call_args
    facade(**inputs, decode_workspace=None)
    assert selected.call_args.kwargs["decode_workspace"] is None
    assert select.call_args == before


def test_portable_adapter_forwards_required_workspace(inputs):
    implementation = Mock(return_value=object())
    # The registered adapter derives its stage-1 KV tile once through
    # ``grouped_decode_block_n`` (_triton/decode.py) and forwards it as the
    # host's required ``block_n``; the host has no default for it.
    derive = Mock(return_value=128)
    adapter = _function(
        MHA / "triton.py",
        "triton_mha_decode_with_kvcache",
        {
            "torch": torch,
            "_triton_mha_decode_with_kvcache_impl": implementation,
            "grouped_decode_block_n": derive,
        },
    )
    workspace = object()
    assert adapter(**inputs, decode_workspace=workspace) is implementation.return_value
    assert implementation.call_args.kwargs["decode_workspace"] is workspace
    derive.assert_called_once_with(inputs["q"], inputs["k_cache"], inputs["v_cache"])
    assert implementation.call_args.kwargs["block_n"] == derive.return_value
    with pytest.raises(TypeError, match="decode_workspace"):
        adapter(**inputs)


@pytest.mark.parametrize(
    "filename,name,target",
    [
        ("cuda.py", "fa4_mha_decode_with_kvcache", "flash_attn_varlen_func"),
        ("cuda.py", "fa4_mha_decode_with_kvcache_fp8", "flash_attn_varlen_func"),
        ("cuda.py", "fa4_mha_decode_with_kvcache_mxfp8", "flash_attn_varlen_func"),
        ("cuda.py", "fa3_mha_decode_with_kvcache", "flash_attn_with_kvcache"),
        (
            "flashinfer.py",
            "flashinfer_trtllm_mha_decode_with_kvcache",
            "trtllm_batch_decode_with_kv_cache",
        ),
        ("gluon.py", "gluon_mha_decode_gfx950", "_decode_impl"),
        ("gluon.py", "gluon_mha_decode_gfx1250", "_decode_gfx1250_impl"),
        ("triton.py", "torch_npu_mha_decode_with_kvcache", "_mha_decode_with_kvcache"),
    ],
)
def test_other_adapters_consume_workspace_without_forwarding_to_vendor(
    inputs, filename, name, target
):
    vendor = Mock()
    if target == "flash_attn_varlen_func":
        vendor.return_value = (inputs["q"], None)
    else:
        vendor.return_value = inputs["q"]
    namespace = {
        "torch": torch,
        "math": math,
        "_workspace_buffer": object(),
        "_resolve_enable_pdl": lambda value: value,
        target: vendor,
    }
    adapter = _function(MHA / filename, name, namespace)
    parameter = inspect.signature(adapter).parameters["decode_workspace"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    if "mxfp8" in name:
        inputs = {
            **inputs,
            "q_scale": torch.ones((2, 4, 1)),
            "k_scale": object(),
            "v_scale": object(),
        }
    adapter(**inputs, decode_workspace=object())
    vendor.assert_called_once()
    assert "decode_workspace" not in vendor.call_args.kwargs
    with pytest.raises(TypeError, match="decode_workspace"):
        adapter(**inputs)
