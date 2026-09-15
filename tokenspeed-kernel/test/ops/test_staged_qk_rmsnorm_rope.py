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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.layernorm import staged_qk_rmsnorm_rope
from tokenspeed_kernel.ops.layernorm.triton import fused_qk_rmsnorm_rope
from tokenspeed_kernel.registry import KernelRegistry

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _reference(x, weight, cos, sin, eps):
    by_head = x.reshape(x.shape[0], -1, 128).float()
    normalized = (
        by_head * torch.rsqrt(by_head.square().mean(-1, keepdim=True) + eps)
    ).bfloat16()
    affine = (normalized * weight.bfloat16()).bfloat16()
    rotated = torch.cat((-affine[..., 64:], affine[..., :64]), dim=-1)
    first_product = (affine * cos.bfloat16()[:, None, :]).bfloat16()
    second_product = (rotated * sin.bfloat16()[:, None, :]).bfloat16()
    return (first_product + second_product).bfloat16().reshape_as(x)


def _inputs(tokens, q_heads, k_heads, strided, weight_dtype, factor_dtype):
    torch.manual_seed(811 + tokens + q_heads + k_heads)
    packed = torch.randn(
        (tokens + 1, (q_heads + k_heads + 1) * 128), dtype=torch.bfloat16, device="cuda"
    )
    q = packed[1:, : q_heads * 128]
    k = packed[1:, q_heads * 128 : (q_heads + k_heads) * 128]
    if not strided:
        q, k = q.contiguous(), k.contiguous()
    q_weight = (1.0 + 0.1 * torch.randn(128, device="cuda")).to(weight_dtype)
    k_weight = (1.0 + 0.1 * torch.randn(128, device="cuda")).to(weight_dtype)
    theta = torch.randn((tokens, 64), device="cuda")
    cos, sin = (
        torch.cat((value, value), dim=-1).to(factor_dtype)
        for value in (theta.cos(), theta.sin())
    )
    if strided:
        storage = torch.empty((2, tokens + 1, 256), device="cuda", dtype=factor_dtype)
        cos_view, sin_view = storage[0, 1:, 64:192], storage[1, 1:, 64:192]
        cos_view.copy_(cos)
        sin_view.copy_(sin)
        cos, sin = cos_view, sin_view
    return q, k, q_weight, k_weight, cos, sin


def test_staged_qk_rope_registration():
    registry = KernelRegistry.get()
    spec = registry.get_by_name("triton_staged_qk_rmsnorm_rope")
    assert spec is not None
    assert (spec.family, spec.mode) == ("layernorm", "staged_qk_rmsnorm_rope")
    assert registry.get_impl(spec.name) is staged_qk_rmsnorm_rope


@requires_gpu
@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (1, 3, 2), (5, 5, 3), (7, 7, 2), (17, 9, 3), (129, 16, 5)]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("preallocated", [False, True])
@pytest.mark.parametrize(
    "weight_dtype,factor_dtype",
    [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
)
def test_staged_qk_rope_matches_reference(
    shape, strided, preallocated, weight_dtype, factor_dtype
):
    args = _inputs(*shape, strided, weight_dtype, factor_dtype)
    q, k, qw, kw, cos, sin = args
    originals = [tensor.clone() for tensor in args]
    out = (
        (
            torch.full(q.shape, float("nan"), dtype=q.dtype, device="cuda"),
            torch.full(k.shape, float("nan"), dtype=k.dtype, device="cuda"),
        )
        if preallocated
        else None
    )
    result = staged_qk_rmsnorm_rope(*args, 1e-6, out)
    if out is not None:
        assert result is out
    for actual, source, weight in zip(result, (q, k), (qw, kw)):
        assert actual.is_contiguous()
        torch.testing.assert_close(
            actual, _reference(source, weight, cos, sin, 1e-6), atol=0.016, rtol=0.008
        )
    for actual, original in zip(args, originals):
        assert torch.equal(actual, original)
    repeated = staged_qk_rmsnorm_rope(*args, 1e-6, None)
    assert all(torch.equal(a, b) for a, b in zip(result, repeated))


@requires_gpu
def test_staged_qk_rope_preserves_rounding_boundaries_exactly():
    args = list(_inputs(2, 2, 1, False, torch.float32, torch.float32))
    q, k, qw, kw, cos, sin = args
    q.fill_(1.0)
    k.fill_(-1.0)
    # Mean squares are exactly one, so normalized values round to +/-1.
    # This isolates the affine and rotary rounding stages from reduction order.
    result = staged_qk_rmsnorm_rope(*args, 1e-6, None)
    expected = (_reference(q, qw, cos, sin, 1e-6), _reference(k, kw, cos, sin, 1e-6))
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    cache = torch.cat((cos[:, :64], sin[:, :64]), dim=-1).contiguous()
    positions = torch.arange(2, device="cuda", dtype=torch.int64)
    ordinary = fused_qk_rmsnorm_rope(q, k, qw, kw, cache, positions, 1e-6, 2, 1, 128)
    assert any(
        torch.count_nonzero(a != b).item() > 0 for a, b in zip(expected, ordinary)
    )


@requires_gpu
@pytest.mark.parametrize("preallocated", [False, True])
def test_staged_qk_rope_graph_replay_reads_updated_inputs(preallocated):
    args = _inputs(9, 3, 2, True, torch.float32, torch.float32)
    q, k, qw, kw, cos, sin = args
    out = (
        (
            torch.empty(q.shape, device="cuda", dtype=q.dtype),
            torch.empty(k.shape, device="cuda", dtype=k.dtype),
        )
        if preallocated
        else None
    )
    staged_qk_rmsnorm_rope(*args, 1e-6, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = staged_qk_rmsnorm_rope(*args, 1e-6, out)
    if out is not None:
        assert result is out
    for shift in (0.0, 0.25):
        q.add_(shift)
        k.neg_()
        qw.add_(0.015625)
        kw.sub_(0.0078125)
        cos.mul_(0.9)
        sin.mul_(0.9)
        for tensor in result:
            tensor.fill_(float("nan"))
        graph.replay()
        for actual, source, weight in zip(result, (q, k), (qw, kw)):
            torch.testing.assert_close(
                actual,
                _reference(source, weight, cos, sin, 1e-6),
                atol=0.016,
                rtol=0.008,
            )


@requires_gpu
def test_staged_qk_rope_output_storage_offsets():
    args = _inputs(7, 3, 2, True, torch.float32, torch.float32)
    q_storage = torch.full((8, 384), 77.0, dtype=torch.bfloat16, device="cuda")
    k_storage = torch.full((8, 256), 77.0, dtype=torch.bfloat16, device="cuda")
    out = (q_storage[1:], k_storage[1:])
    assert staged_qk_rmsnorm_rope(*args, 1e-6, out) is out
    for actual, source, weight in zip(out, args[:2], args[2:4]):
        torch.testing.assert_close(
            actual, _reference(source, weight, *args[4:], 1e-6), atol=0.016, rtol=0.008
        )
    assert (q_storage[0] == 77.0).all()
    assert (k_storage[0] == 77.0).all()


@requires_gpu
@pytest.mark.parametrize("preallocated", [False, True])
def test_staged_qk_rope_empty(preallocated):
    args = _inputs(0, 3, 2, False, torch.float32, torch.bfloat16)
    out = (
        (torch.empty_like(args[0]), torch.empty_like(args[1])) if preallocated else None
    )
    result = staged_qk_rmsnorm_rope(*args, 1e-6, out)
    assert result[0].shape == (0, 384)
    assert result[1].shape == (0, 256)
    if out is not None:
        assert result is out


def test_staged_qk_rope_rejects_cpu():
    q, k = torch.empty((2, 256), dtype=torch.bfloat16), torch.empty(
        (2, 128), dtype=torch.bfloat16
    )
    weight = torch.empty(128)
    factors = torch.empty((2, 128))
    with pytest.raises(ValueError, match="one GPU"):
        staged_qk_rmsnorm_rope(q, k, weight, weight, factors, factors, 1e-6, None)


@requires_gpu
@pytest.mark.parametrize("eps", [0.0, -1.0, float("inf"), float("nan")])
def test_staged_qk_rope_rejects_invalid_epsilon(eps):
    args = _inputs(2, 2, 1, False, torch.float32, torch.float32)
    with pytest.raises(ValueError, match="finite and positive"):
        staged_qk_rmsnorm_rope(*args, eps, None)


@requires_gpu
@pytest.mark.parametrize(
    "invalid",
    [
        "q_rank",
        "tokens",
        "q_dtype",
        "k_dtype",
        "q_width",
        "k_empty",
        "q_stride",
        "k_device",
        "weight_shape",
        "weight_dtype",
        "weight_stride",
        "weight_device",
        "cos_shape",
        "sin_dtype",
        "sin_stride",
        "cos_device",
    ],
)
def test_staged_qk_rope_rejects_invalid_inputs(invalid):
    args = list(_inputs(2, 2, 1, False, torch.float32, torch.float32))
    if invalid == "q_rank":
        args[0] = args[0].flatten()
    elif invalid == "tokens":
        args[1] = args[1][:1]
    elif invalid == "q_dtype":
        args[0] = args[0].float()
    elif invalid == "k_dtype":
        args[1] = args[1].half()
    elif invalid == "q_width":
        args[0] = args[0][:, :255]
    elif invalid == "k_empty":
        args[1] = args[1][:, :0]
    elif invalid == "q_stride":
        args[0] = torch.empty((2, 512), dtype=torch.bfloat16, device="cuda")[:, ::2]
    elif invalid == "k_device":
        args[1] = args[1].cpu()
    elif invalid == "weight_shape":
        args[2] = args[2][:64]
    elif invalid == "weight_dtype":
        args[3] = args[3].half()
    elif invalid == "weight_stride":
        args[2] = torch.empty(256, device="cuda")[::2]
    elif invalid == "weight_device":
        args[3] = args[3].cpu()
    elif invalid == "cos_shape":
        args[4] = args[4][:, :64]
    elif invalid == "sin_dtype":
        args[5] = args[5].double()
    elif invalid == "sin_stride":
        args[5] = torch.empty((2, 256), device="cuda")[:, ::2]
    elif invalid == "cos_device":
        args[4] = args[4].cpu()
    with pytest.raises((TypeError, ValueError)):
        staged_qk_rmsnorm_rope(*args, 1e-6, None)


@requires_gpu
@pytest.mark.parametrize(
    "invalid", ["list", "length", "non_tensor", "shape", "dtype", "device", "stride"]
)
def test_staged_qk_rope_rejects_invalid_outputs(invalid):
    args = _inputs(2, 2, 1, False, torch.float32, torch.float32)
    out = (torch.empty_like(args[0]), torch.empty_like(args[1]))
    if invalid == "list":
        out = list(out)
    elif invalid == "length":
        out = out[:1]
    elif invalid == "non_tensor":
        out = (None, out[1])
    elif invalid == "shape":
        out = (out[0].flatten(), out[1])
    elif invalid == "dtype":
        out = (out[0].float(), out[1])
    elif invalid == "device":
        out = (out[0].cpu(), out[1])
    elif invalid == "stride":
        out = (
            torch.empty((2, 512), dtype=torch.bfloat16, device="cuda")[:, ::2],
            out[1],
        )
    with pytest.raises(ValueError):
        staged_qk_rmsnorm_rope(*args, 1e-6, out)


@requires_gpu
@pytest.mark.parametrize("alias", ["q", "k", "cos", "outputs", "disjoint_view"])
def test_staged_qk_rope_rejects_output_aliases(alias):
    args = _inputs(2, 1, 1, False, torch.bfloat16, torch.bfloat16)
    out = (torch.empty_like(args[0]), torch.empty_like(args[1]))
    if alias == "q":
        out = (args[0], out[1])
    elif alias == "k":
        out = (args[1], out[1])
    elif alias == "cos":
        out = (args[4], out[1])
    elif alias == "outputs":
        out = (out[0], out[0])
    else:
        storage = torch.empty((4, 128), dtype=torch.bfloat16, device="cuda")
        out = (storage[:2], storage[2:])
    with pytest.raises(ValueError, match="storage must be separate"):
        staged_qk_rmsnorm_rope(*args, 1e-6, out)
