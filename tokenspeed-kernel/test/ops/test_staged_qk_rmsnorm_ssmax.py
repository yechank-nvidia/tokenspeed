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
from tokenspeed_kernel.ops.layernorm import staged_qk_rmsnorm_ssmax
from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm
from tokenspeed_kernel.registry import KernelRegistry

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _norm_reference(x, weight, eps):
    by_head = x.reshape(
        x.shape[0], x.shape[1] // weight.numel(), weight.numel()
    ).float()
    normalized = (
        by_head * torch.rsqrt(by_head.square().mean(-1, keepdim=True) + eps)
    ).bfloat16()
    return (normalized * weight.bfloat16()).bfloat16().reshape_as(x)


def _reference(q, k, qw, kw, positions, table, eps):
    position = positions.long()
    valid = (position >= 0) & (position < table.numel())
    scale = torch.where(
        valid, table[position.clamp(0, table.numel() - 1)], float("nan")
    )
    q_out = (_norm_reference(q, qw, eps).float() * scale[:, None]).bfloat16()
    return q_out, _norm_reference(k, kw, eps)


def _inputs(tokens, q_heads, k_heads, dim, strided, weight_dtypes, position_dtype):
    torch.manual_seed(811 + tokens + q_heads + dim)
    packed = torch.randn(
        (tokens + 1, (q_heads + k_heads + 1) * dim), dtype=torch.bfloat16, device="cuda"
    )
    q = packed[1:, : q_heads * dim]
    k = packed[1:, q_heads * dim : (q_heads + k_heads) * dim]
    if not strided:
        q, k = q.contiguous(), k.contiguous()
    qw, kw = (
        (1.0 + 0.1 * torch.randn(dim, device="cuda")).to(dtype)
        for dtype in weight_dtypes
    )
    # Nonzero storage offsets exercise pointers for both metadata inputs.
    table = torch.linspace(-1.25, 2.25, 98, device="cuda")[1:]
    positions = torch.randint(
        0, table.numel(), (tokens + 1,), device="cuda", dtype=position_dtype
    )[1:]
    return q, k, qw, kw, positions, table


def _check(actual, expected):
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0.016, rtol=0.008, equal_nan=True)


def test_staged_qk_ssmax_registration():
    registry = KernelRegistry.get()
    spec = registry.get_by_name("triton_staged_qk_rmsnorm_ssmax")
    assert spec is not None
    assert (spec.family, spec.mode) == ("layernorm", "staged_qk_rmsnorm_ssmax")
    assert registry.get_impl(spec.name) is staged_qk_rmsnorm_ssmax


@requires_gpu
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 128),
        (1, 3, 2, 64),
        (7, 5, 3, 96),
        (17, 9, 3, 128),
        (33, 3, 5, 256),
        (129, 5, 2, 31),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("preallocated", [False, True])
@pytest.mark.parametrize(
    "weight_dtypes,position_dtype",
    [
        ((torch.bfloat16, torch.float32), torch.int32),
        ((torch.float32, torch.bfloat16), torch.int64),
    ],
)
def test_staged_qk_ssmax_matches_reference(
    shape, strided, preallocated, weight_dtypes, position_dtype
):
    args = _inputs(*shape, strided, weight_dtypes, position_dtype)
    originals = [t.clone() for t in args]
    out = (
        tuple(
            torch.full(t.shape, float("nan"), device=t.device, dtype=t.dtype)
            for t in args[:2]
        )
        if preallocated
        else None
    )
    result = staged_qk_rmsnorm_ssmax(*args, 1e-6, out)
    if out is not None:
        assert result is out
    assert all(t.is_contiguous() for t in result)
    _check(result, _reference(*args, 1e-6))
    assert all(torch.equal(t, original) for t, original in zip(args, originals))
    repeated = staged_qk_rmsnorm_ssmax(*args, 1e-6, None)
    assert all(torch.equal(a, b) for a, b in zip(result, repeated))


@requires_gpu
@pytest.mark.parametrize("dim", [1, 2, 33, 127, 129, 1023, 1024])
def test_staged_qk_ssmax_head_dimension_boundaries(dim):
    args = _inputs(3, 3, 2, dim, True, (torch.float32, torch.float32), torch.int64)
    _check(staged_qk_rmsnorm_ssmax(*args, 1e-6, None), _reference(*args, 1e-6))


@requires_gpu
def test_staged_qk_ssmax_many_heads():
    # More than 65535 head groups must not depend on a two-dimensional grid limit.
    args = _inputs(1, 262143, 2, 1, False, (torch.float32, torch.float32), torch.int64)
    _check(staged_qk_rmsnorm_ssmax(*args, 1e-6, None), _reference(*args, 1e-6))


@requires_gpu
def test_staged_qk_ssmax_preserves_rounding_boundaries_exactly():
    args = _inputs(3, 2, 1, 128, False, (torch.float32, torch.float32), torch.int64)
    q, k, qw, kw, positions, table = args
    # Repeated pairs give exact mean squares, isolating casts from reduction order.
    q[:, ::2] = 1.0
    q[:, 1::2] = 3.0
    k.fill_(-1.0)
    expected = _reference(*args, 1e-6)
    result = staged_qk_rmsnorm_ssmax(*args, 1e-6, None)
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    ordinary = qk_rmsnorm(q, k, qw, kw, 1e-6, False)
    ordinary = ((ordinary[0].float() * table[positions, None]).bfloat16(), ordinary[1])
    assert any(torch.count_nonzero(a != b).item() for a, b in zip(expected, ordinary))


@requires_gpu
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dim", [64, 128])
def test_staged_qk_ssmax_invalid_positions_only_affect_q(position_dtype, dim):
    args = list(
        _inputs(7, 3, 5, dim, True, (torch.float32, torch.float32), position_dtype)
    )
    length = args[5].numel()
    maximum = torch.iinfo(position_dtype).max
    args[4].copy_(
        torch.tensor(
            [-1, 0, length - 1, length, maximum, torch.iinfo(position_dtype).min, 0],
            device="cuda",
            dtype=position_dtype,
        )
    )
    result = staged_qk_rmsnorm_ssmax(*args, 1e-6, None)
    _check(result, _reference(*args, 1e-6))
    assert torch.isnan(result[0][[0, 3, 4, 5]]).all()
    assert torch.isfinite(result[0][[1, 2, 6]]).all()
    assert torch.isfinite(result[1]).all()
    args[4].zero_()
    valid_result = staged_qk_rmsnorm_ssmax(*args, 1e-6, None)
    assert torch.equal(result[1], valid_result[1])


@requires_gpu
@pytest.mark.parametrize("scale", [0.0, -1.0, 1.003, float("nan"), float("inf")])
def test_staged_qk_ssmax_single_scale_table(scale):
    args = list(
        _inputs(3, 3, 2, 128, True, (torch.bfloat16, torch.bfloat16), torch.int64)
    )
    args[0].fill_(1.0)
    args[4].zero_()
    args[5] = torch.tensor([scale], device="cuda", dtype=torch.float32)
    _check(staged_qk_rmsnorm_ssmax(*args, 1e-6, None), _reference(*args, 1e-6))


@requires_gpu
@pytest.mark.parametrize("preallocated", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
def test_staged_qk_ssmax_graph_replay_reads_updated_inputs(preallocated, dim):
    args = _inputs(9, 3, 2, dim, True, (torch.float32, torch.float32), torch.int64)
    q, k, qw, kw, positions, table = args
    out = (
        tuple(torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in (q, k))
        if preallocated
        else None
    )
    staged_qk_rmsnorm_ssmax(*args, 1e-6, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = staged_qk_rmsnorm_ssmax(*args, 1e-6, out)
    if out is not None:
        assert result is out
    for position in (0, table.numel(), table.numel() - 1, 1 << 32):
        q.add_(0.125)
        k.neg_()
        qw.add_(0.015625)
        kw.sub_(0.0078125)
        positions.fill_(position)
        table.mul_(-0.9)
        for tensor in result:
            tensor.fill_(float("nan"))
        graph.replay()
        _check(result, _reference(*args, 1e-6))


@requires_gpu
def test_staged_qk_ssmax_broadcast_rows_and_output_storage_offsets():
    args = list(
        _inputs(1, 3, 2, 128, True, (torch.float32, torch.float32), torch.int64)
    )
    args[0], args[1] = (t.expand(7, -1) for t in args[:2])
    args[4] = torch.arange(7, device="cuda", dtype=torch.int32)
    storage = tuple(
        torch.full((9, t.shape[1]), 77.0, device=t.device, dtype=t.dtype)
        for t in args[:2]
    )
    out = tuple(t[1:8] for t in storage)
    assert staged_qk_rmsnorm_ssmax(*args, 1e-6, out) is out
    _check(out, _reference(*args, 1e-6))
    assert all((t[[0, 8]] == 77.0).all() for t in storage)


@requires_gpu
@pytest.mark.parametrize("preallocated", [False, True])
def test_staged_qk_ssmax_empty(preallocated):
    args = _inputs(0, 3, 2, 128, False, (torch.float32, torch.float32), torch.int64)
    out = tuple(torch.empty_like(t) for t in args[:2]) if preallocated else None
    result = staged_qk_rmsnorm_ssmax(*args, 1e-6, out)
    assert [t.shape for t in result] == [t.shape for t in args[:2]]
    assert all(t.numel() == 0 for t in result)
    if out is not None:
        assert result is out


def test_staged_qk_ssmax_rejects_cpu():
    q = torch.empty((2, 256), dtype=torch.bfloat16)
    k = torch.empty((2, 128), dtype=torch.bfloat16)
    weight = torch.empty(128)
    with pytest.raises(ValueError, match="one GPU"):
        staged_qk_rmsnorm_ssmax(
            q, k, weight, weight, torch.zeros(2, dtype=torch.int64), weight, 1e-6, None
        )


@requires_gpu
@pytest.mark.parametrize("eps", [0.0, -1.0, float("inf"), float("nan")])
def test_staged_qk_ssmax_rejects_invalid_epsilon(eps):
    args = _inputs(2, 2, 1, 128, False, (torch.float32, torch.float32), torch.int64)
    with pytest.raises(ValueError, match="finite and positive"):
        staged_qk_rmsnorm_ssmax(*args, eps, None)


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
        "weight_rank",
        "weight_shape",
        "weight_dtype",
        "weight_stride",
        "weight_device",
        "weight_empty",
        "weight_large",
        "position_shape",
        "position_rank",
        "position_dtype",
        "position_stride",
        "position_device",
        "table_rank",
        "table_dtype",
        "table_stride",
        "table_empty",
        "table_device",
    ],
)
def test_staged_qk_ssmax_rejects_invalid_inputs(invalid):
    args = list(
        _inputs(2, 2, 1, 128, False, (torch.float32, torch.float32), torch.int64)
    )
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
    elif invalid == "weight_rank":
        args[2] = args[2][None, :]
    elif invalid == "weight_shape":
        args[3] = args[3][:64]
    elif invalid == "weight_dtype":
        args[2] = args[2].half()
    elif invalid == "weight_stride":
        args[3] = torch.empty(256, device="cuda")[::2]
    elif invalid == "weight_device":
        args[2] = args[2].cpu()
    elif invalid == "weight_empty":
        args[2], args[3] = args[2][:0], args[3][:0]
    elif invalid == "weight_large":
        args[2], args[3] = (torch.empty(1025, device="cuda") for _ in range(2))
    elif invalid == "position_shape":
        args[4] = args[4][:1]
    elif invalid == "position_rank":
        args[4] = args[4][None, :]
    elif invalid == "position_dtype":
        args[4] = args[4].float()
    elif invalid == "position_stride":
        args[4] = torch.zeros(4, device="cuda", dtype=torch.int64)[::2]
    elif invalid == "position_device":
        args[4] = args[4].cpu()
    elif invalid == "table_rank":
        args[5] = args[5][None, :]
    elif invalid == "table_dtype":
        args[5] = args[5].bfloat16()
    elif invalid == "table_stride":
        args[5] = args[5][::2]
    elif invalid == "table_empty":
        args[5] = args[5][:0]
    elif invalid == "table_device":
        args[5] = args[5].cpu()
    with pytest.raises((ValueError, TypeError)):
        staged_qk_rmsnorm_ssmax(*args, 1e-6, None)


@requires_gpu
@pytest.mark.parametrize(
    "invalid",
    [
        "length",
        "type",
        "shape",
        "dtype",
        "device",
        "stride",
        "input_alias",
        "output_alias",
        "weight_alias",
        "table_alias",
        "disjoint_alias",
    ],
)
def test_staged_qk_ssmax_rejects_invalid_outputs(invalid):
    args = list(
        _inputs(2, 2, 2, 128, False, (torch.float32, torch.float32), torch.int64)
    )
    out = [torch.empty_like(t) for t in args[:2]]
    if invalid == "length":
        out.pop()
    elif invalid == "type":
        out[0] = None
    elif invalid == "shape":
        out[0] = out[0][:1]
    elif invalid == "dtype":
        out[0] = out[0].float()
    elif invalid == "device":
        out[0] = out[0].cpu()
    elif invalid == "stride":
        out[0] = torch.empty((2, 512), device="cuda", dtype=torch.bfloat16)[:, ::2]
    elif invalid == "input_alias":
        out[0] = args[0]
    elif invalid == "output_alias":
        out[1] = out[0]
    elif invalid == "weight_alias":
        args[2] = out[0].flatten()[:128]
    elif invalid == "table_alias":
        args[5] = out[0].view(torch.float32).flatten()
    elif invalid == "disjoint_alias":
        storage = torch.empty((4, 256), device="cuda", dtype=torch.bfloat16)
        args[0], out[0] = storage[:2], storage[2:]
    with pytest.raises(ValueError):
        staged_qk_rmsnorm_ssmax(*args, 1e-6, tuple(out))
