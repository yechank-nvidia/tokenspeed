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
from tokenspeed_kernel.ops.gemm import fp32_decode_gemv
from tokenspeed_kernel.registry import KernelRegistry

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP32 GEMV requires a CUDA or ROCm GPU"
)


def _inputs(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(71 + n + k)
    x = torch.randn((1, k), device="cuda", dtype=torch.float32, generator=generator)
    weight = torch.randn(
        (n, k), device="cuda", dtype=torch.float32, generator=generator
    )
    return x, weight


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.double() @ weight.double().T).float()


def test_fp32_decode_gemv_registration() -> None:
    spec = KernelRegistry.get().get_by_name("triton_fp32_decode_gemv")
    assert spec is not None
    assert spec.family == "gemm"
    assert spec.mode == "fp32_decode_gemv"
    assert KernelRegistry.get().get_impl(spec.name) is fp32_decode_gemv


@requires_gpu
@pytest.mark.parametrize(
    ("n", "k"),
    [
        (1, 1),
        (7, 3),
        (19, 127),
        (67, 128),
        (257, 1025),
        (1024, 2048),
        (257, 8192),
        (3, 65536),
    ],
)
@pytest.mark.parametrize("preallocated", [False, True])
def test_fp32_decode_gemv_matches_reference(n: int, k: int, preallocated: bool) -> None:
    x, weight = _inputs(n, k)
    original_x, original_weight = x.clone(), weight.clone()
    out = torch.full((1, n), float("nan"), device="cuda") if preallocated else None
    result = fp32_decode_gemv(x, weight, out)
    if out is not None:
        assert result is out
    assert result.shape == (1, n)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, _reference(x, weight), rtol=2e-5, atol=2e-4)
    assert torch.equal(x, original_x)
    assert torch.equal(weight, original_weight)
    assert torch.equal(result, fp32_decode_gemv(x, weight, None))


@requires_gpu
def test_fp32_decode_gemv_contiguous_storage_offsets() -> None:
    x_storage, weight_storage = _inputs(14, 258)
    x = x_storage[:, 1:]
    # Drop a whole row before reshaping to preserve a nonzero contiguous offset.
    weight = weight_storage.view(-1)[258 : 258 + 13 * 257].view(13, 257)
    out_storage = torch.full((1, 14), 77.0, dtype=torch.float32, device="cuda")
    out = out_storage[:, 1:]
    result = fp32_decode_gemv(x, weight, out)
    assert result is out
    assert out_storage[0, 0].item() == 77.0
    torch.testing.assert_close(result, _reference(x, weight), rtol=2e-5, atol=2e-4)


@requires_gpu
@pytest.mark.parametrize("preallocated", [False, True])
def test_fp32_decode_gemv_graph_replay_reads_updated_inputs(preallocated: bool) -> None:
    x, weight = _inputs(129, 1025)
    out = (
        torch.empty((1, 129), dtype=torch.float32, device="cuda")
        if preallocated
        else None
    )
    fp32_decode_gemv(x, weight, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fp32_decode_gemv(x, weight, out)
    if out is not None:
        assert result is out
    for delta in (0.0, 0.125):
        x.add_(delta)
        graph.replay()
        torch.testing.assert_close(result, _reference(x, weight), rtol=2e-5, atol=2e-4)


@requires_gpu
@pytest.mark.parametrize("preallocated", [False, True])
def test_fp32_decode_gemv_empty_output(preallocated: bool) -> None:
    x, weight = _inputs(0, 17)
    out = (
        torch.empty((1, 0), dtype=torch.float32, device="cuda")
        if preallocated
        else None
    )
    result = fp32_decode_gemv(x, weight, out)
    assert result.shape == (1, 0)
    assert result.dtype == torch.float32
    if out is not None:
        assert result is out


@pytest.mark.parametrize(
    ("x_shape", "w_shape", "dtype", "error", "message"),
    [
        ((8,), (4, 8), torch.float32, ValueError, "rank-2"),
        ((1, 8), (8,), torch.float32, ValueError, "rank-2"),
        ((2, 8), (4, 8), torch.float32, ValueError, "requires x shaped"),
        ((0, 8), (4, 8), torch.float32, ValueError, "requires x shaped"),
        ((1, 8), (4, 9), torch.float32, ValueError, "requires x shaped"),
        ((1, 8), (4, 8), torch.bfloat16, TypeError, "requires FP32"),
        ((1, 8), (4, 8), torch.float64, TypeError, "requires FP32"),
        ((1, 8), (4, 8), torch.float32, ValueError, "same GPU"),
    ],
)
def test_fp32_decode_gemv_rejects_unsupported_inputs(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    dtype: torch.dtype,
    error: type[Exception],
    message: str,
) -> None:
    x = torch.empty(x_shape, dtype=dtype)
    weight = torch.empty(w_shape, dtype=dtype)
    with pytest.raises(error, match=message):
        fp32_decode_gemv(x, weight, None)


@requires_gpu
@pytest.mark.parametrize("k", [0, 65537])
@pytest.mark.parametrize("n", [0, 3])
def test_fp32_decode_gemv_rejects_invalid_reduction_width(n: int, k: int) -> None:
    x = torch.empty((1, k), dtype=torch.float32, device="cuda")
    weight = torch.empty((n, k), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="K "):
        fp32_decode_gemv(x, weight, None)


@requires_gpu
@pytest.mark.parametrize(
    "invalid",
    [
        "x_strides",
        "weight_strides",
        "weight_dtype",
        "weight_device",
        "out_shape",
        "out_dtype",
        "out_device",
        "out_strides",
    ],
)
def test_fp32_decode_gemv_rejects_invalid_buffers(invalid: str) -> None:
    x, weight = _inputs(7, 16)
    out = torch.empty((1, 7), dtype=torch.float32, device="cuda")
    if invalid == "x_strides":
        x = torch.empty((1, 32), device="cuda")[:, ::2]
    elif invalid == "weight_strides":
        weight = torch.empty((7, 32), device="cuda")[:, ::2]
    elif invalid == "weight_dtype":
        weight = weight.to(torch.bfloat16)
    elif invalid == "weight_device":
        weight = weight.cpu()
    elif invalid == "out_shape":
        out = torch.empty((7,), device="cuda")
    elif invalid == "out_dtype":
        out = out.to(torch.bfloat16)
    elif invalid == "out_device":
        out = out.cpu()
    elif invalid == "out_strides":
        out = torch.empty((1, 14), device="cuda")[:, ::2]
    with pytest.raises((TypeError, ValueError)):
        fp32_decode_gemv(x, weight, out)


@requires_gpu
@pytest.mark.parametrize("alias", ["x", "weight", "disjoint_storage_view"])
def test_fp32_decode_gemv_rejects_output_alias(alias: str) -> None:
    x, weight = _inputs(16, 16)
    if alias == "x":
        out = x
    elif alias == "weight":
        out = weight[:1]
    else:
        storage = torch.empty((2, 16), dtype=torch.float32, device="cuda")
        x, out = storage[:1], storage[1:]
    with pytest.raises(ValueError, match="must not alias"):
        fp32_decode_gemv(x, weight, out)
