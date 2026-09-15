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

"""Row-per-CTA M=1 bf16 GEMV.

Streams each weight row through one CTA (whole row in a single masked load,
dot against the L2-resident activation, one store). In the L2-cold regime a
decode step actually runs in, this beats every cublasLt tactic on the K3
skinny shapes by 13-14% (measured: 6288x7168 15.8us vs 18.1; 3584x7168
10.2us vs 11.9) while staying ~10% off the pure read+sum ceiling.
Deterministic by construction: one fixed-order reduction per output, no
split-K phase.
"""

from __future__ import annotations

import functools

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["decode_gemv", "triton_fp32_decode_gemv", "triton_rowcta_gemv"]


@triton.jit
def _rowcta_gemv_add3_kernel(
    x_ptr,
    w_ptr,
    a_ptr,
    c_ptr,
    out_ptr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    """Row dot-product with a fused two-addend epilogue:
    ``out[n] = a[n] + x . w[n] + c[n]`` (the MoE residual accumulate rides
    the up-projection store; a/c row strides support lane column slices)."""
    n = tl.program_id(0)
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + n * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wv * xv
    av = tl.load(a_ptr + n).to(tl.float32)
    cv = tl.load(c_ptr + n).to(tl.float32)
    tl.store(
        out_ptr + n,
        (av + tl.sum(acc) + cv).to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _row_dot(x_ptr, w_ptr, K: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wv * xv
    return tl.sum(acc)


@triton.jit
def _rowcta_gemv_kernel(x_ptr, w_ptr, out_ptr, K: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0)
    value = _row_dot(x_ptr, w_ptr + n * K, K, BK)
    tl.store(out_ptr + n, value.to(out_ptr.dtype.element_ty))


@triton.jit
def _grouped_rowcta_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K: tl.constexpr,
    BK: tl.constexpr,
    X_GROUP_STRIDE: tl.constexpr,
    W_GROUP_STRIDE: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    OUT_GROUP_STRIDE: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    value = _row_dot(
        x_ptr + group * X_GROUP_STRIDE,
        w_ptr + group * W_GROUP_STRIDE + n * W_ROW_STRIDE,
        K,
        BK,
    )
    tl.store(out_ptr + group * OUT_GROUP_STRIDE + n, value.to(out_ptr.dtype.element_ty))


# Registry dispatch: rowcta owns M == 1 while torch handles other shapes.
_BF16_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.bfloat16),
        )
    }
)


@register_kernel(
    "gemm",
    "decode_gemv",
    name="triton_rowcta_gemv",
    solution="triton",
    signatures=_BF16_SIG,
    traits={
        "m": frozenset({1}),
        "n_min_128": frozenset({True}),
        "k_min_128": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
)
def triton_rowcta_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for ``M == 1`` decode activations.

    Args:
        x: ``[1, K]`` contiguous bf16 activation row.
        weight: ``[N, K]`` contiguous bf16 weight.
        out: optional ``[1, N]`` destination.

    Returns:
        ``[1, N]`` output in ``x``'s dtype.
    """
    assert x.shape[0] == 1 and x.stride(-1) == 1 and weight.stride(-1) == 1
    n, k = weight.shape
    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)
    # BK=512 (4 fp32 accumulator regs/thread): standalone parity, and aux-stream kernels co-reside instead of stalling behind the GEMV wave.
    _rowcta_gemv_kernel[(n,)](
        x.view(-1),
        weight,
        out.view(-1),
        K=k,
        BK=512,
        num_warps=4,
    )
    return out


@register_kernel(
    "gemm",
    "decode_gemv",
    name="gluon_wmma_dense_gemv_gfx1250",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 5),
        max_arch_version=ArchVersion(12, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_BF16_SIG,
    traits={
        "m": frozenset(range(2, 33)),
        "k_align_128": frozenset({True}),
        "n_align_16": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
)
def gluon_wmma_dense_gemv_gfx1250(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for small-M decode activations on CDNA5.

    Args:
        x: ``[M, K]`` contiguous bf16 activation, K a multiple of 128.
        weight: ``[N, K]`` contiguous bf16 weight, N a multiple of 16.
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` output in ``x``'s dtype.
    """
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        gluon_wmma_tdm_dense_gfx1250,
    )

    return gluon_wmma_tdm_dense_gfx1250(x, weight, out=out)


@register_kernel(
    "gemm",
    "decode_gemv",
    name="torch_decode_gemv",
    solution="torch",
    signatures=_BF16_SIG,
    traits={},
    priority=Priority.PORTABLE,
)
def torch_decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return torch.mm(x, weight.t(), out=out)
    return x @ weight.t()


@functools.lru_cache(maxsize=64)
def _select(m: int, n: int, k: int, on_cuda: bool):
    if not on_cuda:
        return torch_decode_gemv
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import (
        spec_matches_shape_traits,
        spec_matches_traits,
    )

    reg = KernelRegistry.get()
    # platform= makes the registry honor each spec's capability gate; without
    # it an arch-gated spec (the measured sm103 route) would match anywhere.
    for spec in reg.get_for_operator(
        "gemm", "decode_gemv", platform=current_platform()
    ):
        if spec_matches_traits(spec, {"m": m}) and spec_matches_shape_traits(
            spec, {"N": n, "K": k}
        ):
            return reg.get_impl(spec.name)
    return torch_decode_gemv


def decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ weight.T`` with registry-selected decode kernels.

    Selection is cached per (M, N, K, device kind); the shape traits keep
    the specialized kernels inside their validated envelope and everything
    else routes to the portable fallback.
    """
    expected = (x.shape[0], weight.shape[0])
    if out is not None:
        if (
            tuple(out.shape) != expected
            or out.dtype != x.dtype
            or out.device != x.device
            or out.stride(-1) != 1
        ):
            raise ValueError(f"out must match x and have shape {expected}")
        if not out.is_contiguous():
            return torch_decode_gemv(x, weight, out)
    return _select(x.shape[0], weight.shape[0], weight.shape[1], x.is_cuda)(
        x, weight, out
    )


def rowcta_gemv_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``a + x @ weight.T + c`` for ``M == 1`` (fused MoE residual epilogue).

    Args:
        x: ``[1, K]`` bf16 latent row; weight: ``[N, K]``.
        a/c: ``[1, N]`` addends (``c`` may be a wider-lane column slice --
            only unit inner stride is required).

    Returns:
        ``[1, N]`` prefix row.
    """
    assert x.shape[0] == 1 and a.shape == (1, weight.shape[0])
    assert a.stride(1) == 1 and c.stride(1) == 1 and c.shape[1] == weight.shape[0]
    n, k = weight.shape
    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)
    _rowcta_gemv_add3_kernel[(n,)](
        x.view(-1),
        weight,
        a,
        c,
        out,
        K=k,
        BK=512,
        num_warps=4,
    )
    return out


@register_kernel(
    "gemm",
    "grouped_bf16_projection",
    name="grouped_bf16_projection_rowcta",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0)
    ),
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={
        "batch": frozenset({2}),
        "m": frozenset({1}),
        "n": frozenset({1024}),
        "k": frozenset({4096}),
        "is_cuda": frozenset({True}),
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
)
def grouped_bf16_projection_rowcta(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Single-token grouped projection; the public API validates the layout."""
    groups, rows, dim = weight.shape
    if out is None:
        out = torch.empty((1, groups, rows), dtype=x.dtype, device=x.device)
    _grouped_rowcta_gemv_kernel[(rows, groups)](
        x,
        weight,
        out,
        K=dim,
        BK=4096,
        X_GROUP_STRIDE=x.stride(1),
        W_GROUP_STRIDE=weight.stride(0),
        W_ROW_STRIDE=weight.stride(1),
        OUT_GROUP_STRIDE=out.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _fp32_decode_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One FP32 weight row per CTA for bandwidth-bound decode projection."""
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    weight = tl.load(
        weight_ptr + row * K + offsets,
        mask=mask,
        other=0.0,
    )
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.sum(x * weight))


@register_kernel(
    "gemm",
    "fp32_decode_gemv",
    name="triton_fp32_decode_gemv",
    solution="triton",
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.float32),
                weight=dense_tensor_format(torch.float32),
            )
        }
    ),
    traits={"m": frozenset({1})},
    priority=Priority.SPECIALIZED,
)
def triton_fp32_decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute an FP32 ``x @ weight.T`` for a single decode row.

    The kernel assigns one contiguous weight row to each CTA and performs a
    fixed-order FP32 reduction. It is intended for memory-bound, large-output
    projections. Multi-row inputs remain GEMM
    workloads and should use :func:`torch.mm` instead.

    Args:
        x: Contiguous FP32 activation shaped ``[1, K]``, with ``1 <= K <= 65536``.
        weight: Contiguous FP32 weight shaped ``[N, K]``.
        out: Contiguous FP32 destination shaped ``[1, N]``, or ``None`` to
            allocate one. Nonempty destinations must not share input storage.

    Returns:
        The FP32 output shaped ``[1, N]``.
    """
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("x and weight must be rank-2 tensors")
    if x.shape[0] != 1 or x.shape[1] != weight.shape[1]:
        raise ValueError(
            "fp32_decode_gemv requires x shaped [1, K] and weight shaped [N, K]"
        )
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("fp32_decode_gemv requires FP32 x and weight")
    if x.device != weight.device or not x.is_cuda:
        raise ValueError("x and weight must be on the same GPU")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")

    n, k = weight.shape
    if k == 0:
        raise ValueError("K must be non-zero")
    if k > 65536:
        raise ValueError("K is too large for the row-CTA reduction")
    if out is None:
        out = torch.empty((1, n), device=x.device, dtype=torch.float32)
    elif (
        out.shape != (1, n)
        or out.dtype != torch.float32
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous FP32 with shape {(1, n)}")
    if n == 0:
        return out
    if out.untyped_storage().data_ptr() in {
        x.untyped_storage().data_ptr(),
        weight.untyped_storage().data_ptr(),
    }:
        raise ValueError("out must not alias x or weight storage")

    block_k = triton.next_power_of_2(k)
    _fp32_decode_gemv_kernel[(n,)](
        x,
        weight,
        out,
        K=k,
        BLOCK_K=block_k,
        num_warps=8 if block_k >= 4096 else 4,
    )
    return out
