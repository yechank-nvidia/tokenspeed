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

FP32 inputs take the same row-per-CTA shape: a single row through the full-row
reduction, and 2 to 32 rows through one CTA per weight row that streams the
weight once against every activation row, where the Torch FP32 path would run
a SIMT GEMM plus a split-K reduction.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import (
    FormatSignature,
    dense_tensor_format,
    format_signature,
)

__all__ = [
    "TORCH_DECODE_GEMV",
    "decode_gemv",
    "triton_rowcta_gemm_fp32",
    "triton_rowcta_gemv",
]

# Registered name of the portable ``gemm.decode_gemv`` leaf. It is the OFF
# target of the operator's override switch
# (``TOKENSPEED_KERNEL_OVERRIDE_GEMM_DECODE_GEMV=torch_decode_gemv``), and the
# measured BF16 route tests the selected leaf against it by name.
TORCH_DECODE_GEMV = "torch_decode_gemv"


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
    n = tl.program_id(0).to(tl.int64)
    value = _row_dot(x_ptr, w_ptr + n * K, K, BK)
    tl.store(out_ptr + n, value.to(out_ptr.dtype.element_ty))


@triton.jit
def _rowcta_multirow_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """One weight row against every activation row; ``BM`` pads ``M``."""
    n = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BM)
    row_mask = rows < M
    acc = tl.zeros([BM, BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        col_mask = offs < K
        wv = tl.load(w_ptr + n * K + offs, mask=col_mask, other=0.0).to(tl.float32)
        xv = tl.load(
            x_ptr + rows[:, None] * K + offs[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += xv * wv[None, :]
    values = tl.sum(acc, axis=1)
    tl.store(out_ptr + rows * N + n, values.to(out_ptr.dtype.element_ty), mask=row_mask)


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
_FP32_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.float32),
            weight=dense_tensor_format(torch.float32),
        )
    }
)
# The only two signatures the operator serves, keyed by dtype so ``_select``
# does not rebuild a FormatSignature on every eager call.
_SIGNATURE_BY_DTYPE: dict[torch.dtype, FormatSignature] = {
    torch.bfloat16: next(iter(_BF16_SIG)),
    torch.float32: next(iter(_FP32_SIG)),
}


@register_kernel(
    "gemm",
    "decode_gemv",
    name="triton_rowcta_gemv_fp32",
    solution="triton",
    signatures=_FP32_SIG,
    traits={"m": frozenset({1})},
    priority=Priority.SPECIALIZED,
)
@register_kernel(
    "gemm",
    "decode_gemv",
    name="triton_rowcta_gemv",
    solution="triton",
    signatures=_BF16_SIG,
    traits={
        "m": frozenset({1}),
        "n_min": frozenset({128}),
        "k_min": frozenset({128}),
    },
    priority=Priority.SPECIALIZED,
)
def triton_rowcta_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for ``M == 1`` decode activations.

    Args:
        x: ``[1, K]`` contiguous BF16 or FP32 activation row.
        weight: ``[N, K]`` contiguous weight in the same dtype as ``x``.
        out: optional ``[1, N]`` destination.

    Returns:
        ``[1, N]`` output in ``x``'s dtype.
    """
    assert x.shape[0] == 1 and x.stride(-1) == 1 and weight.stride(-1) == 1
    n, k = weight.shape
    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)
    if n == 0:
        return out
    if k == 0:
        return out.zero_()
    # Keep BF16's tiled accumulation. FP32 uses a full-row reduction for
    # ordinary projection widths, capping the tile for wider inputs.
    fp32 = x.dtype == torch.float32
    block_k = min(triton.next_power_of_2(k), 65536) if fp32 else 512
    _rowcta_gemv_kernel[(n,)](
        x.view(-1),
        weight,
        out.view(-1),
        K=k,
        BK=block_k,
        num_warps=8 if fp32 and block_k >= 4096 else 4,
        enable_fp_fusion=not fp32,
    )
    return out


@register_kernel(
    "gemm",
    "decode_gemv",
    name="triton_rowcta_gemm_fp32",
    solution="triton",
    signatures=_FP32_SIG,
    traits={"m": frozenset(range(2, 33))},
    priority=Priority.SPECIALIZED,
)
def triton_rowcta_gemm_fp32(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for a few FP32 decode rows, one CTA per weight row.

    The FP32 Torch path serves these shapes with a SIMT GEMM plus a split-K
    reduction; a narrow FP32 projection such as an expert router is memory
    bound on its weight, which this kernel streams once while the activation
    rows stay cache resident. Products and the tiled accumulation stay FP32
    without fused multiply-add, as for the single-row FP32 kernel.

    Args:
        x: ``[M, K]`` contiguous FP32 activations, ``2 <= M <= 32``.
        weight: ``[N, K]`` contiguous FP32 weight.
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` FP32 output.
    """
    m, k = x.shape
    n = weight.shape[0]
    assert 2 <= m <= 32 and x.stride(-1) == 1 and weight.stride(-1) == 1
    if out is None:
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    if n == 0:
        return out
    if k == 0:
        return out.zero_()
    block_m = triton.next_power_of_2(m)
    block_k = min(triton.next_power_of_2(k), 512)
    _rowcta_multirow_kernel[(n,)](
        x,
        weight,
        out,
        M=m,
        N=n,
        K=k,
        BM=block_m,
        BK=block_k,
        num_warps=8 if block_m >= 16 else 4,
        enable_fp_fusion=False,
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
        "n_align": frozenset({16}),
        "k_align": frozenset({128}),
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


# Portable leaf for both dtypes the operator serves: BF16 dense projections
# and the FP32 expert router. Registering the FP32 signature here gives every
# FP32 shape a candidate, so ``select_kernel`` resolves M >= 33 to this leaf
# by ranking instead of by an in-function fall-through.
@register_kernel(
    "gemm",
    "decode_gemv",
    name=TORCH_DECODE_GEMV,
    solution="torch",
    signatures=_BF16_SIG | _FP32_SIG,
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


def _select(
    m: int, n: int, k: int, on_cuda: bool, dtype: torch.dtype
) -> SelectedKernel:
    """Resolve the ``gemm.decode_gemv`` leaf for one ``[M, K] @ [N, K].T`` call.

    Non-CUDA inputs have no kernel platform and stay on the portable leaf.
    CUDA inputs go through :func:`select_kernel` with the call's dtype
    signature and ``(m, n, k)`` shape traits, so the operator's override
    (``TOKENSPEED_KERNEL_OVERRIDE_GEMM_DECODE_GEMV`` or
    :func:`~tokenspeed_kernel.selection.kernel_override`), verbose logging and
    selection listeners all apply, and the registry's selection cache holds the
    result per shape. Ranking is priority order under the same platform,
    signature and trait filters the specs declare; no ``gemm`` oracle exists,
    so the highest-priority admitted spec wins and the portable leaf is the
    lowest-priority candidate for every BF16 and FP32 shape.

    Args:
        m/n/k: the projection extents.
        on_cuda: whether ``x`` lives on a CUDA (or ROCm) device.
        dtype: the shared dtype of ``x`` and ``weight``.

    Returns:
        The :class:`SelectedKernel` to call as ``kernel(x, weight, out)``.

    Raises:
        NoKernelFoundError: for a CUDA dtype no ``gemm.decode_gemv`` kernel is
            registered for (only BF16 and FP32 are).
    """
    if not on_cuda:
        return SelectedKernel(name=TORCH_DECODE_GEMV, impl=torch_decode_gemv)
    signature = _SIGNATURE_BY_DTYPE.get(dtype)
    if signature is None:
        # Unregistered dtype: build its signature so ``select_kernel`` raises
        # ``NoKernelFoundError`` naming it.
        signature = format_signature(
            x=dense_tensor_format(dtype),
            weight=dense_tensor_format(dtype),
        )
    return select_kernel(
        "gemm",
        "decode_gemv",
        signature,
        traits={"m": m, "n": n, "k": k},
    )


def decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ weight.T`` with registry-selected decode kernels.

    Contiguous same-dtype CUDA inputs are dispatched through
    :func:`select_kernel` (see :func:`_select`): the shape traits keep the
    specialized kernels inside their validated envelope, everything else ranks
    down to the portable leaf, and ``TOKENSPEED_KERNEL_OVERRIDE_GEMM_DECODE_GEMV``
    forces one leaf by name for every shape. Layout mismatches and non-CUDA
    inputs take the portable leaf directly. Contiguous single-row FP32 GPU
    inputs use FP32 products and accumulation independently of Torch matmul
    precision.
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
    if (
        x.dtype != weight.dtype
        or x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        return torch_decode_gemv(x, weight, out)
    return _select(x.shape[0], weight.shape[0], weight.shape[1], x.is_cuda, x.dtype)(
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
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({True}),
        "is_cuda": frozenset({True}),
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
