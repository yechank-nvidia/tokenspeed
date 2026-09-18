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

"""Sampling kernel entry points."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["argmax", "try_gather_token_logprobs"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_OUT_DTYPES = (torch.int32, torch.int64)


def _supports_selected_token_logprobs(
    logits: torch.Tensor, tokens: torch.Tensor
) -> bool:
    """Inspect metadata only; never read indices or allocate/copy tensors."""
    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(tokens, torch.Tensor)
        or logits.layout != torch.strided
        or tokens.layout != torch.strided
        or not logits.is_cuda
        or not tokens.is_cuda
        or logits.device != tokens.device
        or logits.dtype != torch.float32
        or tokens.dtype != torch.int32
        or logits.shape != (1, 151936)
        or tokens.shape != (1,)
        or logits.stride() != (151936, 1)
        or tokens.stride() != (1,)
        or logits.storage_offset() != 0
        or tokens.storage_offset() != 0
        or logits.requires_grad
        or tokens.requires_grad
        or logits.is_neg()
        or logits.is_conj()
        or tokens.is_neg()
        or tokens.is_conj()
        or torch.is_autocast_enabled("cuda")
    ):
        return False
    platform = current_platform()
    if (
        not platform.is_nvidia
        or (platform.arch_version.major, platform.arch_version.minor) != (10, 0)
        or not pdl_enabled()
        or logits.device.index != torch.cuda.current_device()
    ):
        return False
    logits_ptr, tokens_ptr = logits.data_ptr(), tokens.data_ptr()
    if logits_ptr <= 0 or tokens_ptr <= 0 or logits_ptr % 16 or tokens_ptr % 16:
        return False
    return not (logits_ptr < tokens_ptr + 4 and tokens_ptr < logits_ptr + 151936 * 4)


def try_gather_token_logprobs(
    logits: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor | None:
    """Try raw-distribution selected-token logprobs without sampling transforms.

    Args:
        logits: Raw logits, without temperature/top-k/top-p preprocessing.
            The current specialization accepts FP32 [1, 151936] with exact
            stride (151936, 1) on the current SM100 CUDA device.
        tokens: Colocated INT32 [1] with stride (1,). Its index must be in
            [0, 151936); this caller invariant is device-asserted, never read
            on the host. Both inputs require zero storage offset, 16-byte
            pointer alignment, disjoint storage, no lazy negation/conjugation,
            no gradients, and no CUDA
            autocast. The platform PDL setting must be enabled.

    Returns:
        A fresh FP32 [1] result, or None for unsupported metadata/selection.
        None performs no tensor allocation, copy, synchronization or launch;
        callers retain their existing reference/fallback operation. Admitted
        calls allocate 8196 logical scratch bytes plus a separate fresh output,
        retain no global/backend workspace, and read live inputs on replay.
        Graphs own capture allocations; release graphs before their owners.
        NaNs/infinities are not sanitized and all FP32 value bits are admitted.
    """
    if not _supports_selected_token_logprobs(logits, tokens):
        return None
    try:
        # Registration is lazy so rejected/other-vendor inputs need no Triton.
        import tokenspeed_kernel.ops.sampling.triton.ordered_logprobs  # noqa: F401
    except ImportError:
        return None
    signature = format_signature(
        logits=dense_tensor_format(logits.dtype),
        tokens=dense_tensor_format(tokens.dtype),
    )
    try:
        kernel = select_kernel(
            "sampling",
            "gather_token_logprobs",
            signature,
            traits={"rows": 1, "vocab_size": 151936},
        )
    except NoKernelFoundError:
        return None
    return kernel(logits, tokens)


def _validate_argmax_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    if out.shape != (logits.shape[0],):
        raise ValueError(
            f"out must have shape (M,)={(logits.shape[0],)}, got {tuple(out.shape)}"
        )
    if out.dtype not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(f"out must be int32 or int64; got {out.dtype}")
    if out.device != logits.device:
        raise ValueError("out must be on the same device as logits")


def _argmax_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        _validate_argmax_out(logits, out)
    result = torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def argmax(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    solution: str | None = None,
    override: str | None = None,
) -> torch.Tensor:
    """Return row-wise argmax indices over the last logits dimension.

    Args:
        logits: Input logits with shape ``(M, N)``. The argmax is taken
            over the last dimension.
        out: Optional output buffer with shape ``(M,)`` and dtype int32 or
            int64 on the same device as ``logits``.
        solution: Optional kernel solution to force through normal selection.
        override: Optional exact kernel-name or solution override.

    NaN handling:
        Kernel-backed sampling paths treat NaNs as invalid candidates.
        Rows with at least one non-NaN value return the index of the
        maximum non-NaN value, with ties broken toward the lowest index.
        Rows with no valid non-NaN values return the ``-1`` sentinel.
        Unsupported inputs fall back to ``torch.argmax`` semantics.

    Returns:
        A tensor containing argmax indices for each row of ``logits``.
    """
    if (
        logits.dim() != 2
        or logits.shape[0] == 0
        or not logits.is_cuda
        or logits.dtype not in _SUPPORTED_DTYPES
    ):
        return _argmax_torch_fallback(logits, out=out)

    signature = format_signature(logits=dense_tensor_format(logits.dtype))
    try:
        kernel = select_kernel(
            "sampling",
            "argmax",
            signature,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return _argmax_torch_fallback(logits, out=out)

    shape_params = {
        "M": logits.shape[0],
        "N": logits.shape[1],
        "has_out": out is not None,
    }
    ShapeCapture.get().record(
        "sampling", "argmax", kernel.name, logits.dtype, shape_params
    )
    with kernel_scope(
        "sampling", "argmax", logits.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(logits, out=out)


# Backend registration (side-effect imports).
import tokenspeed_kernel.ops.sampling.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.sampling.gluon  # noqa: E402,F401
