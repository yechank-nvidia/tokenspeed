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

"""Disconnected vision-only RMSNorm prototype with original eager fallback."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec

import torch

TORCH_VERSION = "2.13.0+cu130"


def reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return the original vision RMSNorm, including each FP32 rounding boundary."""
    normalized = x.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + eps
    )
    return (weight.float() * normalized).to(x.dtype)


def supports_layout(x: torch.Tensor, weight: torch.Tensor, eps: float) -> bool:
    """Check metadata only; never copy values or synchronize a device."""
    return (
        x.ndim == 2
        and x.shape[1] == 1024
        and 16 <= x.shape[0] < 2**21
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.shape == (1024,)
        and x.device == weight.device
        and x.stride(1) == 1
        and x.stride(0) >= 1024
        and weight.is_contiguous()
        and type(eps) is float
        and eps == 1e-6
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
    )


@lru_cache(maxsize=1)
def _implementation():
    if find_spec("tvm_ffi") is None:
        return None
    try:
        from tokenspeed_kernel.ops.vision_rmsnorm.cute_dsl import apply
    except ModuleNotFoundError as exc:
        if exc.name in ("cutlass", "cuda") or (exc.name or "").startswith(
            ("cutlass.", "cuda.")
        ):
            return None
        raise
    return apply


def apply_vision_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Normalize vision activations with a bounded, pinned CUDA implementation.

    Args:
        x: Activations; the fast path accepts BF16 [N,1024], N >= 16, unit
            column stride, row gaps and nonzero storage offsets.
        weight: Contiguous BF16 [1024] parameter on x's device.
        eps: Original epsilon; only 1e-6 selects the custom kernel.

    Returns:
        A new tensor of x's shape and dtype. Unsupported metadata, autograd,
        torch versions, devices or absent optional dependencies use eager math.
        No shared language-backbone implementation is dispatched or modified.
    """
    if (
        not x.is_cuda
        or torch.version.hip is not None
        or torch.__version__ != TORCH_VERSION
        or not supports_layout(x, weight, eps)
        or torch.cuda.get_device_capability(x.device) != (10, 0)
    ):
        return reference(x, weight, eps)
    implementation = _implementation()
    return (
        reference(x, weight, eps)
        if implementation is None
        else implementation(x, weight)
    )
