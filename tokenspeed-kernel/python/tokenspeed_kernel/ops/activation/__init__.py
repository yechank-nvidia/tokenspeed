# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Activation kernel entry points."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.flashinfer import (
    silu_and_mul as flashinfer_silu_and_mul,
)
from tokenspeed_kernel.ops.activation.triton import (
    add3,
    relu2,
)
from tokenspeed_kernel.ops.activation.triton import silu_and_mul as triton_silu_and_mul
from tokenspeed_kernel.ops.activation.triton import situ_and_mul as triton_situ_and_mul
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import error_fn
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """Apply SwiGLU through the platform implementation.

    Positive ``limit`` values use the portable Triton implementation because
    the CUDA implementation does not expose the checkpoint's clamp semantics.

    """
    if (
        limit is not None
        or current_platform().is_amd
        or flashinfer_silu_and_mul is error_fn
    ):
        return triton_silu_and_mul(x, out, limit=limit)
    return flashinfer_silu_and_mul(x, out, enable_pdl=pdl_enabled())


def situ_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """Apply SiTU through the portable Triton implementation."""

    return triton_situ_and_mul(
        x,
        out,
        beta=beta,
        linear_beta=linear_beta,
    )


def sigmoid_mul(
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    head_bias: torch.Tensor | None = None,
    # kernel options
    enable_pdl: bool,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Multiply ``x`` in place by ``sigmoid(gate)``, or by ``sigmoid(gate + bias)``.

    Args:
        x: Contiguous ``[num_tokens, hidden_dim]`` input, overwritten.
        gate: ``[num_tokens, hidden_dim]``, or ``[num_tokens, num_heads,
            head_dim]`` with a unit last stride (for example a column chunk of
            a packed projection).
        head_bias: Optional contiguous 1-D bias with one value per gate head,
            added to the gate before the sigmoid (feature ``head_bias``).
        enable_pdl: Launch with Programmatic Dependent Launch.
        override: Optional exact registered kernel override.
        solution: Optional restriction to a registered backend family.

    Returns:
        ``x``.
    """
    # Each option the call uses must be one the kernel implements.
    features = frozenset({"head_bias"}) if head_bias is not None else None
    signature = format_signature(x=dense_tensor_format(x.dtype))
    kernel = select_kernel(
        "activation",
        "sigmoid_mul",
        signature,
        features=features,
        solution=solution,
        override=override,
    )
    shape_params = {"shape": tuple(x.shape), "head_bias": head_bias is not None}
    ShapeCapture.get().record(
        "activation", "sigmoid_mul", kernel.name, x.dtype, shape_params
    )
    with kernel_scope(
        "activation",
        "sigmoid_mul",
        x.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(x, gate, head_bias=head_bias, enable_pdl=enable_pdl)


__all__ = [
    "add3",
    "relu2",
    "sigmoid_mul",
    "silu_and_mul",
    "situ_and_mul",
]
