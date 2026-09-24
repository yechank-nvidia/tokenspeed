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

"""Kept-group expert mask of grouped (group-limited) top-k routing: the ``moe.group_mask`` operator.

Grouped routers (the ``n_group`` / ``topk_group`` family) keep the highest-scoring
expert groups of each row and mask every other expert's biased score to ``-inf``
before the expert top-k. That mask is the operator
``("moe", "group_mask")``: :func:`moe_group_mask` takes the route's four
intermediates and returns a fresh FP32 ``[rows, experts]`` tensor equal to
``choices.masked_fill(~keep, -inf)`` where ``keep`` marks the experts of the
kept groups. Two kernels are registered:

* ``torch_group_mask`` (solution ``torch``, PORTABLE, any device): the three
  torch mask statements of the route verbatim plus the ``masked_fill`` they
  fed. Admits every shape; the ranked default and the production path.
* ``triton_group_mask_bits`` (solution ``triton``, REFERENCE, NVIDIA;
  :mod:`tokenspeed_kernel.ops.moe.triton.group_mask`): a uint32 bit-select
  that copies kept words and writes ``0xFF800000`` for excluded ones. Never
  auto-selected; reached only by name through the ordinary registry override
  (``--kernel-override moe.group_mask=triton_group_mask_bits``, mirrored to
  ``TOKENSPEED_KERNEL_OVERRIDE_MOE_GROUP_MASK``, or a
  :func:`tokenspeed_kernel.selection.kernel_override` context). It admits any
  positive row count of the example shape (256 experts in 8 groups with 4
  kept, contiguous FP32/int64 CUDA tensors) and raises on any other call, zero
  rows included: no in-function fallback.

The two kernels agree byte for byte (the bit-select does no floating-point
arithmetic), which the GPU byte-equality test checks.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.moe.triton.group_mask import TRITON_GROUP_MASK_BITS
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import (
    KernelRegistry,
    KernelSpec,
    Priority,
    register_kernel,
)
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "GROUP_MASK_FAMILY",
    "GROUP_MASK_MODE",
    "GROUP_MASK_OVERRIDE_ENV",
    "GROUP_MASK_SIGNATURES",
    "TORCH_GROUP_MASK",
    "TRITON_GROUP_MASK_BITS",
    "group_mask_traits",
    "moe_group_mask",
    "torch_group_mask",
]

GROUP_MASK_FAMILY = "moe"
GROUP_MASK_MODE = "group_mask"
# Same derivation as ``select_kernel``'s environment override key.
GROUP_MASK_OVERRIDE_ENV = (
    f"TOKENSPEED_KERNEL_OVERRIDE_{GROUP_MASK_FAMILY.upper()}"
    f"_{GROUP_MASK_MODE.upper()}"
)
# Registered name of the production default (the ON target of the switch).
TORCH_GROUP_MASK = "torch_group_mask"
# The one dtype pair the route produces: FP32 biased scores (``scores +
# correction_bias`` after ``.float().sigmoid()``) and int64 ``topk(...).indices``.
GROUP_MASK_SIGNATURES = frozenset(
    {
        format_signature(
            choices=dense_tensor_format(torch.float32),
            group_ids=dense_tensor_format(torch.int64),
        )
    }
)


def group_mask_traits(
    choices: torch.Tensor,
    group_ids: torch.Tensor,
    group_scores: torch.Tensor,
    grouped: torch.Tensor,
) -> dict[str, int | bool]:
    """The trait dict :func:`moe_group_mask` selects with (metadata only).

    ``rows``/``experts`` come from ``choices``, ``num_groups`` from
    ``group_scores``, ``topk_groups`` from ``group_ids`` and ``contiguous`` is
    whether all four tensors are contiguous. ``experts``, ``num_groups``,
    ``topk_groups`` and ``contiguous`` are the values the
    ``triton_group_mask_bits`` spec constrains; ``rows`` is carried for the
    shape capture and the kernel scope (no registered spec constrains it since
    the Triton kernel admits every positive row count, and ``spec_matches_traits``
    ignores traits a spec does not declare); ``torch_group_mask`` declares no
    traits and admits every dict.
    """
    return {
        "rows": int(choices.shape[0]),
        "experts": int(choices.shape[1]),
        "num_groups": int(group_scores.shape[1]),
        "topk_groups": int(group_ids.shape[1]),
        "contiguous": all(
            value.is_contiguous()
            for value in (choices, group_ids, group_scores, grouped)
        ),
    }


def _operator_spec(kernel_name: str) -> KernelSpec:
    """The registry spec of ``kernel_name`` if it belongs to ``moe.group_mask``.

    Raises:
        ValueError: ``kernel_name`` is unregistered or registered under another
            operator. ``select_kernel`` resolves an override by name alone, so
            the operator check is made here before the kernel is called with
            the four route tensors.
    """
    spec = KernelRegistry.get().get_by_name(kernel_name)
    if spec is None or (spec.family, spec.mode) != (
        GROUP_MASK_FAMILY,
        GROUP_MASK_MODE,
    ):
        raise ValueError(
            f"{kernel_name!r} is not registered under "
            f"{GROUP_MASK_FAMILY}.{GROUP_MASK_MODE}"
        )
    return spec


def moe_group_mask(
    choices: torch.Tensor,
    group_ids: torch.Tensor,
    group_scores: torch.Tensor,
    grouped: torch.Tensor,
    *,
    solution: str | None,
    override: str | None,
) -> torch.Tensor:
    """Mask ``choices`` to the kept expert groups through ``moe.group_mask``.

    Args:
        choices: ``[rows, experts]`` FP32 biased scores (``scores + correction_bias``).
        group_ids: ``[rows, topk_groups]`` int64 kept-group indices from
            ``group_scores.topk(topk_groups).indices``; values in ``[0, num_groups)``
            are a producer precondition (never read on the host).
        group_scores: ``[rows, num_groups]`` FP32 group scores (shape/device witness of
            the torch statements' ``zeros_like``).
        grouped: ``[rows, num_groups, experts // num_groups]`` view of ``choices``
            (the ``expand_as`` target of the torch statements).
        solution: Restrict ranked selection to one solution (``"torch"``/``"triton"``)
            or ``None`` for any.
        override: Exact kernel name to force, or ``None`` for ranked selection; the
            ``kernel_override`` context and ``TOKENSPEED_KERNEL_OVERRIDE_MOE_GROUP_MASK``
            outrank it (see ``select_kernel``).

    Returns:
        A fresh ``[rows, experts]`` FP32 tensor equal to
        ``choices.masked_fill(~keep, -inf)`` where ``keep`` marks the experts of the kept
        groups; inputs are not modified.

    Raises:
        ValueError: The tensors do not form the route's intermediates, or the
            resolved kernel is not registered under ``moe.group_mask``.
        NoKernelFoundError: No registered kernel serves the dtype pair.
    """
    if choices.ndim != 2:
        raise ValueError("choices must have shape [rows, experts]")
    rows, experts = (int(size) for size in choices.shape)
    if group_scores.ndim != 2 or group_scores.shape[0] != rows:
        raise ValueError("group_scores must have shape [rows, num_groups]")
    num_groups = int(group_scores.shape[1])
    if num_groups == 0 or experts % num_groups != 0:
        raise ValueError(
            f"experts ({experts}) must be a positive multiple of num_groups "
            f"({num_groups})"
        )
    if (
        group_ids.ndim != 2
        or group_ids.shape[0] != rows
        or not 0 < group_ids.shape[1] <= num_groups
    ):
        raise ValueError(
            "group_ids must have shape [rows, topk_groups] with "
            f"0 < topk_groups <= num_groups ({num_groups})"
        )
    if tuple(grouped.shape) != (rows, num_groups, experts // num_groups):
        raise ValueError(
            "grouped must have shape [rows, num_groups, experts // num_groups] = "
            f"{(rows, num_groups, experts // num_groups)}, got {tuple(grouped.shape)}"
        )
    if not (
        choices.device == group_ids.device == group_scores.device == grouped.device
    ):
        raise ValueError(
            "choices, group_ids, group_scores and grouped must share one device"
        )
    # An unregistered dtype pair makes ``select_kernel`` raise ``NoKernelFoundError``
    # naming the signature.
    signature = format_signature(
        choices=dense_tensor_format(choices.dtype),
        group_ids=dense_tensor_format(group_ids.dtype),
    )
    traits = group_mask_traits(choices, group_ids, group_scores, grouped)
    kernel = select_kernel(
        GROUP_MASK_FAMILY,
        GROUP_MASK_MODE,
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    _operator_spec(kernel.name)
    shape_params = {
        "rows": traits["rows"],
        "experts": traits["experts"],
        "num_groups": traits["num_groups"],
        "topk_groups": traits["topk_groups"],
    }
    ShapeCapture.get().record(
        GROUP_MASK_FAMILY,
        GROUP_MASK_MODE,
        kernel.name,
        choices.dtype,
        shape_params,
    )
    with kernel_scope(
        GROUP_MASK_FAMILY,
        GROUP_MASK_MODE,
        choices.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(choices, group_ids, group_scores, grouped)


@register_kernel(
    "moe",
    "group_mask",
    name=TORCH_GROUP_MASK,
    solution="torch",
    signatures=GROUP_MASK_SIGNATURES,
    # Admits every shape: the statements are general. While this is the only
    # non-REFERENCE registration under the operator, ranked selection lands here
    # for every call, so the route's behaviour is byte-identical to the inlined
    # statements it replaced.
    traits={},
    priority=Priority.PORTABLE,
)
def torch_group_mask(
    choices: torch.Tensor,
    group_ids: torch.Tensor,
    group_scores: torch.Tensor,
    grouped: torch.Tensor,
) -> torch.Tensor:
    """The three torch statements of the grouped-routing mask plus the
    ``masked_fill`` they fed; runs on any device."""
    keep_groups = torch.zeros_like(group_scores, dtype=torch.bool)
    keep_groups.scatter_(1, group_ids, True)
    keep = keep_groups.unsqueeze(-1).expand_as(grouped).reshape_as(choices)
    return choices.masked_fill(~keep, float("-inf"))
