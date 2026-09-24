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

"""Triton uint32 bit-select for ``moe.group_mask`` (``triton_group_mask_bits``).

One program per row copies the row's 256 FP32 ``choices`` words as uint32 and
replaces the words of the groups not listed in ``group_ids`` by ``0xFF800000``,
the bit pattern of FP32 ``-inf``. No floating-point arithmetic runs: kept words
(NaN payloads, ``-0.0``) are copied bit for bit, so the output equals the torch
``masked_fill`` of :func:`tokenspeed_kernel.ops.moe.group_mask.torch_group_mask`
byte for byte.

Its select statement is the shared grouped top-k select of ``minimax_topk.py``
with the ``-inf`` operand as the uint32 word. Correctness is established by the
byte-equality tests, not by a binary hash.

Registered in the REFERENCE band, so ranked selection never picks it while
``torch_group_mask`` is registered; it is reached only by name (``override=``,
``kernel_override("moe", "group_mask", ...)``,
``TOKENSPEED_KERNEL_OVERRIDE_MOE_GROUP_MASK``). Its admission -- any positive
row count, 256 experts in 8 groups with 4 kept, contiguous FP32/int64 CUDA
tensors on one device -- is declared twice on purpose: as spec traits, so
ranked selection and callers that re-check traits reject it, and as the host guard
:func:`group_mask_bits_rejection`, so a by-name override on a rejected shape
raises instead of falling back. Zero rows are rejected by the host guard alone
(an empty grid is never launched; ``torch_group_mask`` serves that call).

``rows`` is only the grid size (one program per row; it is not a kernel
argument and does not change the compiled binary), so every positive row count
is admitted; the GPU test (``test/nvidia/ops/moe/test_group_mask_bits.py``)
runs rows 1..4096 including the non-powers of two 3, 5, 7 and 33.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "TRITON_GROUP_MASK_BITS",
    "group_mask_bits_rejection",
    "triton_group_mask_bits",
]

# Registered name of the bit-select kernel; the OFF target of the
# ``moe.group_mask`` switch.
TRITON_GROUP_MASK_BITS = "triton_group_mask_bits"
# Compile-time constants of the example shape: 256 experts in 8 groups, 4 kept.
# ``rows`` is the grid size only (no ``rows`` constant or trait: see the module doc).
_BLOCK_E = 256
_NUM_GROUPS = 8
_TOPK_GROUP = 4
# Launch options, pinned explicitly.
_LAUNCH_OPTIONS = dict(
    num_warps=4,
    enable_fp_fusion=False,
    enable_reflect_ftz=False,
    launch_pdl=False,
)


@triton.jit
def _group_mask_bits(
    choices_ptr,
    group_ids_ptr,
    output_ptr,
    BLOCK_E: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
):
    """Copy one row of FP32 ``choices`` as uint32 words, replacing the words of
    excluded groups by ``0xFF800000`` (the bit pattern of FP32 ``-inf``). No
    floating-point arithmetic: kept words are copied bit for bit."""
    token_id = tl.program_id(0)
    offsets = token_id * BLOCK_E + tl.arange(0, BLOCK_E)
    choice_scores = tl.load(choices_ptr.to(tl.pointer_type(tl.uint32)) + offsets)
    group_choices = tl.reshape(choice_scores, (NUM_GROUPS, BLOCK_E // NUM_GROUPS))
    groups = tl.arange(0, NUM_GROUPS)
    kept = tl.full((NUM_GROUPS,), False, tl.int1)
    for index in tl.static_range(TOPK_GROUP):
        selected = tl.load(group_ids_ptr + token_id * TOPK_GROUP + index)
        kept |= groups == selected
    choice_scores = tl.reshape(
        tl.where(kept[:, None], group_choices, tl.full((), 0xFF800000, tl.uint32)),
        (BLOCK_E,),
    )
    tl.store(output_ptr.to(tl.pointer_type(tl.uint32)) + offsets, choice_scores)


def group_mask_bits_rejection(
    choices: torch.Tensor,
    group_ids: torch.Tensor,
    group_scores: torch.Tensor,
    grouped: torch.Tensor,
) -> str:
    """Why ``triton_group_mask_bits`` cannot serve these tensors; empty when admitted.

    Metadata only (no device read): the row count must be positive (an empty
    grid is not launched); ``choices`` FP32 ``[rows, 256]``, ``group_ids`` int64
    ``[rows, 4]``, ``group_scores`` FP32 ``[rows, 8]`` and ``grouped`` FP32
    ``[rows, 8, 32]``, each contiguous (the kernel's pointer arithmetic has no
    stride arguments); all four CUDA tensors on one device. The reason names the
    first rejecting attribute; device clauses come last so that shape and layout
    findings are reported on any device.

    Args:
        choices: Biased FP32 scores the kernel reads as uint32 words.
        group_ids: Kept-group indices the kernel reads through an int64 pointer.
        group_scores: Group scores; admission witness only, never launched.
        grouped: Grouped view of ``choices``; admission witness only.

    Returns:
        The rejection reason, or ``""`` when the kernel admits the call.
    """
    if choices.ndim != 2:
        return f"choices must be 2-D, got {choices.ndim}-D"
    rows = int(choices.shape[0])
    if rows == 0:
        return "rows=0: an empty grid is not launched (rows must be positive)"
    layouts = (
        ("choices", choices, torch.float32, (rows, _BLOCK_E)),
        ("group_ids", group_ids, torch.int64, (rows, _TOPK_GROUP)),
        ("group_scores", group_scores, torch.float32, (rows, _NUM_GROUPS)),
        (
            "grouped",
            grouped,
            torch.float32,
            (rows, _NUM_GROUPS, _BLOCK_E // _NUM_GROUPS),
        ),
    )
    for name, value, dtype, shape in layouts:
        if value.dtype != dtype:
            return f"{name} must be {dtype}, got {value.dtype}"
        if tuple(value.shape) != shape:
            return f"{name} must have shape {shape}, got {tuple(value.shape)}"
        if not value.is_contiguous():
            return f"{name} must be contiguous, got strides {tuple(value.stride())}"
    for name, value, _, _ in layouts:
        if not value.is_cuda:
            return f"{name} must be a CUDA tensor, got device {value.device}"
    if not (
        choices.device == group_ids.device == group_scores.device == grouped.device
    ):
        return "choices, group_ids, group_scores and grouped must share one device"
    return ""


@register_kernel(
    "moe",
    "group_mask",
    name=TRITON_GROUP_MASK_BITS,
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    # The same FP32/int64 pair as ``torch_group_mask`` (ops/moe/group_mask.py), so
    # a name override between the two never changes the facade's filtering.
    signatures=frozenset(
        {
            format_signature(
                choices=dense_tensor_format(torch.float32),
                group_ids=dense_tensor_format(torch.int64),
            )
        }
    ),
    # No ``rows`` trait: every positive row count is admitted (module doc); the
    # facade still passes ``rows`` in its trait dict, which no spec constrains.
    traits={
        "experts": frozenset({_BLOCK_E}),
        "num_groups": frozenset({_NUM_GROUPS}),
        "topk_groups": frozenset({_TOPK_GROUP}),
        "contiguous": frozenset({True}),
    },
    # REFERENCE band: never auto-selected while torch_group_mask is registered;
    # reached only by name (``override=``, ``kernel_override()``,
    # ``TOKENSPEED_KERNEL_OVERRIDE_MOE_GROUP_MASK``).
    priority=Priority.REFERENCE,
)
def triton_group_mask_bits(
    choices: torch.Tensor,
    group_ids: torch.Tensor,
    group_scores: torch.Tensor,
    grouped: torch.Tensor,
) -> torch.Tensor:
    """Launch the uint32 bit-select on an admitted call; raise otherwise.

    Args:
        choices: ``[rows, 256]`` contiguous FP32 CUDA biased scores.
        group_ids: ``[rows, 4]`` contiguous int64 CUDA kept-group indices in
            ``[0, 8)`` (a producer precondition, never read on the host).
        group_scores: ``[rows, 8]`` contiguous FP32 CUDA group scores (admission
            witness only).
        grouped: ``[rows, 8, 32]`` contiguous FP32 CUDA view of ``choices``
            (admission witness only).

    Returns:
        A fresh contiguous FP32 ``[rows, 256]`` tensor: kept words copied bit
        for bit, excluded words ``0xFF800000``.

    Raises:
        ValueError: :func:`group_mask_bits_rejection` rejects the call (zero
            rows included). There is no fallback to the torch statements.
    """
    rejection = group_mask_bits_rejection(choices, group_ids, group_scores, grouped)
    if rejection:
        raise ValueError(
            f"{TRITON_GROUP_MASK_BITS} cannot serve this call: {rejection}"
        )
    output = torch.empty_like(choices)  # fresh, contiguous, disjoint from the inputs
    _group_mask_bits[(choices.shape[0],)](
        choices,
        group_ids,
        output,
        BLOCK_E=_BLOCK_E,
        NUM_GROUPS=_NUM_GROUPS,
        TOPK_GROUP=_TOPK_GROUP,
        **_LAUNCH_OPTIONS,
    )
    return output
