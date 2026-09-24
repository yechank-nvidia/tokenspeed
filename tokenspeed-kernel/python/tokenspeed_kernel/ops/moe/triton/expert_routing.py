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

"""Triton kernels of ``moe.expert_routing``: the full, the compact and the compact-ordered route tables.

The full and the compact kernels start with the optional prep step -- under
``StagePolicy.sort_routes`` the statements ``topk_ids.sort(dim=-1)`` and
``topk_weights.gather(1, order)`` (before the int32 cast, in that order);
otherwise the inputs pass through unchanged, which is what ``_moe`` asks for --
and then build the per-expert route tables the stage kernels traverse; the
compact-ordered kernel moves those two statements into its CTA at 8 or 32
routes:

* ``triton_moe_routing_full``: the canonical ``_routing_kernel`` moved here
  verbatim from ``ops/moe/triton/bf16.py``. One program per expert (grid
  ``(E,)``, ``BLOCK_ROUTES = 128`` for ``R <= 128`` else ``1024``,
  ``num_warps=4``) scans the flattened ids in blocks and writes that expert's
  route ids ascending into row ``e`` of the ``[E, R]`` table and its count into
  ``counts[e]``. Returns ``compact=False``: the stage kernels visit every
  expert.
* ``triton_moe_routing_compact``: a one-CTA packing kernel (grid ``(1,)``,
  constexpr ``NUM_ROUTES``/``NUM_EXPERTS``/``BLOCK_ROUTES = next_power_of_2(R)``,
  ``num_warps=4``, the launch option) that writes the same table rows and
  counts and appends a tail to ``counts``: layout ``[E counts | min(E, R)
  ascending active expert ids | active count]``. With ``compact=True`` the
  stage kernels read the active count as their trip count and visit only the
  listed experts (a device-side load, so CUDA-graph replay follows the live
  routing).
* ``triton_moe_routing_compact_ordered``: the compact kernel with the sort and
  the gather inside the CTA. ``_sort8`` sorts each token's eight unsorted ids
  with a 32-lane bitonic network (eight valid lanes, key/index 0 padding) and
  returns the sorted ids and their slot permutation; the paired FP32 weights
  ``weights[row * 8 + slot]`` are gathered and stored to a fresh tensor (one
  allocation in place of the gather output), then the compact packing runs
  verbatim on the sorted ids. One CTA, ``BLOCK_ROUTES = R`` in ``{8, 32}``,
  ``num_warps=4``. At every other admitted route count the kernel function
  runs the torch sort + gather and the compact or the full tables (below).

The sort network is the compare-exchange schedule of ATen's
``bitonicSortKVInPlace`` for ``[N, 8]`` int32 sorts, so tie permutations
(duplicate or invalid ids: which FP32 weight lands in which sorted slot) match
torch's; the byte comparison of ``route_weights`` on tie rows in
``test/nvidia/ops/moe/test_expert_routing_ordered_cuda.py`` is the detector of
a torch dispatch change. Do not add a tie-break. Correctness is established by
the byte-equality tests (``test/nvidia/ops/moe/test_expert_routing_cuda.py``
and ``test_expert_routing_ordered_cuda.py``), not by a binary hash.

Registered in the REFERENCE band, so ranked selection never picks the compact
kernel while ``triton_moe_routing_full`` is registered; it is reached only by
name (``override=``, ``kernel_override("moe", "expert_routing", ...)``,
``TOKENSPEED_KERNEL_OVERRIDE_MOE_EXPERT_ROUTING``, ``--kernel-override
moe.expert_routing=triton_moe_routing_compact``). Its admission -- 256 experts
under a BF16 SiLU stage policy (``COMPACT_STAGE_POLICIES``: sorted or unsorted
routes), CUDA tensors on one device, a positive route count -- is declared
twice on purpose: as spec traits (``num_experts``, ``stage_policy``), so ranked
selection and callers that re-check traits reject it, and as the host guard
:func:`routing_compact_rejection`, so a by-name override on a rejected call
raises instead of falling back. The route count is not a trait: for ``0 < R <=
MAX_COMPACT_ROUTES`` (32) the kernel packs, for ``R > 32`` it emits the full
tables and ``compact=False`` -- the documented shape rule of this kernel's
return value (prefill under a process-wide override runs), not a switch to
another registered kernel. ``R == 0`` is refused by the host guard alone (the
one-CTA kernel is never launched on an empty route set; ``_moe`` returns before
routing at zero tokens, so only a direct facade call reaches it; the full
kernel serves it).

Admission of the ordered kernel: the compact envelope restricted to
``sort_routes=True`` (trait ``stage_policy`` = ``ORDERED_STAGE_POLICY``; the
in-CTA sort is the sorted prep, an unsorted caller has nothing to fuse) plus
top-k 8, contiguous int32 ids and contiguous FP32 weights, declared as spec
traits and an int32-only signature (so ``explain_selection`` lists it under
"Filtered out" for int64 ids) and re-stated by the host guard
:func:`routing_compact_ordered_rejection` (top-k, dtype and layout are guard
clauses only: the facade's trait dict carries ``num_experts``, ``num_routes``
and ``stage_policy``, and a spec trait the request does not carry is never
evaluated). ``_moe`` passes ``sort_routes=False``, so this kernel is reachable
through the facade (tests, callers that sort) and refused under
``--kernel-override`` from the default MoE path. Shape rule on the return
value: ``R in ORDERED_ROUTES`` (8 or 32, rows 1 and 4 at top-8) -> the in-CTA
sort, ``compact=True, ordered=True``; ``R`` in ``{16, 24}`` -> the torch sort +
gather and the compact tables, ``ordered=False``; ``R > 32`` -> the torch sort
+ gather and the full tables, ``compact=False``. int64 ids are refused because
the torch path sorts before the int32 cast (an in-kernel int32 sort cannot
reproduce the sort-then-alias of values such as ``2**32 + 7``); ``R == 0`` is
refused as for the compact kernel. One ``--kernel-override
moe.expert_routing=...`` names one kernel, so the ordered override replaces the
compact override rather than stacking on it.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe.expert_routing import (
    COMPACT_NUM_EXPERTS,
    COMPACT_STAGE_POLICIES,
    EXPERT_ROUTING_SIGNATURES,
    MAX_COMPACT_ROUTES,
    ORDERED_ROUTES,
    ORDERED_STAGE_POLICY,
    ORDERED_TOP_K,
    TRITON_MOE_ROUTING_COMPACT,
    TRITON_MOE_ROUTING_COMPACT_ORDERED,
    TRITON_MOE_ROUTING_FULL,
    ExpertRouting,
    StagePolicy,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "routing_compact_ordered_rejection",
    "routing_compact_rejection",
    "triton_moe_routing_compact",
    "triton_moe_routing_compact_ordered",
    "triton_moe_routing_full",
]


@triton.jit
def _routing_kernel(
    topk_ids_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_routes,
    BLOCK_ROUTES: tl.constexpr,
):
    expert_id = tl.program_id(0)
    count = 0
    num_blocks = tl.cdiv(num_routes, BLOCK_ROUTES)
    for block_id in range(num_blocks):
        route_ids = block_id * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
        route_mask = route_ids < num_routes
        selected_experts = tl.load(topk_ids_ptr + route_ids, mask=route_mask, other=-1)
        matches = route_mask & (selected_experts == expert_id)
        local_rank = tl.cumsum(matches.to(tl.int32), axis=0) - 1
        tl.store(
            expert_route_ids_ptr + expert_id * num_routes + count + local_rank,
            route_ids,
            mask=matches,
        )
        count += tl.sum(matches.to(tl.int32), axis=0)
    tl.store(expert_counts_ptr + expert_id, count)


def _routing(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = topk_ids.to(torch.int32).contiguous()
    num_routes = topk_ids.numel()
    expert_route_ids = torch.empty(
        (num_experts, num_routes), device=topk_ids.device, dtype=torch.int32
    )
    expert_counts = torch.empty(num_experts, device=topk_ids.device, dtype=torch.int32)
    block_routes = 128 if num_routes <= 128 else 1024
    _routing_kernel[(num_experts,)](
        topk_ids,
        expert_route_ids,
        expert_counts,
        num_routes,
        BLOCK_ROUTES=block_routes,
        num_warps=4,
    )
    return expert_route_ids, expert_counts


# Launch option of the compact kernel.
_LAUNCH_OPTIONS_COMPACT = dict(num_warps=4)
# The ordered kernel's signature: int32 ids only (int64 ids are a refusal, module doc).
_SIGNATURES_INT32_ONLY = frozenset(
    {format_signature(topk_ids=dense_tensor_format(torch.int32))}
)


@triton.jit
def _routing_compact_kernel(
    topk_ids_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    # One CTA; rows are experts in ascending order, columns original route IDs.
    experts = tl.arange(0, NUM_EXPERTS)
    routes = tl.arange(0, BLOCK_ROUTES)
    selected = tl.load(topk_ids_ptr + routes, mask=routes < NUM_ROUTES, other=-1)
    matches = (routes[None, :] < NUM_ROUTES) & (selected[None, :] == experts[:, None])
    local_ranks = tl.cumsum(matches.to(tl.int32), axis=1) - 1
    counts = tl.sum(matches.to(tl.int32), axis=1)
    tl.store(
        expert_route_ids_ptr + experts[:, None] * NUM_ROUTES + local_ranks,
        routes[None, :],
        mask=matches,
    )
    tl.store(expert_counts_ptr + experts, counts)
    present = counts > 0
    active_positions = tl.cumsum(present.to(tl.int32), axis=0) - 1
    # Tail layout: [E counts | min(E,R) ascending active IDs | active count].
    tl.store(expert_counts_ptr + NUM_EXPERTS + active_positions, experts, mask=present)
    tl.store(
        expert_counts_ptr + NUM_EXPERTS + tl.minimum(NUM_EXPERTS, NUM_ROUTES),
        tl.sum(present.to(tl.int32), axis=0),
    )


def _routing_compact(topk_ids, num_experts):
    # Preserve the owner's int32 conversion, including its existing cast semantics.
    ids = topk_ids.to(torch.int32).contiguous()
    routes = ids.numel()
    if num_experts != 256 or not 0 < routes <= MAX_COMPACT_ROUTES:
        raise ValueError("Compact routing only accepts E256 and 1..32 routes")
    route_ids = torch.empty((num_experts, routes), device=ids.device, dtype=torch.int32)
    counts = torch.empty(
        num_experts + min(num_experts, routes) + 1,
        device=ids.device,
        dtype=torch.int32,
    )
    _routing_compact_kernel[(1,)](
        ids,
        route_ids,
        counts,
        NUM_ROUTES=routes,
        NUM_EXPERTS=num_experts,
        BLOCK_ROUTES=triton.next_power_of_2(routes),
        **_LAUNCH_OPTIONS_COMPACT,
    )
    return route_ids, counts


# The 15 compare-exchange stages of ATen's SortUtils.cuh bitonic network for 32
# items, comparator ``((a < b) & valid_a) | ~valid_b`` -- not a stable sort; the
# tie permutation is Torch's, do not add a tie-break.
@triton.jit
def _swap(keys, order, valid, STRIDE: tl.constexpr, SIZE: tl.constexpr):
    lane = tl.arange(0, 32)[None, :]
    partner = tl.broadcast_to(lane ^ STRIDE, keys.shape)
    other_keys = tl.gather(keys, partner, axis=1)
    other_order = tl.gather(order, partner, axis=1)
    other_valid = tl.gather(valid, partner, axis=1)
    low = (lane & STRIDE) == 0
    a = tl.where(low, keys, other_keys)
    b = tl.where(low, other_keys, keys)
    valid_a = tl.where(low, valid, other_valid)
    valid_b = tl.where(low, other_valid, valid)
    # Exact SortUtils.cuh bitonicSwap: not a stable or lexicographic sort.
    comparison = ((a < b) & valid_a) | ~valid_b
    lower = lane & ~STRIDE
    thread = (lower // (2 * STRIDE)) * STRIDE + lower % STRIDE
    if SIZE == 0:
        direction = tl.full(keys.shape, False, tl.int1)
    else:
        direction = tl.broadcast_to((thread & (SIZE // 2)) != 0, keys.shape)
    exchange = comparison == direction
    return (
        tl.where(exchange, other_keys, keys),
        tl.where(exchange, other_order, order),
        tl.where(exchange, other_valid, valid),
    )


@triton.jit
def _sort8(ids_ptr, ROWS: tl.constexpr):
    lane = tl.arange(0, 32)[None, :]
    row = tl.arange(0, ROWS)[:, None]
    valid = tl.broadcast_to(lane < 8, (ROWS, 32))
    keys = tl.load(ids_ptr + row * 8 + lane, mask=valid, other=0)
    order = tl.broadcast_to(tl.where(lane < 8, lane, 0), (ROWS, 32))
    keys, order, valid = _swap(keys, order, valid, 1, 2)
    keys, order, valid = _swap(keys, order, valid, 2, 4)
    keys, order, valid = _swap(keys, order, valid, 1, 4)
    keys, order, valid = _swap(keys, order, valid, 4, 8)
    keys, order, valid = _swap(keys, order, valid, 2, 8)
    keys, order, valid = _swap(keys, order, valid, 1, 8)
    keys, order, valid = _swap(keys, order, valid, 8, 16)
    keys, order, valid = _swap(keys, order, valid, 4, 16)
    keys, order, valid = _swap(keys, order, valid, 2, 16)
    keys, order, valid = _swap(keys, order, valid, 1, 16)
    keys, order, valid = _swap(keys, order, valid, 16, 0)
    keys, order, valid = _swap(keys, order, valid, 8, 0)
    keys, order, valid = _swap(keys, order, valid, 4, 0)
    keys, order, valid = _swap(keys, order, valid, 2, 0)
    keys, order, valid = _swap(keys, order, valid, 1, 0)
    first = tl.broadcast_to(tl.arange(0, 8)[None, :], (ROWS, 8))
    return tl.gather(keys, first, axis=1), tl.gather(order, first, axis=1)


# The compact kernel with six statements in place of the ``selected`` load:
# ``_sort8`` over the unsorted ids, the paired FP32 gather ``weights[row * 8 +
# slot]`` stored to ``paired_ptr``, then the compact packing verbatim.
# BLOCK_ROUTES == NUM_ROUTES (8 or 32): ``routes`` and ``slot`` are broadcast together.
@triton.jit
def _routing_compact_ordered_kernel(
    topk_ids_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    weights_ptr,
    paired_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    # One CTA; rows are experts in ascending order, columns original route IDs.
    experts = tl.arange(0, NUM_EXPERTS)
    routes = tl.arange(0, BLOCK_ROUTES)
    ordered, permutation = _sort8(topk_ids_ptr, NUM_ROUTES // 8)
    selected = ordered.reshape((NUM_ROUTES,))
    slot = permutation.reshape((NUM_ROUTES,))
    source = routes // 8 * 8 + slot
    paired = tl.load(weights_ptr + source)
    tl.store(paired_ptr + routes, paired)
    matches = (routes[None, :] < NUM_ROUTES) & (selected[None, :] == experts[:, None])
    local_ranks = tl.cumsum(matches.to(tl.int32), axis=1) - 1
    counts = tl.sum(matches.to(tl.int32), axis=1)
    tl.store(
        expert_route_ids_ptr + experts[:, None] * NUM_ROUTES + local_ranks,
        routes[None, :],
        mask=matches,
    )
    tl.store(expert_counts_ptr + experts, counts)
    present = counts > 0
    active_positions = tl.cumsum(present.to(tl.int32), axis=0) - 1
    # Tail layout: [E counts | min(E,R) ascending active IDs | active count].
    tl.store(expert_counts_ptr + NUM_EXPERTS + active_positions, experts, mask=present)
    tl.store(
        expert_counts_ptr + NUM_EXPERTS + tl.minimum(NUM_EXPERTS, NUM_ROUTES),
        tl.sum(present.to(tl.int32), axis=0),
    )


def _routing_compact_ordered(ids, weights, num_experts):
    routes = ids.numel()
    if num_experts != COMPACT_NUM_EXPERTS or routes not in ORDERED_ROUTES:
        raise ValueError("Ordered compact routing only accepts E256 and 8 or 32 routes")
    if ids.dtype is not torch.int32 or not ids.is_contiguous():
        raise ValueError("Ordered compact routing needs contiguous int32 ids")
    if weights.dtype is not torch.float32 or not weights.is_contiguous():
        raise ValueError("Ordered compact routing needs contiguous float32 weights")
    route_ids = torch.empty((num_experts, routes), device=ids.device, dtype=torch.int32)
    counts = torch.empty(
        num_experts + min(num_experts, routes) + 1,
        device=ids.device,
        dtype=torch.int32,
    )
    paired = torch.empty_like(weights)  # one allocation in place of the gather output
    _routing_compact_ordered_kernel[(1,)](
        ids,
        route_ids,
        counts,
        weights,
        paired,
        NUM_ROUTES=routes,
        NUM_EXPERTS=num_experts,
        BLOCK_ROUTES=triton.next_power_of_2(routes),
        **_LAUNCH_OPTIONS_COMPACT,
    )
    return route_ids, counts, paired


def _prep(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, stage_policy: StagePolicy
) -> tuple[torch.Tensor, torch.Tensor]:
    """The prep step shared by the full and the compact kernels: the per-token
    sort and the paired gather under ``sort_routes``, the inputs unchanged
    otherwise (what ``_moe`` asks for)."""
    if stage_policy.sort_routes:
        # Ascending-expert accumulation in the combine, independently of the
        # routing's score-sorted output order. Keep the original top-k tensors
        # unchanged.
        topk_ids, order = topk_ids.sort(dim=-1)
        topk_weights = topk_weights.gather(1, order)
    return topk_ids, topk_weights


def _compact_or_full_tables(
    ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """The documented shape rule: ``0 < R <= MAX_COMPACT_ROUTES`` packs
    (``compact=True``); ``R > MAX_COMPACT_ROUTES`` emits the full tables
    (``compact=False``)."""
    if ids.numel() <= MAX_COMPACT_ROUTES:
        route_ids, counts = _routing_compact(ids, num_experts)
        return route_ids, counts, True
    route_ids, counts = _routing(ids, num_experts)
    return route_ids, counts, False


def _envelope_rejection(
    topk_ids: torch.Tensor, num_experts: int, stage_policy: StagePolicy
) -> str:
    """The compact envelope clauses (metadata only): 256 experts, a BF16 SiLU
    stage policy (``COMPACT_STAGE_POLICIES``), a positive route count; empty
    when they hold."""
    if num_experts != COMPACT_NUM_EXPERTS:
        return f"num_experts must be {COMPACT_NUM_EXPERTS}, got {num_experts}"
    if stage_policy.name not in COMPACT_STAGE_POLICIES:
        return (
            f"stage_policy must be one of {', '.join(sorted(COMPACT_STAGE_POLICIES))} "
            f"(BF16 SiLU, sorted or unsorted routes), got {stage_policy.name}"
        )
    if topk_ids.numel() == 0:
        return (
            "num_routes=0: the one-CTA packing kernel is not launched on an empty "
            "route set (num_routes must be positive)"
        )
    return ""


def _device_rejection(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> str:
    """The device clauses (last in every guard): CUDA tensors on one device."""
    if not topk_ids.is_cuda:
        return f"topk_ids must be a CUDA tensor, got device {topk_ids.device}"
    if topk_weights.device != topk_ids.device:
        return "topk_ids and topk_weights must share one device"
    return ""


def routing_compact_rejection(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    stage_policy: StagePolicy,
) -> str:
    """Why ``triton_moe_routing_compact`` cannot serve this call; empty when admitted.

    Metadata only (no device read): 256 experts, a BF16 SiLU stage policy
    (``COMPACT_STAGE_POLICIES``: ``bf16_silu_unsorted`` or ``bf16_silu_sorted``),
    a positive route count (the one-CTA kernel is not launched on an empty
    route set), CUDA tensors on one device. The reason names the first
    rejecting attribute; device clauses come last so that envelope findings
    are reported on any device. The route count above ``MAX_COMPACT_ROUTES``
    is not a rejection: the kernel then emits the full tables (module doc).

    Args:
        topk_ids: Expert ids the kernel packs.
        topk_weights: Route weights the kernel gathers; device witness only.
        num_experts: Expert count of the tables.
        stage_policy: The stage-kernel policy of the call.

    Returns:
        The rejection reason, or ``""`` when the kernel admits the call.
    """
    return _envelope_rejection(
        topk_ids, num_experts, stage_policy
    ) or _device_rejection(topk_ids, topk_weights)


def _ordered_sort_rejection(stage_policy: StagePolicy) -> str:
    """The ordered kernel's policy clause (metadata only): ``sort_routes`` must
    be set -- the in-CTA sort is the sorted prep; an unsorted caller has
    nothing to fuse. Empty when it holds."""
    if not stage_policy.sort_routes:
        return (
            "stage_policy.sort_routes must be True (the in-CTA sort is the sorted "
            "prep; an unsorted caller has nothing to fuse), got "
            f"{stage_policy.name}"
        )
    return ""


def _ordered_layout_rejection(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> str:
    """The ordered kernel's layout clauses (metadata only): top-k 8, contiguous
    int32 ids, contiguous FP32 weights; empty when they hold."""
    if topk_ids.shape[1] != ORDERED_TOP_K:
        return (
            f"top_k must be {ORDERED_TOP_K} (the in-CTA sort network reads eight ids "
            f"per token), got {topk_ids.shape[1]}"
        )
    if topk_ids.dtype is not torch.int32:
        return (
            "topk_ids must be int32 (the torch kernels sort int64 ids before the "
            "int32 cast; sorting the cast values would alias ids such as 2**32 + 7), "
            f"got {topk_ids.dtype}"
        )
    if not topk_ids.is_contiguous():
        return "topk_ids must be contiguous (the sort network addresses row * 8 + lane)"
    if topk_weights.dtype is not torch.float32:
        return (
            "topk_weights must be float32 (the paired gather copies FP32 words), got "
            f"{topk_weights.dtype}"
        )
    if not topk_weights.is_contiguous():
        return "topk_weights must be contiguous"
    return ""


def routing_compact_ordered_rejection(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    stage_policy: StagePolicy,
) -> str:
    """Why ``triton_moe_routing_compact_ordered`` cannot serve this call; empty when admitted.

    Metadata only (no device read): the compact envelope (256 experts, a BF16
    SiLU policy, a positive route count), then ``sort_routes=True``, then
    top-k 8 / contiguous int32 ids / contiguous FP32 weights, then the device
    clauses (last, so envelope, policy and layout findings are reported on any
    device). The reason names the first rejecting attribute. The route count
    is not a rejection: 8 or 32 routes take the in-CTA sort, 16 or 24 the
    torch sort with the compact tables, more than 32 the full tables (module
    doc).

    Args:
        topk_ids: Expert ids the kernel sorts and packs.
        topk_weights: Route weights the kernel gathers in the CTA.
        num_experts: Expert count of the tables.
        stage_policy: The stage-kernel policy of the call.

    Returns:
        The rejection reason, or ``""`` when the kernel admits the call.
    """
    return (
        _envelope_rejection(topk_ids, num_experts, stage_policy)
        or _ordered_sort_rejection(stage_policy)
        or _ordered_layout_rejection(topk_ids, topk_weights)
        or _device_rejection(topk_ids, topk_weights)
    )


def _ordered_tables(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    stage_policy: StagePolicy,
) -> ExpertRouting:
    """The ordered kernel's shape rule on an admitted call (module doc).

    ``R in ORDERED_ROUTES`` (8 or 32) -> :func:`_routing_compact_ordered`: one
    CTA sorts, gathers and packs; ``compact=True, ordered=True``. Otherwise the
    torch sort + gather (:func:`_prep`, the same function object the full and
    the compact kernels call) and then :func:`_compact_or_full_tables`: ``R <=
    MAX_COMPACT_ROUTES`` (16, 24) -> the compact tables, ``compact=True``; ``R
    > MAX_COMPACT_ROUTES`` -> the full tables, ``compact=False``; both
    ``ordered=False``. The torch-sorted branches are therefore the compact and
    the full kernels' bytes by construction.
    """
    if topk_ids.numel() in ORDERED_ROUTES:
        route_ids, counts, paired = _routing_compact_ordered(
            topk_ids, topk_weights, num_experts
        )
        return ExpertRouting(route_ids, counts, paired, compact=True, ordered=True)
    ids, weights = _prep(topk_ids, topk_weights, stage_policy)
    route_ids, counts, compact = _compact_or_full_tables(ids, num_experts)
    return ExpertRouting(route_ids, counts, weights, compact=compact, ordered=False)


@register_kernel(
    "moe",
    "expert_routing",
    name=TRITON_MOE_ROUTING_FULL,
    solution="triton",
    signatures=EXPERT_ROUTING_SIGNATURES,
    # Admits every call: the statements are general. While this is the only
    # non-REFERENCE registration under the operator, ranked selection lands here
    # for every call, so ``_moe``'s tables are the canonical ones it inlined.
    traits={},
    priority=Priority.PORTABLE,
)
def triton_moe_routing_full(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    stage_policy: StagePolicy,
) -> ExpertRouting:
    """The optional prep step and the canonical route tables ``_moe`` built inline.

    Args:
        topk_ids: ``[tokens, top_k]`` int32/int64 expert ids.
        topk_weights: ``[tokens, top_k]`` floating-point route weights.
        num_experts: Expert count ``E``.
        stage_policy: Decides the sort (``sort_routes``); not otherwise
            constrained.

    Returns:
        ``ExpertRouting`` with int32 ``[E, R]`` route ids (ascending per
        expert, slots at or beyond the count unwritten), int32 ``[E]`` counts,
        the gathered (``sort_routes``) or original route weights,
        ``compact=False`` and ``ordered=False``.
    """
    ids, weights = _prep(topk_ids, topk_weights, stage_policy)
    route_ids, counts = _routing(ids, num_experts)
    return ExpertRouting(route_ids, counts, weights, compact=False, ordered=False)


@register_kernel(
    "moe",
    "expert_routing",
    name=TRITON_MOE_ROUTING_COMPACT,
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    # The same int32/int64 signatures as ``triton_moe_routing_full``, so a name
    # override between the two never changes the facade's filtering.
    signatures=EXPERT_ROUTING_SIGNATURES,
    # No ``num_routes`` trait: the route rule is this kernel's return-value rule
    # (module doc); the facade still passes ``num_routes``, which no spec
    # constrains. ``stage_policy`` is the envelope the stage byte-equality test
    # covers (BF16 SiLU, sorted or unsorted routes); widen it after running that
    # test at other dtypes / activations / expert counts.
    traits={
        "num_experts": frozenset({COMPACT_NUM_EXPERTS}),
        "stage_policy": COMPACT_STAGE_POLICIES,
    },
    # REFERENCE band: never auto-selected while triton_moe_routing_full is
    # registered; reached only by name (``override=``, ``kernel_override()``,
    # ``TOKENSPEED_KERNEL_OVERRIDE_MOE_EXPERT_ROUTING``).
    priority=Priority.REFERENCE,
)
def triton_moe_routing_compact(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    stage_policy: StagePolicy,
) -> ExpertRouting:
    """The optional prep step and the compact route tables on an admitted call;
    raise otherwise.

    Args:
        topk_ids: ``[tokens, top_k]`` int32/int64 CUDA expert ids.
        topk_weights: ``[tokens, top_k]`` floating-point CUDA route weights on
            the same device.
        num_experts: Must be 256.
        stage_policy: Must name one of ``COMPACT_STAGE_POLICIES``
            (``bf16_silu_unsorted`` or ``bf16_silu_sorted``).

    Returns:
        ``ExpertRouting`` with the tables of the packing kernel and
        ``compact=True`` for ``0 < R <= 32`` routes; the full tables and
        ``compact=False`` for more (the documented shape rule);
        ``ordered=False``. The gathered (``sort_routes``) or original route
        weights in both cases.

    Raises:
        ValueError: :func:`routing_compact_rejection` rejects the call (an
            empty route set included). There is no fallback to the full kernel
            on a rejected call.
    """
    rejection = routing_compact_rejection(
        topk_ids, topk_weights, num_experts, stage_policy
    )
    if rejection:
        raise ValueError(
            f"{TRITON_MOE_ROUTING_COMPACT} cannot serve this call: {rejection}"
        )
    ids, weights = _prep(topk_ids, topk_weights, stage_policy)
    route_ids, counts, compact = _compact_or_full_tables(ids, num_experts)
    return ExpertRouting(route_ids, counts, weights, compact=compact, ordered=False)


@register_kernel(
    "moe",
    "expert_routing",
    name=TRITON_MOE_ROUTING_COMPACT_ORDERED,
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    # int32 only: int64 ids are a refusal of this kernel (alias hazard, module doc).
    signatures=_SIGNATURES_INT32_ONLY,
    # The compact envelope restricted to the sorted prep; top-k, dtype and layout
    # are host-guard clauses because the facade's trait dict does not carry them
    # (``expert_routing_traits``).
    traits={
        "num_experts": frozenset({COMPACT_NUM_EXPERTS}),
        "stage_policy": frozenset({ORDERED_STAGE_POLICY}),
    },
    # REFERENCE band: never auto-selected; reached only by name. One override
    # names one kernel, so this override replaces the compact override.
    priority=Priority.REFERENCE,
)
def triton_moe_routing_compact_ordered(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    stage_policy: StagePolicy,
) -> ExpertRouting:
    """The compact route tables with the sort and the paired weight gather moved
    into the CTA on an admitted call; raise otherwise.

    Args:
        topk_ids: ``[tokens, 8]`` contiguous int32 CUDA expert ids (unsorted;
            int64 ids are refused, see the module doc).
        topk_weights: ``[tokens, 8]`` contiguous FP32 CUDA route weights on the
            same device.
        num_experts: Must be 256.
        stage_policy: Must name ``bf16_silu_sorted`` (``sort_routes=True``).

    Returns:
        ``ExpertRouting`` per the shape rule on the route count ``R``: 8 or 32
        (``ORDERED_ROUTES``) -> the one-CTA kernel sorts each token's ids,
        gathers the paired FP32 weights and packs the compact tables,
        ``compact=True, ordered=True``; 16 or 24 -> the torch sort + gather
        and the compact tables (the compact kernel's statements),
        ``compact=True, ordered=False``; more than 32 -> the torch sort +
        gather and the full tables (the full kernel's statements),
        ``compact=False, ordered=False``. The route weights are always a fresh
        tensor gathered with the sort order.

    Raises:
        ValueError: :func:`routing_compact_ordered_rejection` rejects the call
            (an empty route set, an unsorted policy, another top-k, int64 or
            strided ids, non-FP32 or strided weights, another envelope or
            device). There is no fallback to another registered kernel on a
            rejected call.
    """
    rejection = routing_compact_ordered_rejection(
        topk_ids, topk_weights, num_experts, stage_policy
    )
    if rejection:
        raise ValueError(
            f"{TRITON_MOE_ROUTING_COMPACT_ORDERED} cannot serve this call: {rejection}"
        )
    return _ordered_tables(topk_ids, topk_weights, num_experts, stage_policy)
