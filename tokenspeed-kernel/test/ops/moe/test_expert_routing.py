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

"""``moe.expert_routing`` through ``select_kernel``: registration, admission, the by-name switch and byte equality with the routing statement of ``_moe``.

CPU only. Importing ``tokenspeed_kernel`` registers the three kernels (the
package init imports ``ops.moe.triton.expert_routing``); ``load_builtin_kernels``
re-populates the singleton when an earlier test module reset it. No Triton
kernel is launched here: the three ``triton.jit`` objects of the kernel module
are monkeypatched with :class:`FakeLaunch` stand-ins that record every launch
(grid, arguments, launch options) and compute the tables in torch with the
kernels' documented semantics -- full: per expert the ascending positions of
that expert in the flattened ids and its count, unwritten slots filled with
``-1`` so byte comparison is meaningful; compact: the same plus the tail
``[E counts | min(E, R) ascending active ids | active count]``; compact-ordered:
each row sorted with the stdlib replay of the documented bitonic network
(``header_sort``, tie permutation included), the paired weights written, then
the compact tables of the sorted ids. So the facade, the prep step, the route
rules and the launch geometry are exercised for real while the compact and the
ordered host guards reject CPU tensors, which is what the override tests
assert -- the by-name switch has no fallback
(``test/nvidia/ops/moe/test_expert_routing_cuda.py`` and
``test_expert_routing_ordered_cuda.py`` run the kernels). The ordered kernel's
source is proven to be the compact kernel plus the six ordered-kernel
statements by a reverse AST rewrite, and its sort network to be the documented
schedule.
"""

from __future__ import annotations

import ast
import inspect
import random
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.expert_routing import (
    COMPACT_NUM_EXPERTS,
    COMPACT_STAGE_POLICIES,
    EXPERT_ROUTING_FAMILY,
    EXPERT_ROUTING_MODE,
    EXPERT_ROUTING_OVERRIDE_ENV,
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
    expert_routing_traits,
    moe_expert_routing,
)
from tokenspeed_kernel.ops.moe.triton import expert_routing as kernels
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import (
    KernelRegistry,
    Priority,
    describe_kernel,
    load_builtin_kernels,
)
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectionEvent,
    _witnessed_overrides,
    add_selection_listener,
    explain_selection,
    kernel_override,
    remove_selection_listener,
    select_kernel,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

OPERATOR = (EXPERT_ROUTING_FAMILY, EXPERT_ROUTING_MODE)
FULL, COMPACT = TRITON_MOE_ROUTING_FULL, TRITON_MOE_ROUTING_COMPACT
ORDERED = TRITON_MOE_ROUTING_COMPACT_ORDERED
SIGNATURE_INT32 = format_signature(topk_ids=dense_tensor_format(torch.int32))
SIGNATURE_INT64 = format_signature(topk_ids=dense_tensor_format(torch.int64))
# Example expert layer: 256 experts, top-8, BF16 SiLU.
EXPERTS, TOP_K = 256, 8
# The policy ``_moe`` passes (top-k order kept) and the sorted policy the
# ordered kernel is qualified for.
STAGE_POLICY = StagePolicy(torch.bfloat16, "silu", False)
SORTED_POLICY = StagePolicy(torch.bfloat16, "silu", True)
CROSS_OPERATOR_KERNEL = "torch_decode_gemv"  # registered under gemm.decode_gemv
BF16_SOURCE = Path(kernels.__file__).resolve().with_name("bf16.py")
KERNELS_SOURCE = Path(kernels.__file__).resolve()
# The six statements of ``_routing_compact_ordered_kernel`` that stand in place
# of the compact kernel's ``selected`` load, restated here as text.
ORDERED_STATEMENTS = (
    "ordered, permutation = _sort8(topk_ids_ptr, NUM_ROUTES // 8)",
    "selected = ordered.reshape((NUM_ROUTES,))",
    "slot = permutation.reshape((NUM_ROUTES,))",
    "source = routes // 8 * 8 + slot",
    "paired = tl.load(weights_ptr + source)",
    "tl.store(paired_ptr + routes, paired)",
)
ORDERED_POINTERS = ("weights_ptr", "paired_ptr")
# A tie row (duplicates): the sort permutation is not the identity.
TIE_ROW = (7, 0, 7, 255, 0, 7, 255, 7)
SORT_ALPHABET = (-(2**31), -1, 0, 7, 255, 256, 2**31 - 1)


@pytest.fixture(autouse=True)
def expert_routing_registry():
    """The three kernels registered; no cached selection or override witness leaks between tests."""
    if KernelRegistry.get().get_by_name(FULL) is None:
        load_builtin_kernels()  # an earlier module reset the registry singleton
    KernelRegistry.get().clear_cache()
    _witnessed_overrides.clear()
    yield
    KernelRegistry.get().clear_cache()
    _witnessed_overrides.clear()


@pytest.fixture
def selection_events():
    events: list[SelectionEvent] = []
    add_selection_listener(events.append)
    yield events
    remove_selection_listener(events.append)


class FakeLaunch:
    """Stand-in for a ``triton.jit`` kernel object: ``kernel[grid](*args, **options)``
    records ``(grid, args, options)`` and computes the tables in torch."""

    def __init__(self, compute) -> None:
        self.compute = compute
        self.launches: list[tuple[tuple, tuple, dict]] = []

    def __getitem__(self, grid):
        def launch(*args, **options):
            self.launches.append((grid, args, options))
            self.compute(*args, **options)

        return launch


def _write_rows(ids: torch.Tensor, table: torch.Tensor, counts: torch.Tensor) -> None:
    """Per expert the ascending positions of that expert in the flattened ids (what both
    kernels write) and the count; unwritten slots set to -1."""
    flat = ids.reshape(-1).to(torch.int64)
    table.fill_(-1)
    for expert in range(table.shape[0]):
        positions = torch.nonzero(flat == expert).squeeze(1)
        table[expert, : positions.numel()] = positions.to(torch.int32)
        counts[expert] = positions.numel()


def _full_tables(ids, table, counts, num_routes, *, BLOCK_ROUTES, num_warps):
    assert ids.dtype is torch.int32 and ids.is_contiguous()
    assert ids.numel() == num_routes and table.shape == (counts.shape[0], num_routes)
    _write_rows(ids, table, counts)


def _compact_tables(
    ids, table, counts, *, NUM_ROUTES, NUM_EXPERTS, BLOCK_ROUTES, num_warps
):
    assert ids.dtype is torch.int32 and ids.is_contiguous()
    assert ids.numel() == NUM_ROUTES and table.shape == (NUM_EXPERTS, NUM_ROUTES)
    assert counts.shape == (NUM_EXPERTS + min(NUM_EXPERTS, NUM_ROUTES) + 1,)
    counts.fill_(-1)
    _write_rows(ids, table, counts[:NUM_EXPERTS])
    active = torch.nonzero(counts[:NUM_EXPERTS] > 0).squeeze(1).to(torch.int32)
    counts[NUM_EXPERTS : NUM_EXPERTS + active.numel()] = active
    counts[NUM_EXPERTS + min(NUM_EXPERTS, NUM_ROUTES)] = active.numel()


@pytest.fixture
def fake_launches(monkeypatch) -> tuple[FakeLaunch, FakeLaunch]:
    full, compact = FakeLaunch(_full_tables), FakeLaunch(_compact_tables)
    monkeypatch.setattr(kernels, "_routing_kernel", full)
    monkeypatch.setattr(kernels, "_routing_compact_kernel", compact)
    return full, compact


def _network() -> tuple[tuple[int, int], ...]:
    """The 15 ``(STRIDE, SIZE)`` compare-exchange stages of ATen's 32-item bitonic
    network: sizes 2, 4, 8, 16 with halving strides, then the final merge 16..1."""
    stages: list[tuple[int, int]] = []
    for size in (2, 4, 8, 16):
        stride = size // 2
        while stride:
            stages.append((stride, size))
            stride //= 2
    stages.extend((stride, 0) for stride in (16, 8, 4, 2, 1))
    return tuple(stages)


NETWORK = _network()


def header_sort(values) -> tuple[list[int], list[int]]:
    """Literal sequential replay of the pinned SortUtils bitonicSort + bitonicSwap on
    eight valid keys padded to 32 (key/index 0, valid False); returns the sorted keys
    and the slot permutation. Not a stable sort: the tie permutation is Torch's."""
    keys = list(values) + [0] * 24
    order = list(range(8)) + [0] * 24
    valid = [True] * 8 + [False] * 24
    for stride, size in NETWORK:
        for thread in range(16):
            pos = 2 * thread - (thread & (stride - 1))
            other = pos + stride
            direction = bool(thread & (size // 2)) if size else False
            swap = (keys[pos] < keys[other] and valid[pos]) or not valid[other]
            if swap == direction:
                for array in (keys, order, valid):
                    array[pos], array[other] = array[other], array[pos]
    return keys[:8], order[:8]


def lane_sort(values) -> tuple[list[int], list[int]]:
    """Independent 32-lane form of the same network (the ``_swap`` lane arithmetic)."""
    keys, order, valid = (
        list(values) + [0] * 24,
        list(range(8)) + [0] * 24,
        [True] * 8 + [False] * 24,
    )
    for stride, size in NETWORK:
        old = keys[:], order[:], valid[:]
        for lane in range(32):
            lower = lane & ~stride
            upper = lower + stride
            thread = lower // (2 * stride) * stride + lower % stride
            direction = bool(thread & (size // 2)) if size else False
            swap = (old[0][lower] < old[0][upper] and old[2][lower]) or not old[2][
                upper
            ]
            src = lane ^ stride if swap == direction else lane
            keys[lane], order[lane], valid[lane] = (array[src] for array in old)
    return keys[:8], order[:8]


def _ordered_tables_fake(
    ids,
    table,
    counts,
    weights,
    paired,
    *,
    NUM_ROUTES,
    NUM_EXPERTS,
    BLOCK_ROUTES,
    num_warps,
):
    """The ordered kernel's documented semantics: each row sorted by the network replay
    (``header_sort``), the paired weights ``weights[row, perm]`` written, then the compact
    tables of the sorted flat ids."""
    assert ids.dtype is torch.int32 and ids.is_contiguous()
    assert ids.ndim == 2 and ids.shape[1] == 8 and ids.numel() == NUM_ROUTES
    assert BLOCK_ROUTES == NUM_ROUTES  # routes and slot are broadcast together
    assert weights.dtype is torch.float32 and weights.is_contiguous()
    assert paired.dtype is torch.float32 and paired.shape == weights.shape
    sorted_rows = []
    for row in range(ids.shape[0]):
        keys, permutation = header_sort(ids[row].tolist())
        sorted_rows.append(keys)
        paired[row] = weights[row, permutation]
    sorted_ids = torch.tensor(sorted_rows, dtype=torch.int32)
    _compact_tables(
        sorted_ids,
        table,
        counts,
        NUM_ROUTES=NUM_ROUTES,
        NUM_EXPERTS=NUM_EXPERTS,
        BLOCK_ROUTES=BLOCK_ROUTES,
        num_warps=num_warps,
    )


@pytest.fixture
def fake_ordered_launch(monkeypatch) -> FakeLaunch:
    ordered = FakeLaunch(_ordered_tables_fake)
    monkeypatch.setattr(kernels, "_routing_compact_ordered_kernel", ordered)
    return ordered


def _spec(name: str):
    spec = KernelRegistry.get().get_by_name(name)
    assert spec is not None, name
    return spec


def _traits(
    num_routes: int,
    num_experts: int = EXPERTS,
    stage_policy: str = STAGE_POLICY.name,
) -> dict[str, int | str]:
    return {
        "num_experts": num_experts,
        "num_routes": num_routes,
        "stage_policy": stage_policy,
    }


def _routing_inputs(
    rows: int, dtype: torch.dtype, seed: int, top_k: int = TOP_K, experts: int = EXPERTS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded ``[rows, top_k]`` ids with duplicates, an id ``>= E`` and a ``-1`` (all
    admitted by ``_validate`` and absent from the tables), never ascending, plus
    positive FP32 weights."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, experts, (rows, top_k), generator=generator)
    ids[0, 0] = experts - 1
    ids[0, 1] = 0
    if top_k >= 4:
        ids[0, 2] = ids[0, 3]  # a duplicate within one token
    if rows >= 2:
        ids[1, 0] = -1  # invalid ids: absent from routing
        ids[1, 1] = experts + 3
    weights = torch.rand((rows, top_k), generator=generator) + 0.1
    return ids.to(dtype).contiguous(), weights.contiguous()


def _policy(name: str) -> StagePolicy:
    dtype, activation, ordering = name.split("_")
    return StagePolicy(
        {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype],
        activation,
        ordering == "sorted",
    )


def test_registration_lists_full_then_compact_then_ordered():
    registry = KernelRegistry.get()
    specs = registry.list_kernels(*OPERATOR)
    assert [spec.name for spec in specs] == [FULL, COMPACT, ORDERED]
    full_spec, compact_spec, ordered_spec = specs
    assert (full_spec.solution, full_spec.priority) == ("triton", Priority.PORTABLE)
    assert (compact_spec.solution, compact_spec.priority) == (
        "triton",
        Priority.REFERENCE,
    )
    assert (ordered_spec.solution, ordered_spec.priority) == (
        "triton",
        Priority.REFERENCE,
    )
    # The ordered kernel declares int32 ids only (int64 is its documented refusal)
    # and the compact kernel's traits restricted to the sorted policy (top-k, dtype
    # and layout are guard clauses).
    assert ordered_spec.format_signatures == frozenset({SIGNATURE_INT32})
    assert not ordered_spec.supports_format_signature(SIGNATURE_INT64)
    assert ordered_spec.capability == CapabilityRequirement(
        vendors=frozenset({"nvidia"})
    )
    assert ordered_spec.traits == {
        "num_experts": frozenset({COMPACT_NUM_EXPERTS}),
        "stage_policy": frozenset({ORDERED_STAGE_POLICY}),
    }
    assert ordered_spec.traits["stage_policy"] < compact_spec.traits["stage_policy"]
    assert (
        full_spec.format_signatures
        == compact_spec.format_signatures
        == EXPERT_ROUTING_SIGNATURES
        == frozenset({SIGNATURE_INT32, SIGNATURE_INT64})
    )
    assert full_spec.capability == CapabilityRequirement()
    assert compact_spec.capability == CapabilityRequirement(
        vendors=frozenset({"nvidia"})
    )
    assert full_spec.traits == {}
    # No ``num_routes`` trait: the route rule is the kernel's return-value rule.
    assert compact_spec.traits == {
        "num_experts": frozenset({COMPACT_NUM_EXPERTS}),
        "stage_policy": COMPACT_STAGE_POLICIES,
    }
    assert (COMPACT_NUM_EXPERTS, COMPACT_STAGE_POLICIES, MAX_COMPACT_ROUTES) == (
        256,
        frozenset({"bf16_silu_unsorted", "bf16_silu_sorted"}),
        32,
    )
    assert ORDERED_STAGE_POLICY == "bf16_silu_sorted"
    assert ORDERED_STAGE_POLICY in COMPACT_STAGE_POLICIES
    full_text = describe_kernel(FULL)
    assert "Operator: moe.expert_routing" in full_text
    assert "Solution: triton" in full_text and "Priority: 4 (PORTABLE)" in full_text
    compact_text = describe_kernel(COMPACT)
    assert "Operator: moe.expert_routing" in compact_text
    assert (
        "Solution: triton" in compact_text and "Priority: 0 (REFERENCE)" in compact_text
    )
    ordered_text = describe_kernel(ORDERED)
    assert "Operator: moe.expert_routing" in ordered_text
    assert (
        "Solution: triton" in ordered_text and "Priority: 0 (REFERENCE)" in ordered_text
    )
    assert registry.get_impl(FULL).__name__ == FULL
    assert registry.get_impl(COMPACT).__name__ == COMPACT
    assert registry.get_impl(ORDERED).__name__ == ORDERED
    assert (
        EXPERT_ROUTING_OVERRIDE_ENV == "TOKENSPEED_KERNEL_OVERRIDE_MOE_EXPERT_ROUTING"
    )
    assert tokenspeed_kernel.moe_expert_routing is moe_expert_routing
    assert kernels.__all__ == [
        "routing_compact_ordered_rejection",
        "routing_compact_rejection",
        "triton_moe_routing_compact",
        "triton_moe_routing_compact_ordered",
        "triton_moe_routing_full",
    ]


def test_stage_policy_names_and_validation():
    assert STAGE_POLICY.name == "bf16_silu_unsorted"
    assert SORTED_POLICY.name == "bf16_silu_sorted" == ORDERED_STAGE_POLICY
    assert {STAGE_POLICY.name, SORTED_POLICY.name} == COMPACT_STAGE_POLICIES
    assert StagePolicy(torch.float16, "situ", False).name == "fp16_situ_unsorted"
    assert StagePolicy(torch.bfloat16, "swiglu", True).name == "bf16_swiglu_sorted"
    assert StagePolicy(torch.float16, "silu", True).name == "fp16_silu_sorted"
    with pytest.raises(ValueError, match="input_dtype"):
        StagePolicy(torch.float32, "silu", True)
    with pytest.raises(ValueError, match="activation"):
        StagePolicy(torch.bfloat16, "gelu", True)
    with pytest.raises(TypeError, match="sort_routes"):
        StagePolicy(torch.bfloat16, "silu", 1)
    with pytest.raises(Exception):  # frozen dataclass
        STAGE_POLICY.activation = "situ"  # type: ignore[misc]
    assert STAGE_POLICY == StagePolicy(torch.bfloat16, "silu", False)
    assert STAGE_POLICY != SORTED_POLICY
    assert hash(SORTED_POLICY) == hash(StagePolicy(torch.bfloat16, "silu", True))
    assert _policy(STAGE_POLICY.name) == STAGE_POLICY
    assert _policy(SORTED_POLICY.name) == SORTED_POLICY
    routing = ExpertRouting(
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, 1),
        compact=False,
        ordered=False,
    )
    assert (routing.compact, routing.ordered) == (False, False)


@pytest.mark.parametrize(
    ("num_experts", "policy", "num_routes", "spec_admits", "guard_names"),
    [
        (256, "bf16_silu_unsorted", 1, True, "CUDA"),
        (256, "bf16_silu_unsorted", 8, True, "CUDA"),
        (256, "bf16_silu_unsorted", 32, True, "CUDA"),
        (256, "bf16_silu_unsorted", 33, True, "CUDA"),
        (256, "bf16_silu_unsorted", 8192, True, "CUDA"),
        (256, "bf16_silu_sorted", 8, True, "CUDA"),
        (256, "bf16_silu_sorted", 8192, True, "CUDA"),
        (16, "bf16_silu_unsorted", 8, False, "num_experts must be 256"),
        (256, "fp16_silu_unsorted", 8, False, "stage_policy must be one of"),
        (256, "fp16_silu_sorted", 8, False, "got fp16_silu_sorted"),
        (256, "bf16_situ_unsorted", 8, False, "stage_policy must be one of"),
        (256, "bf16_swiglu_sorted", 8, False, "stage_policy must be one of"),
        (256, "bf16_silu_unsorted", 0, True, "num_routes=0"),
        (256, "bf16_silu_sorted", 0, True, "num_routes=0"),
    ],
)
def test_compact_spec_and_host_guard_agree_in_lock_step(
    num_experts, policy, num_routes, spec_admits, guard_names
):
    traits = _traits(num_routes, num_experts=num_experts, stage_policy=policy)
    assert spec_matches_traits(_spec(COMPACT), traits) is spec_admits
    assert spec_matches_traits(_spec(FULL), traits)  # admits every dict
    ids = torch.zeros((1, num_routes), dtype=torch.int32)
    weights = torch.zeros((1, num_routes), dtype=torch.float32)
    stage_policy = _policy(policy)
    assert expert_routing_traits(ids, num_experts, stage_policy) == traits
    reason = kernels.routing_compact_rejection(ids, weights, num_experts, stage_policy)
    assert guard_names in reason
    if guard_names == "CUDA":
        # Lock-step: on CPU tensors of an admitted call only the device clause fails.
        assert "cpu" in reason
    else:
        assert "CUDA" not in reason
    if "stage_policy must be one of" in reason:
        # The guard names the admitted policies, in one order.
        assert "bf16_silu_sorted, bf16_silu_unsorted" in reason


@pytest.mark.parametrize(
    "platform_name", ["b200_platform", "h100_platform", "mi350_platform"]
)
@pytest.mark.parametrize("num_routes", (8, 16, 32, 40, 64, 8192))
def test_ranked_selection_never_picks_the_compact_kernel(
    num_routes, platform_name, request
):
    platform = request.getfixturevalue(platform_name)
    for signature in (SIGNATURE_INT32, SIGNATURE_INT64):
        for policy in (STAGE_POLICY, SORTED_POLICY):
            selected = select_kernel(
                *OPERATOR,
                signature,
                platform=platform,
                traits=_traits(num_routes, stage_policy=policy.name),
            )
            assert selected.name == FULL and selected.impl.__name__ == FULL

    def candidates(signature) -> list[str]:
        return [
            spec.name
            for spec in KernelRegistry.get().get_for_operator(
                *OPERATOR, platform=platform, format_signature=signature
            )
        ]

    if platform_name == "mi350_platform":
        # The vendor gate filters the compact and the ordered kernels.
        assert candidates(SIGNATURE_INT32) == candidates(SIGNATURE_INT64) == [FULL]
    else:
        assert candidates(SIGNATURE_INT32) == [FULL, COMPACT, ORDERED]
        # The ordered spec declares int32 only.
        assert candidates(SIGNATURE_INT64) == [FULL, COMPACT]


def test_override_by_name_ignores_traits_and_the_cpu_host_guard_raises_without_fallback(
    selection_events,
):
    ids, weights = _routing_inputs(1, torch.int32, seed=1)
    with kernel_override(*OPERATOR, COMPACT):
        # By name: neither an admitted nor a rejected trait dict changes the result.
        for traits in (_traits(8), _traits(8192), _traits(8, num_experts=16)):
            selected = select_kernel(*OPERATOR, SIGNATURE_INT32, traits=traits)
            assert selected.name == COMPACT
        with pytest.raises(ValueError, match="CUDA") as info:
            moe_expert_routing(
                ids, weights, EXPERTS, STAGE_POLICY, solution=None, override=None
            )
    assert str(info.value).startswith(f"{COMPACT} cannot serve this call")
    assert [
        (event.kernel_name, event.source, event.override) for event in selection_events
    ] == [(COMPACT, "override", COMPACT)] * 4


def test_environment_override_outranks_the_context_and_the_argument(
    monkeypatch, fake_launches
):
    full, compact = fake_launches
    ids, weights = _routing_inputs(2, torch.int32, seed=2)
    monkeypatch.setenv(EXPERT_ROUTING_OVERRIDE_ENV, FULL)
    with kernel_override(*OPERATOR, COMPACT):
        assert (
            select_kernel(*OPERATOR, SIGNATURE_INT32, traits=_traits(16)).name == FULL
        )
        routing = moe_expert_routing(
            ids, weights, EXPERTS, STAGE_POLICY, solution=None, override=None
        )
    assert routing.compact is False and len(full.launches) == 1
    assert compact.launches == []
    monkeypatch.setenv(EXPERT_ROUTING_OVERRIDE_ENV, COMPACT)
    with pytest.raises(ValueError, match="CUDA"):
        moe_expert_routing(
            ids, weights, EXPERTS, STAGE_POLICY, solution=None, override=FULL
        )
    assert len(full.launches) == 1 and compact.launches == []


def test_compact_or_full_tables_follows_the_route_rule_on_cpu_with_fake_launches(
    fake_launches,
):
    full, compact = fake_launches
    # R = 8: packed; the tail lists the ascending active experts and their number.
    ids = torch.tensor([[255, 0, 7, 7, 3, 300, -1, 128]], dtype=torch.int32)
    route_ids, counts, is_compact = kernels._compact_or_full_tables(ids, EXPERTS)
    assert is_compact is True
    assert route_ids.shape == (EXPERTS, 8) and route_ids.dtype is torch.int32
    assert counts.shape == (EXPERTS + 8 + 1,) and counts.dtype is torch.int32
    active = counts[EXPERTS : EXPERTS + int(counts[EXPERTS + 8])]
    assert active.tolist() == [0, 3, 7, 128, 255] and int(counts[EXPERTS + 8]) == 5
    assert counts[:EXPERTS].sum().item() == 6  # 300 and -1 are absent
    assert route_ids[7, :2].tolist() == [2, 3] and int(counts[7]) == 2
    grid, args, options = compact.launches[-1]
    assert grid == (1,) and args[0].dtype is torch.int32 and args[0].is_contiguous()
    assert options == {
        "NUM_ROUTES": 8,
        "NUM_EXPERTS": EXPERTS,
        "BLOCK_ROUTES": 8,
        "num_warps": 4,
    }
    assert full.launches == []
    # R = 5: BLOCK_ROUTES is the next power of two.
    _, _, is_compact = kernels._compact_or_full_tables(ids[:, :5], EXPERTS)
    assert is_compact is True and compact.launches[-1][2]["BLOCK_ROUTES"] == 8
    # R = 40 (> 32): the full tables, the full launch geometry, compact False.
    ids40 = torch.randint(0, EXPERTS, (5, 8), dtype=torch.int32)
    route_ids, counts, is_compact = kernels._compact_or_full_tables(ids40, EXPERTS)
    assert is_compact is False and counts.shape == (EXPERTS,)
    assert route_ids.shape == (EXPERTS, 40)
    grid, args, options = full.launches[-1]
    assert grid == (EXPERTS,) and args[3] == 40
    assert options == {"BLOCK_ROUTES": 128, "num_warps": 4}
    # R = 200 (> 128): the wide block.
    ids200 = torch.randint(0, EXPERTS, (25, 8), dtype=torch.int32)
    _, _, is_compact = kernels._compact_or_full_tables(ids200, EXPERTS)
    assert is_compact is False and full.launches[-1][2]["BLOCK_ROUTES"] == 1024
    assert len(compact.launches) == 2 and len(full.launches) == 2
    # The moved host function still guards a direct call (unreachable from the
    # registered kernel for R > 32 and R == 0).
    with pytest.raises(ValueError, match="E256 and 1..32 routes"):
        kernels._routing_compact(ids40, EXPERTS)
    with pytest.raises(ValueError, match="E256 and 1..32 routes"):
        kernels._routing_compact(ids, 16)


def test_cross_operator_override_is_refused_by_the_facade(fake_launches):
    other = KernelRegistry.get().get_by_name(CROSS_OPERATOR_KERNEL)
    assert other is not None and (other.family, other.mode) == ("gemm", "decode_gemv")
    ids, weights = _routing_inputs(1, torch.int32, seed=3)
    with pytest.raises(ValueError, match="not registered under moe.expert_routing"):
        moe_expert_routing(
            ids,
            weights,
            EXPERTS,
            STAGE_POLICY,
            solution=None,
            override=CROSS_OPERATOR_KERNEL,
        )
    with kernel_override(*OPERATOR, CROSS_OPERATOR_KERNEL):
        with pytest.raises(ValueError, match="not registered under moe.expert_routing"):
            moe_expert_routing(
                ids, weights, EXPERTS, STAGE_POLICY, solution=None, override=None
            )
    full, compact = fake_launches
    assert full.launches == [] and compact.launches == []


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", (1, 4, 5, 33))
def test_facade_equals_the_inline_routing_statement_byte_for_byte_on_cpu(
    rows, dtype, fake_launches, selection_events
):
    """Under ``_moe``'s policy (no sort) the facade is the moved routing host
    function on the raw ids and hands back the input weights; under the sorted
    policy it is the per-token sort, the paired gather and the same host function."""
    full, compact = fake_launches
    topk_ids, topk_weights = _routing_inputs(rows, dtype, seed=20260921 + rows)
    saved = (topk_ids.clone(), topk_weights.clone())
    # The statement _moe used to inline (bf16.py) with the moved routing host function.
    expected_ids, expected_counts = kernels._routing(topk_ids, EXPERTS)
    assert len(full.launches) == 1
    routing = moe_expert_routing(
        topk_ids, topk_weights, EXPERTS, STAGE_POLICY, solution=None, override=None
    )
    assert isinstance(routing, ExpertRouting)
    assert routing.compact is False and routing.ordered is False
    assert torch.equal(routing.route_ids, expected_ids)
    assert torch.equal(routing.counts, expected_counts)
    assert (
        routing.route_ids.dtype is torch.int32 and routing.counts.dtype is torch.int32
    )
    assert routing.route_ids.shape == (EXPERTS, rows * TOP_K)
    assert routing.counts.shape == (EXPERTS,)
    # No sort: the weights are the input tensor itself and the launch saw the
    # unsorted ids (int32 cast, contiguous).
    assert routing.route_weights is topk_weights
    grid, args, options = full.launches[-1]
    assert grid == (EXPERTS,) and options == {
        "BLOCK_ROUTES": 128 if rows * TOP_K <= 128 else 1024,
        "num_warps": 4,
    }
    launched_ids = args[0]
    assert launched_ids.dtype is torch.int32 and launched_ids.is_contiguous()
    assert torch.equal(launched_ids, topk_ids.to(torch.int32))
    assert args[3] == rows * TOP_K
    assert torch.equal(topk_ids, saved[0]) and torch.equal(topk_weights, saved[1])
    # Ids >= E and -1 are absent from the tables.
    valid = (topk_ids >= 0) & (topk_ids < EXPERTS)
    assert int(routing.counts.sum()) == int(valid.sum())
    # The same bytes through the solution filter and the by-name override.
    for kwargs in (
        {"solution": "triton", "override": None},
        {"solution": None, "override": FULL},
    ):
        again = moe_expert_routing(
            topk_ids, topk_weights, EXPERTS, STAGE_POLICY, **kwargs
        )
        assert torch.equal(again.route_ids, expected_ids)
        assert torch.equal(again.counts, expected_counts)
        assert again.route_weights is topk_weights
    # Sorted policy: the per-token sort and the paired gather (before the int32
    # cast, in that order), then the same host function; a fresh weights tensor.
    ids, order = topk_ids.sort(dim=-1)
    expected_weights = topk_weights.gather(1, order)
    expected_ids, expected_counts = kernels._routing(ids, EXPERTS)
    sorted_routing = moe_expert_routing(
        topk_ids, topk_weights, EXPERTS, SORTED_POLICY, solution=None, override=None
    )
    assert sorted_routing.compact is False and sorted_routing.ordered is False
    assert torch.equal(sorted_routing.route_ids, expected_ids)
    assert torch.equal(sorted_routing.counts, expected_counts)
    assert torch.equal(
        sorted_routing.route_weights.view(torch.uint8),
        expected_weights.view(torch.uint8),
    )
    assert sorted_routing.route_weights is not topk_weights  # a fresh gathered tensor
    assert sorted_routing.route_weights.data_ptr() != topk_weights.data_ptr()
    assert torch.equal(full.launches[-1][1][0], ids.to(torch.int32))
    # Every id row is ascending after the sort; the duplicate stays adjacent.
    assert torch.all(ids[:, 1:] >= ids[:, :-1])
    assert int(sorted_routing.counts.sum()) == int(valid.sum())
    assert torch.equal(topk_ids, saved[0]) and torch.equal(topk_weights, saved[1])
    assert compact.launches == []
    assert {event.kernel_name for event in selection_events} == {FULL}
    assert selection_events[0].source == "ranked"


def test_metadata_validation_raises_before_selection(selection_events, fake_launches):
    ids, weights = _routing_inputs(4, torch.int32, seed=5)
    with pytest.raises(ValueError, match=r"top-k tensors must have shape"):
        moe_expert_routing(
            ids.reshape(-1),
            weights,
            EXPERTS,
            STAGE_POLICY,
            solution=None,
            override=None,
        )
    with pytest.raises(ValueError, match=r"top-k tensors must have shape"):
        moe_expert_routing(
            ids, weights[:2], EXPERTS, STAGE_POLICY, solution=None, override=None
        )
    with pytest.raises(ValueError, match="num_experts must be positive"):
        moe_expert_routing(ids, weights, 0, STAGE_POLICY, solution=None, override=None)
    with pytest.raises(ValueError, match="share one device"):
        moe_expert_routing(
            ids,
            torch.zeros(ids.shape, dtype=torch.float32, device="meta"),
            EXPERTS,
            STAGE_POLICY,
            solution=None,
            override=None,
        )
    with pytest.raises(TypeError, match="StagePolicy"):
        moe_expert_routing(
            ids, weights, EXPERTS, "bf16_silu_unsorted", solution=None, override=None
        )
    assert selection_events == []
    with pytest.raises(NoKernelFoundError, match="float32") as info:
        moe_expert_routing(
            ids.float(), weights, EXPERTS, STAGE_POLICY, solution=None, override=None
        )
    assert "moe.expert_routing" in str(info.value)
    with pytest.raises(NoKernelFoundError, match="int8"):
        moe_expert_routing(
            ids.to(torch.int8),
            weights,
            EXPERTS,
            STAGE_POLICY,
            solution=None,
            override=None,
        )
    assert selection_events == []
    full, compact = fake_launches
    assert full.launches == [] and compact.launches == []


def test_explain_selection_marks_full_and_reports_the_by_name_override(
    b200_platform,
):
    # Under _moe's policy the ordered kernel is filtered by its stage_policy trait.
    both = explain_selection(
        *OPERATOR, SIGNATURE_INT32, platform=b200_platform, traits=_traits(8)
    )
    assert "Candidates (2 matched, 3 registered)" in both
    assert f"1. {FULL}  [SELECTED]" in both
    assert f"2. {COMPACT}" in both and "Override: none" in both
    assert both.index("Filtered out:") < both.index(f"- {ORDERED}")
    # Under the sorted policy all three are candidates.
    three = explain_selection(
        *OPERATOR,
        SIGNATURE_INT32,
        platform=b200_platform,
        traits=_traits(8, stage_policy=SORTED_POLICY.name),
    )
    assert "Candidates (3 matched, 3 registered)" in three
    assert f"1. {FULL}  [SELECTED]" in three
    assert f"2. {COMPACT}" in three and f"3. {ORDERED}" in three
    # A trait the compact spec constrains (``num_experts``): the kernel is filtered out.
    one = explain_selection(
        *OPERATOR,
        SIGNATURE_INT32,
        platform=b200_platform,
        traits=_traits(8, num_experts=16),
    )
    assert "Candidates (1 matched, 3 registered)" in one
    assert f"1. {FULL}  [SELECTED]" in one
    assert one.index("Filtered out:") < one.index(f"- {COMPACT}")
    assert one.index("Filtered out:") < one.index(f"- {ORDERED}")
    # Any route count in the admitted envelope: the override marks the compact kernel
    # among the candidates (the route rule is not a trait; prefill included).
    for num_routes in (8, 32, 40, 8192):
        forced = explain_selection(
            *OPERATOR,
            SIGNATURE_INT32,
            platform=b200_platform,
            traits=_traits(num_routes),
            override=COMPACT,
        )
        assert f"Override: {COMPACT} (explicit override= argument)" in forced
        assert f"{COMPACT}  [SELECTED (override)]" in forced
        assert "not among the matched candidates" not in forced
    # A rejected envelope (16 experts): the kernel is not a candidate, so only the
    # bypass note names it.
    forced = explain_selection(
        *OPERATOR,
        SIGNATURE_INT32,
        platform=b200_platform,
        traits=_traits(8, num_experts=16),
        override=COMPACT,
    )
    assert "[SELECTED (override)]" not in forced
    assert (
        f"Override selects {COMPACT}, which is not among the matched candidates"
        in forced
    )


def test_facade_path_arguments_are_keyword_only_without_defaults():
    parameters = inspect.signature(moe_expert_routing).parameters
    assert list(parameters) == [
        "topk_ids",
        "topk_weights",
        "num_experts",
        "stage_policy",
        "solution",
        "override",
    ]
    for name in ("solution", "override"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty
    for kernel in (
        kernels.triton_moe_routing_full,
        kernels.triton_moe_routing_compact,
        kernels.triton_moe_routing_compact_ordered,
    ):
        assert list(inspect.signature(kernel).parameters) == [
            "topk_ids",
            "topk_weights",
            "num_experts",
            "stage_policy",
        ]


def _calls(node: ast.AST, predicate) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and predicate(child.func)
    ]


def test_bf16_moe_calls_the_facade_and_forwards_the_compact_flag():
    """AST guard on ``ops/moe/triton/bf16.py``: the routing functions moved out, ``_moe``
    makes exactly one facade call with ``sort_routes=False`` and explicit
    ``solution=None, override=None``, both stage launches carry
    ``COMPACT_EXPERTS=routing.compact``, both stage signatures declare the constexpr
    and the three-argument ``_combine`` consumes ``routing.route_weights``."""
    tree = ast.parse(BF16_SOURCE.read_text(), filename=str(BF16_SOURCE))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "_routing" not in functions and "_routing_kernel" not in functions
    assert (
        "_routing_compact" not in functions and "_use_compact_routing" not in functions
    )
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "tokenspeed_kernel.ops.moe.expert_routing"
        for alias in node.names
    }
    assert imported == {"StagePolicy", "moe_expert_routing"}
    for name in ("_stage1_kernel", "_stage2_kernel"):
        arguments = [argument.arg for argument in functions[name].args.args]
        assert "COMPACT_EXPERTS" in arguments
        assert arguments.index("COMPACT_EXPERTS") == arguments.index("num_tokens") + 1
    moe = functions["_moe"]
    facade_calls = _calls(
        moe, lambda func: isinstance(func, ast.Name) and func.id == "moe_expert_routing"
    )
    assert len(facade_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in facade_calls[0].keywords}
    assert set(keywords) == {"solution", "override"}
    for value in keywords.values():
        assert isinstance(value, ast.Constant) and value.value is None
    assert len(facade_calls[0].args) == 4
    policy = facade_calls[0].args[3]
    assert isinstance(policy, ast.Call) and policy.func.id == "StagePolicy"
    assert [ast.unparse(argument) for argument in policy.args] == [
        "x.dtype",
        "activation",
    ]
    policy_keywords = {keyword.arg: keyword.value for keyword in policy.keywords}
    assert list(policy_keywords) == ["sort_routes"]
    assert (
        isinstance(policy_keywords["sort_routes"], ast.Constant)
        and policy_keywords["sort_routes"].value is False
    )
    for name in ("_stage1_kernel", "_stage2_kernel"):
        launches = _calls(
            moe,
            lambda func, name=name: isinstance(func, ast.Subscript)
            and isinstance(func.value, ast.Name)
            and func.value.id == name,
        )
        assert len(launches) == 1
        flags = [
            ast.unparse(keyword.value)
            for keyword in launches[0].keywords
            if keyword.arg == "COMPACT_EXPERTS"
        ]
        assert flags == ["routing.compact"]
        positional = [ast.unparse(argument) for argument in launches[0].args]
        assert positional[3:6] == ["routing.route_ids", "routing.counts", "num_tokens"]
    combine = _calls(
        moe, lambda func: isinstance(func, ast.Name) and func.id == "_combine"
    )
    assert len(combine) == 1
    assert len(combine[0].args) == 3 and combine[0].keywords == []
    assert ast.unparse(combine[0].args[1]) == "routing.route_weights"
    # The public combine takes exactly (route_output, topk_weights, output).
    assert [argument.arg for argument in functions["_combine"].args.args] == [
        "route_output",
        "topk_weights",
        "output",
    ]


# --- the compact-ordered kernel -------------------------------------------------------


def _distinct_inputs(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, 8]`` contiguous int32 ids distinct per token and never ascending (one
    sort permutation), plus contiguous positive FP32 weights."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.empty((rows, TOP_K), dtype=torch.int64)
    for token in range(rows):
        ids[token] = torch.randperm(EXPERTS, generator=generator)[:TOP_K]
        if bool(torch.all(ids[token, 1:] > ids[token, :-1])):
            ids[token, 0], ids[token, 1] = int(ids[token, 1]), int(ids[token, 0])
    weights = torch.rand((rows, TOP_K), generator=generator) + 0.1
    return ids.to(torch.int32).contiguous(), weights.contiguous()


def _ordered_case(label: str):
    """``(num_experts, policy, ids, weights)`` of one ordered-guard case."""
    ids, weights = _distinct_inputs(1, seed=11)
    if label.startswith("admitted_rows"):
        rows = int(label.rsplit("_", 1)[1])
        ids, weights = _distinct_inputs(rows, seed=12 + rows)
        return EXPERTS, SORTED_POLICY, ids, weights
    if label == "experts_16":
        return 16, SORTED_POLICY, ids, weights
    if label.startswith("policy_"):
        return EXPERTS, _policy(label[len("policy_") :]), ids, weights
    if label == "routes_0":
        return EXPERTS, SORTED_POLICY, ids[:, :0], weights[:, :0]
    if label == "top_k_7":
        return (
            EXPERTS,
            SORTED_POLICY,
            ids[:, :7].contiguous(),
            weights[:, :7].contiguous(),
        )
    if label == "int64_ids":
        return EXPERTS, SORTED_POLICY, ids.to(torch.int64), weights
    if label == "strided_ids":
        return (
            EXPERTS,
            SORTED_POLICY,
            torch.zeros(1, 16, dtype=torch.int32)[:, ::2],
            weights,
        )
    if label == "fp16_weights":
        return EXPERTS, SORTED_POLICY, ids, weights.to(torch.float16)
    if label == "strided_weights":
        return EXPERTS, SORTED_POLICY, ids, torch.zeros(1, 16)[:, ::2]
    raise AssertionError(label)


@pytest.mark.parametrize(
    ("label", "ordered_admits", "compact_admits", "guard_names"),
    [
        ("admitted_rows_1", True, True, "CUDA"),
        ("admitted_rows_2", True, True, "CUDA"),
        ("admitted_rows_3", True, True, "CUDA"),
        ("admitted_rows_4", True, True, "CUDA"),
        ("admitted_rows_5", True, True, "CUDA"),
        ("admitted_rows_1024", True, True, "CUDA"),
        ("experts_16", False, False, "num_experts must be 256"),
        ("policy_fp16_silu_sorted", False, False, "stage_policy must be one of"),
        ("policy_bf16_situ_sorted", False, False, "stage_policy must be one of"),
        # The compact envelope admits the unsorted policy; the ordered kernel does
        # not (the in-CTA sort is the sorted prep).
        ("policy_bf16_silu_unsorted", False, True, "sort_routes must be True"),
        ("routes_0", True, True, "num_routes=0"),
        ("top_k_7", True, True, "top_k must be 8"),
        ("int64_ids", True, True, "topk_ids must be int32"),
        ("strided_ids", True, True, "topk_ids must be contiguous"),
        ("fp16_weights", True, True, "topk_weights must be float32"),
        ("strided_weights", True, True, "topk_weights must be contiguous"),
    ],
)
def test_ordered_spec_and_host_guard_agree_in_lock_step(
    label, ordered_admits, compact_admits, guard_names
):
    num_experts, stage_policy, ids, weights = _ordered_case(label)
    traits = expert_routing_traits(ids, num_experts, stage_policy)
    assert spec_matches_traits(_spec(ORDERED), traits) is ordered_admits
    assert spec_matches_traits(_spec(COMPACT), traits) is compact_admits
    assert spec_matches_traits(_spec(FULL), traits)
    reason = kernels.routing_compact_ordered_rejection(
        ids, weights, num_experts, stage_policy
    )
    assert guard_names in reason
    if guard_names == "CUDA":
        # Lock-step: on CPU tensors of an admitted call only the device clause fails.
        assert "cpu" in reason
    else:
        assert "CUDA" not in reason  # the named clause precedes the device clauses
    if label == "int64_ids":
        # The int64 refusal is also declared at the spec level (signature).
        assert not _spec(ORDERED).supports_format_signature(SIGNATURE_INT64)
        assert _spec(COMPACT).supports_format_signature(SIGNATURE_INT64)
    # The compact guard is the envelope and the device clauses composed, unchanged;
    # the ordered guard adds the sort clause and the layout clauses in between.
    envelope = kernels._envelope_rejection(ids, num_experts, stage_policy)
    sort = kernels._ordered_sort_rejection(stage_policy)
    device = kernels._device_rejection(ids, weights)
    layout = kernels._ordered_layout_rejection(ids, weights)
    assert kernels.routing_compact_rejection(
        ids, weights, num_experts, stage_policy
    ) == (envelope or device)
    assert reason == (envelope or sort or layout or device)
    assert "CUDA" in device and "cpu" in device
    assert (sort == "") is stage_policy.sort_routes
    if label in (
        "top_k_7",
        "int64_ids",
        "strided_ids",
        "fp16_weights",
        "strided_weights",
    ):
        assert envelope == "" and sort == "" and layout == reason
        assert (
            kernels.routing_compact_rejection(ids, weights, num_experts, stage_policy)
            == device
        )  # the compact kernel admits the layout (device clause only)
    if label == "policy_bf16_silu_unsorted":
        assert envelope == "" and sort == reason and "got bf16_silu_unsorted" in reason
        assert (
            kernels.routing_compact_rejection(ids, weights, num_experts, stage_policy)
            == device
        )  # the compact kernel admits the unsorted policy (device clause only)
    if label.startswith("admitted") or label == "routes_0":
        assert layout == "" if label != "routes_0" else True


def test_ordered_route_rule_on_cpu_with_fake_launches(
    fake_launches, fake_ordered_launch
):
    """``_ordered_tables``: rows 1/4 (8/32 routes) launch the ordered kernel once with
    the ordered kernel's geometry and return ``(True, True)``; rows 2/3 run the torch
    sort and the compact launch, rows 5/8 the full launch, byte-equal to the compact
    and the full kernels' paths (the same function objects)."""
    full, compact = fake_launches
    for rows in (1, 4):
        ids, weights = _distinct_inputs(rows, seed=100 + rows)
        saved = (ids.clone(), weights.clone())
        routes = rows * TOP_K
        routing = kernels._ordered_tables(ids, weights, EXPERTS, SORTED_POLICY)
        assert isinstance(routing, ExpertRouting)
        assert (routing.compact, routing.ordered) == (True, True)
        assert len(fake_ordered_launch.launches) == 1
        grid, args, options = fake_ordered_launch.launches[-1]
        assert grid == (1,) and args[0] is ids and args[3] is weights
        assert args[1] is routing.route_ids and args[2] is routing.counts
        assert args[4] is routing.route_weights
        assert args[4].shape == weights.shape and args[4].dtype is torch.float32
        assert args[4].is_contiguous() and args[4].data_ptr() != weights.data_ptr()
        assert options == {
            "NUM_ROUTES": routes,
            "NUM_EXPERTS": EXPERTS,
            "BLOCK_ROUTES": routes,  # next_power_of_2(R) == R for 8 and 32
            "num_warps": 4,
        }
        assert routing.route_ids.shape == (EXPERTS, routes)
        assert routing.counts.shape == (EXPERTS + routes + 1,)
        assert routing.route_ids.dtype is routing.counts.dtype is torch.int32
        # Tables and counts equal the compact kernel's on the sorted ids; the paired
        # weights equal the gather for distinct ids (one permutation).
        sorted_ids, order = ids.sort(dim=-1)
        expected_ids, expected_counts = kernels._routing_compact(sorted_ids, EXPERTS)
        assert torch.equal(routing.route_ids, expected_ids)
        assert torch.equal(routing.counts, expected_counts)
        assert torch.equal(
            routing.route_weights.view(torch.uint8),
            weights.gather(1, order).view(torch.uint8),
        )
        assert torch.equal(ids, saved[0]) and torch.equal(weights, saved[1])
        fake_ordered_launch.launches.clear()
        compact.launches.clear()
    # A tie row: the paired weights follow the network permutation (not the identity),
    # the tables are permutation-independent.
    tie = torch.tensor([TIE_ROW], dtype=torch.int32)
    weights = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8)
    routing = kernels._ordered_tables(tie, weights, EXPERTS, SORTED_POLICY)
    keys, permutation = header_sort(TIE_ROW)
    assert keys == sorted(TIE_ROW) and permutation != list(range(8))
    assert routing.route_weights.tolist() == [[float(1 + p) for p in permutation]]
    expected_ids, expected_counts = kernels._routing_compact(
        tie.sort(dim=-1).values, EXPERTS
    )
    assert torch.equal(routing.route_ids, expected_ids)
    assert torch.equal(routing.counts, expected_counts)
    assert int(routing.counts[7]) == 4 and int(routing.counts[EXPERTS + 8]) == 3
    fake_ordered_launch.launches.clear()
    compact.launches.clear()
    # Rows 2/3: the torch sort ran (the launch saw sorted ids) and the compact tables.
    for rows in (2, 3):
        ids, weights = _routing_inputs(rows, torch.int32, seed=200 + rows)
        routing = kernels._ordered_tables(ids, weights, EXPERTS, SORTED_POLICY)
        assert (routing.compact, routing.ordered) == (True, False)
        assert fake_ordered_launch.launches == [] and full.launches == []
        assert len(compact.launches) == 1
        launched = compact.launches[-1][1][0]
        assert torch.equal(launched, ids.sort(dim=-1).values)
        prepped_ids, prepped_weights = kernels._prep(ids, weights, SORTED_POLICY)
        expected_ids, expected_counts, is_compact = kernels._compact_or_full_tables(
            prepped_ids, EXPERTS
        )
        assert is_compact is True
        assert torch.equal(routing.route_ids, expected_ids)
        assert torch.equal(routing.counts, expected_counts)
        assert torch.equal(
            routing.route_weights.view(torch.uint8), prepped_weights.view(torch.uint8)
        )
        compact.launches.clear()
    # Rows 5/8: the full launch, the full kernel's bytes.
    for rows in (5, 8):
        ids, weights = _routing_inputs(rows, torch.int32, seed=300 + rows)
        routing = kernels._ordered_tables(ids, weights, EXPERTS, SORTED_POLICY)
        assert (routing.compact, routing.ordered) == (False, False)
        assert fake_ordered_launch.launches == [] and compact.launches == []
        assert len(full.launches) == 1 and full.launches[-1][0] == (EXPERTS,)
        assert routing.counts.shape == (EXPERTS,)
        prepped_ids, prepped_weights = kernels._prep(ids, weights, SORTED_POLICY)
        expected_ids, expected_counts, is_compact = kernels._compact_or_full_tables(
            prepped_ids, EXPERTS
        )
        assert is_compact is False
        assert torch.equal(routing.route_ids, expected_ids)
        assert torch.equal(routing.counts, expected_counts)
        assert torch.equal(
            routing.route_weights.view(torch.uint8), prepped_weights.view(torch.uint8)
        )
        full.launches.clear()
    # ``_prep`` under _moe's policy is the identity on both tensors.
    ids, weights = _routing_inputs(2, torch.int32, seed=400)
    prepped_ids, prepped_weights = kernels._prep(ids, weights, STAGE_POLICY)
    assert prepped_ids is ids and prepped_weights is weights
    # The moved host function guards a direct call.
    ids16, weights16 = _distinct_inputs(2, seed=7)
    with pytest.raises(ValueError, match="8 or 32 routes"):
        kernels._routing_compact_ordered(ids16, weights16, EXPERTS)
    ids8, weights8 = _distinct_inputs(1, seed=8)
    with pytest.raises(ValueError, match="E256 and 8 or 32 routes"):
        kernels._routing_compact_ordered(ids8, weights8, 16)
    with pytest.raises(ValueError, match="contiguous int32 ids"):
        kernels._routing_compact_ordered(ids8.to(torch.int64), weights8, EXPERTS)
    with pytest.raises(ValueError, match="contiguous float32 weights"):
        kernels._routing_compact_ordered(ids8, torch.zeros(1, 16)[:, ::2], EXPERTS)
    assert fake_ordered_launch.launches == []


def test_ordered_override_by_name_ignores_traits_and_the_cpu_host_guard_raises_without_fallback(
    selection_events, fake_launches, fake_ordered_launch
):
    ids, weights = _distinct_inputs(1, seed=21)
    with kernel_override(*OPERATOR, ORDERED):
        # By name: neither the traits nor the (undeclared) int64 signature filter.
        for traits in (_traits(8), _traits(8192), _traits(8, num_experts=16)):
            assert (
                select_kernel(*OPERATOR, SIGNATURE_INT32, traits=traits).name == ORDERED
            )
        assert (
            select_kernel(*OPERATOR, SIGNATURE_INT64, traits=_traits(8)).name == ORDERED
        )
        with pytest.raises(ValueError, match="CUDA") as info:
            moe_expert_routing(
                ids, weights, EXPERTS, SORTED_POLICY, solution=None, override=None
            )
        assert str(info.value).startswith(f"{ORDERED} cannot serve this call")
        # int64 ids: the alias clause precedes the device clause.
        with pytest.raises(ValueError, match="topk_ids must be int32") as info:
            moe_expert_routing(
                ids.to(torch.int64),
                weights,
                EXPERTS,
                SORTED_POLICY,
                solution=None,
                override=None,
            )
        assert "CUDA" not in str(info.value)
        # _moe's policy (no sort): the sort clause precedes the device clause, so the
        # ordered override on the default MoE path is refused on any device.
        with pytest.raises(ValueError, match="sort_routes must be True") as info:
            moe_expert_routing(
                ids, weights, EXPERTS, STAGE_POLICY, solution=None, override=None
            )
        assert "CUDA" not in str(info.value)
    events = [
        (event.kernel_name, event.source, event.override) for event in selection_events
    ]
    assert events == [(ORDERED, "override", ORDERED)] * 7
    full, compact = fake_launches
    assert full.launches == compact.launches == fake_ordered_launch.launches == []


def test_environment_override_selects_the_ordered_kernel_and_outranks_the_context(
    monkeypatch, fake_launches, fake_ordered_launch
):
    ids, weights = _distinct_inputs(1, seed=22)
    monkeypatch.setenv(EXPERT_ROUTING_OVERRIDE_ENV, ORDERED)
    with kernel_override(*OPERATOR, COMPACT):
        assert (
            select_kernel(*OPERATOR, SIGNATURE_INT32, traits=_traits(8)).name == ORDERED
        )
        with pytest.raises(ValueError, match="CUDA") as info:
            moe_expert_routing(
                ids, weights, EXPERTS, SORTED_POLICY, solution=None, override=FULL
            )
    assert str(info.value).startswith(f"{ORDERED} cannot serve this call")
    full, compact = fake_launches
    assert full.launches == compact.launches == fake_ordered_launch.launches == []


def _kernel_functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(KERNELS_SOURCE.read_text(), filename=str(KERNELS_SOURCE))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_sort_network_is_the_documented_bitonic_schedule():
    """``_sort8`` is the 15-stage ATen network (sizes 2..16 then the final merge) and
    ``_swap`` carries the SortUtils comparator; the stdlib replay agrees with the
    independent lane form, sorts, and is not stable (the documented tie contract)."""
    functions = _kernel_functions()
    swaps = [
        node.value
        for node in functions["_sort8"].body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_swap"
    ]
    assert len(swaps) == 15
    assert (
        tuple(tuple(ast.literal_eval(arg) for arg in call.args[-2:]) for call in swaps)
        == NETWORK
    )
    assert NETWORK == (
        (1, 2),
        (2, 4),
        (1, 4),
        (4, 8),
        (2, 8),
        (1, 8),
        (8, 16),
        (4, 16),
        (2, 16),
        (1, 16),
        (16, 0),
        (8, 0),
        (4, 0),
        (2, 0),
        (1, 0),
    )
    swap_source = ast.get_source_segment(KERNELS_SOURCE.read_text(), functions["_swap"])
    assert "    comparison = ((a < b) & valid_a) | ~valid_b\n" in swap_source
    assert "not a stable or lexicographic sort" in swap_source
    sort_source = ast.get_source_segment(
        KERNELS_SOURCE.read_text(), functions["_sort8"]
    )
    assert "tl.load(ids_ptr + row * 8 + lane, mask=valid, other=0)" in sort_source
    assert "lane < 8" in sort_source and "tl.arange(0, 32)" in sort_source
    cases = list(product((-1, 7), repeat=8))
    randomizer = random.Random(716)
    cases += [
        tuple(randomizer.choice(SORT_ALPHABET) for _ in range(8)) for _ in range(128)
    ]
    cases += [tuple(range(8)), tuple(reversed(range(8)))]
    for values in cases:
        keys, indices = header_sort(values)
        assert (keys, indices) == lane_sort(values)
        assert keys == sorted(values)
        assert sorted(indices) == list(range(8))
        assert keys == [values[index] for index in indices]
    # Torch's dispatch is not stable on ties, and the replay reproduces that.
    assert header_sort([7] * 8)[1] != list(range(8))
    assert header_sort(list(TIE_ROW))[0] == sorted(TIE_ROW)


def test_ordered_kernel_source_is_the_compact_kernel_plus_the_sort_and_gather():
    """Reverse AST: dropping the two pointer parameters and replacing the six
    ordered-kernel statements by the compact kernel's ``selected`` load gives the
    compact kernel."""
    functions = _kernel_functions()
    compact_kernel = functions["_routing_compact_kernel"]
    ordered = deepcopy(functions["_routing_compact_ordered_kernel"])
    parameters = [argument.arg for argument in ordered.args.args]
    assert parameters[:5] == [
        "topk_ids_ptr",
        "expert_route_ids_ptr",
        "expert_counts_ptr",
        *ORDERED_POINTERS,
    ]
    assert parameters[5:] == ["NUM_ROUTES", "NUM_EXPERTS", "BLOCK_ROUTES"]
    ordered.args.args = [
        argument
        for argument in ordered.args.args
        if argument.arg not in ORDERED_POINTERS
    ]
    first = next(
        index
        for index, node in enumerate(ordered.body)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "(ordered, permutation)"
    )
    matches = next(
        index
        for index, node in enumerate(ordered.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "matches"
    )
    replaced = ordered.body[first:matches]
    assert tuple(ast.unparse(node) for node in replaced) == ORDERED_STATEMENTS
    selected = next(
        index
        for index, node in enumerate(compact_kernel.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "selected"
    )
    assert selected == first
    ordered.body[first:matches] = [deepcopy(compact_kernel.body[selected])]
    ordered.name = compact_kernel.name
    assert ast.dump(ordered) == ast.dump(compact_kernel)
    # Both are ``triton.jit`` objects with the two hosts' geometry: one CTA, R in {8, 32}.
    assert [
        ast.unparse(d)
        for d in functions["_routing_compact_ordered_kernel"].decorator_list
    ] == ["triton.jit"]
    assert [ast.unparse(d) for d in compact_kernel.decorator_list] == ["triton.jit"]
    host = ast.get_source_segment(
        KERNELS_SOURCE.read_text(), functions["_routing_compact_ordered"]
    )
    assert "_routing_compact_ordered_kernel[(1,)](" in host
    assert "BLOCK_ROUTES=triton.next_power_of_2(routes)" in host
    assert "**_LAUNCH_OPTIONS_COMPACT" in host
    assert "paired = torch.empty_like(weights)" in host


def test_ordered_kernel_registration_and_facade_flags_on_cpu():
    """The facade's trait dict is unchanged (no top-k/dtype/layout trait), the ordered
    constants are the documented rule, the ordered spec's traits are the compact
    spec's restricted to the sorted policy and the first two kernels keep both id
    signatures."""
    ids, weights = _distinct_inputs(4, seed=31)
    traits = expert_routing_traits(ids, EXPERTS, SORTED_POLICY)
    assert list(traits) == ["num_experts", "num_routes", "stage_policy"]
    assert traits == {
        "num_experts": 256,
        "num_routes": 32,
        "stage_policy": "bf16_silu_sorted",
    }
    assert expert_routing_traits(ids, EXPERTS, STAGE_POLICY)["stage_policy"] == (
        "bf16_silu_unsorted"
    )
    assert ORDERED_ROUTES == frozenset({8, 32}) and ORDERED_TOP_K == 8
    assert {routes // ORDERED_TOP_K for routes in ORDERED_ROUTES} == {1, 4}
    assert all(routes <= MAX_COMPACT_ROUTES for routes in ORDERED_ROUTES)
    assert _spec(ORDERED).traits == {
        "num_experts": frozenset({COMPACT_NUM_EXPERTS}),
        "stage_policy": frozenset({ORDERED_STAGE_POLICY}),
    }
    assert _spec(COMPACT).traits == {
        "num_experts": frozenset({COMPACT_NUM_EXPERTS}),
        "stage_policy": COMPACT_STAGE_POLICIES,
    }
    assert (
        _spec(FULL).format_signatures
        == _spec(COMPACT).format_signatures
        == EXPERT_ROUTING_SIGNATURES
        == frozenset({SIGNATURE_INT32, SIGNATURE_INT64})
    )
    assert _spec(ORDERED).format_signatures == kernels._SIGNATURES_INT32_ONLY
    assert kernels._SIGNATURES_INT32_ONLY == frozenset({SIGNATURE_INT32})
    assert ORDERED == "triton_moe_routing_compact_ordered"
    assert kernels.triton_moe_routing_compact_ordered.__name__ == ORDERED
    # The sorted trait dict admits the ordered spec for every route count: the shape
    # rule is the return-value rule, not a filter. _moe's unsorted dict never does,
    # while the compact spec admits both.
    for rows in (1, 2, 3, 4, 5, 1024):
        ids, _ = _distinct_inputs(rows, seed=40 + rows)
        sorted_traits = expert_routing_traits(ids, EXPERTS, SORTED_POLICY)
        unsorted_traits = expert_routing_traits(ids, EXPERTS, STAGE_POLICY)
        assert spec_matches_traits(_spec(ORDERED), sorted_traits)
        assert not spec_matches_traits(_spec(ORDERED), unsorted_traits)
        assert spec_matches_traits(_spec(COMPACT), sorted_traits)
        assert spec_matches_traits(_spec(COMPACT), unsorted_traits)
