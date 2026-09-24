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

"""``moe.expert_routing`` on an NVIDIA device: the compact-ordered kernel against the compact
kernel -- tables, counts and ``route_weights`` byte-equal on the 17-case rows x ids axis (the
five tie kinds included), the in-CTA sort's permutation bit for bit against the documented
bitonic network and Torch's own dispatch, the shape rule (16/24 routes torch-sorted, more
than 32 the full tables), stage-output byte equality of the shared BF16 MoE apply under the
two kernels, CUDA-graph replay, rejected calls and ranking.

The in-tree kernel compiles to its own Triton hash, so its correctness is established here
anew. The compact kernel is the reference of every byte comparison
(``test_expert_routing_cuda.py`` proves it equal to the full kernel); on tie rows (duplicate
or invalid ids) the sort permutation decides which FP32 weight lands in which sorted slot, so
``route_weights`` byte equality on those rows pins the in-CTA network to the Torch dispatch
the torch-sorted kernels run -- a future dispatch change fails here first. Every ordered call
runs under the sorted policy (``sort_routes=True``); ``_moe`` passes ``sort_routes=False``,
so the stage-output and graph tests bind the facade name in ``bf16`` to the sorted policy
(``moe_apply`` then runs the unchanged stage kernels on the sorted-prep tables) and the
refusal test shows the ordered override refused, without fallback, on the default path.
"""

from __future__ import annotations

import random

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.expert_routing import (
    EXPERT_ROUTING_FAMILY,
    EXPERT_ROUTING_MODE,
    MAX_COMPACT_ROUTES,
    ORDERED_ROUTES,
    ORDERED_TOP_K,
    TRITON_MOE_ROUTING_COMPACT,
    TRITON_MOE_ROUTING_COMPACT_ORDERED,
    TRITON_MOE_ROUTING_FULL,
    StagePolicy,
    moe_expert_routing,
)
from tokenspeed_kernel.ops.moe.triton import bf16
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    SelectionEvent,
    _witnessed_overrides,
    add_selection_listener,
    explain_selection,
    kernel_override,
    remove_selection_listener,
    select_kernel,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not Platform.get().is_nvidia,
    reason="requires an NVIDIA CUDA device",
)
pytestmark = requires_cuda

OPERATOR = (EXPERT_ROUTING_FAMILY, EXPERT_ROUTING_MODE)
FULL, COMPACT = TRITON_MOE_ROUTING_FULL, TRITON_MOE_ROUTING_COMPACT
ORDERED = TRITON_MOE_ROUTING_COMPACT_ORDERED
SIGNATURE_INT32 = format_signature(topk_ids=dense_tensor_format(torch.int32))
SIGNATURE_INT64 = format_signature(topk_ids=dense_tensor_format(torch.int64))
DEVICE = torch.device("cuda", 0)
SEED = 20260921
EXPERTS, TOP_K = 256, 8
# The rows axis at top-8: 8 and 32 routes take the in-CTA sort, 16 and 24 the torch sort
# with the compact tables, 40/64/256 the full tables.
ROWS_ORDERED = (1, 4)
ROWS_TORCH_SORTED = (2, 3)
ROWS_FULL_TABLES = (5, 8, 32)
ROWS_AXIS = (1, 2, 3, 4, 5, 8, 32)
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
# The five tie kinds (the route states at E = 256, top-k 8), rotated per token by
# ``token % 8``.
TIE_IDS: dict[str, tuple[int, ...]] = {
    "all_same": (255,) * 8,
    "duplicates": (7, 0, 7, 255, 0, 7, 255, 7),
    "mixed_invalid": (-1, 256, 7, 7, 0, 255, 257, 19),
    "all_invalid": (-1, 256, -2, 257, -3, 258, -4, 259),
    "signed_extremes": (INT32_MIN, INT32_MAX, 0, 255, -1, 256, 0, -1),
}
IDS_DISTINCT = "distinct"
IDS_KINDS = (IDS_DISTINCT, *TIE_IDS)
TIE_ROWS = (1, 4)
AXIS = tuple(
    (rows, kind)
    for rows in ROWS_AXIS
    for kind in (IDS_KINDS if rows in TIE_ROWS else (IDS_DISTINCT,))
)
assert len(AXIS) == 17
# The eight FP32 bit patterns of the routing gate: +0, -0, +1, -1, +-2^-24, the smallest
# subnormal, a NaN payload -- copied by the paired gather bit for bit.
BIT_PATTERNS = (
    0x00000000,
    0x80000000,
    0x3F800000,
    0xBF800000,
    0x33800000,
    0xB3800000,
    0x00000001,
    0x7FC12345,
)
SORT_ALPHABET = (INT32_MIN, -1, 0, 7, 255, 256, INT32_MAX)
# Stage byte-equality geometry (the compact GPU test's): 256 experts for the guards, hidden
# 256 and intermediate 64 keep the BF16 weights small.
HIDDEN, INTERMEDIATE = 256, 64
SAME_EXPERTS_ROWS = 32
WEIGHT_STD = 0.02
# The sorted policy the ordered kernel is qualified for, and the policy _moe passes.
SORTED_POLICY = StagePolicy(torch.bfloat16, "silu", True)
STAGE_POLICY = StagePolicy(torch.bfloat16, "silu", False)
STAGE_CASES = tuple((rows, IDS_DISTINCT) for rows in (1, 2, 3, 4, 8, 32)) + tuple(
    (rows, kind) for rows in ROWS_ORDERED for kind in TIE_IDS
)


@pytest.fixture(autouse=True)
def selection_state():
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


@pytest.fixture
def sorted_facade(monkeypatch):
    """Bind the facade name ``_moe`` resolves to the sorted policy. ``_moe`` passes
    ``sort_routes=False``, under which the ordered kernel is refused; a caller that
    sorts is emulated here so ``moe_apply`` runs the unchanged stage kernels and
    combine on the sorted-prep tables of whichever kernel the override names."""

    def facade(
        topk_ids, topk_weights, num_experts, stage_policy, *, solution, override
    ):
        assert stage_policy == STAGE_POLICY  # what _moe passes on this tree
        return moe_expert_routing(
            topk_ids,
            topk_weights,
            num_experts,
            SORTED_POLICY,
            solution=solution,
            override=override,
        )

    monkeypatch.setattr(bf16, "moe_expert_routing", facade)


def _operator_events(events: list[SelectionEvent]) -> list[tuple[str, str]]:
    return [
        (event.kernel_name, event.source)
        for event in events
        if (event.family, event.mode) == OPERATOR
    ]


def _network() -> tuple[tuple[int, int], ...]:
    """The 15 ``(STRIDE, SIZE)`` stages of ATen's 32-item bitonic network."""
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
    """Literal sequential replay of the pinned SortUtils bitonicSort + bitonicSwap on eight
    valid keys padded to 32; returns the sorted keys and the slot permutation (Torch's tie
    permutation, not a stable sort)."""
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


def _header_permutation(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(keys, permutation)`` of every row of ``ids`` by the network replay, on the device."""
    replayed = [header_sort(row) for row in ids.tolist()]
    keys = torch.tensor(
        [keys for keys, _ in replayed], dtype=torch.int32, device=DEVICE
    )
    order = torch.tensor(
        [order for _, order in replayed], dtype=torch.int64, device=DEVICE
    )
    return keys, order


def _canonical(route_ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """The table with slots at or beyond the per-expert count set to -1."""
    slots = torch.arange(route_ids.shape[1], device=route_ids.device)
    written = slots[None, :] < counts[: route_ids.shape[0], None]
    return route_ids.masked_fill(~written, -1)


def _canonical_counts(counts: torch.Tensor, num_routes: int) -> torch.Tensor:
    """``counts`` with the unwritten slots of a compact tail (between the active count and
    ``min(E, R)``) set to -1, so the whole vector is comparable byte for byte."""
    if counts.shape[0] == EXPERTS:
        return counts
    assert counts.shape == (EXPERTS + min(EXPERTS, num_routes) + 1,)
    active = int(counts[EXPERTS + min(EXPERTS, num_routes)])
    canonical = counts.clone()
    canonical[EXPERTS + active : EXPERTS + min(EXPERTS, num_routes)] = -1
    return canonical


def _traits(
    num_routes: int, num_experts: int = EXPERTS, policy: StagePolicy = SORTED_POLICY
) -> dict[str, int | str]:
    return {
        "num_experts": num_experts,
        "num_routes": num_routes,
        "stage_policy": policy.name,
    }


def _route_ids(
    rows: int, same_experts: bool, generator: torch.Generator
) -> torch.Tensor:
    """int32 ``[rows, 8]`` ids, distinct per token and never ascending; token 0 covers
    experts 0 and 255; with ``same_experts`` every token uses token 0's experts."""
    ids = torch.empty((rows, TOP_K), dtype=torch.int64)
    middle = torch.randperm(EXPERTS - 2, generator=generator)[: TOP_K - 2] + 1
    first = torch.cat([torch.tensor([EXPERTS - 1, 0]), middle])
    ids[0] = first[torch.randperm(TOP_K, generator=generator)]
    for token in range(1, rows):
        if same_experts:
            ids[token] = ids[0][torch.randperm(TOP_K, generator=generator)]
        else:
            ids[token] = torch.randperm(EXPERTS, generator=generator)[:TOP_K]
    for token in range(rows):
        if bool(torch.all(ids[token, 1:] > ids[token, :-1])):
            ids[token, 0], ids[token, 1] = int(ids[token, 1]), int(ids[token, 0])
    return ids.to(torch.int32)


def _tie_ids(kind: str, rows: int) -> torch.Tensor:
    values = list(TIE_IDS[kind])
    rotated = [values[token % 8 :] + values[: token % 8] for token in range(rows)]
    return torch.tensor(rotated, dtype=torch.int32)


def _case_ids(rows: int, kind: str, seed: int) -> torch.Tensor:
    """The ids of one axis case: distinct unsorted ids (rows 32 all on the same experts)
    or a tie pattern rotated per token; contiguous int32 on the device."""
    if kind == IDS_DISTINCT:
        generator = torch.Generator().manual_seed(seed)
        ids = _route_ids(rows, rows == SAME_EXPERTS_ROWS, generator)
    else:
        ids = _tie_ids(kind, rows)
    return ids.contiguous().to(DEVICE)


def _route_weights(ids: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """FP32 sigmoid scores of ``1.5 + randn`` renormalized."""
    scores = torch.sigmoid(torch.randn(ids.shape, generator=generator) + 1.5)
    return (scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)).contiguous()


def _case_weights(ids: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return _route_weights(ids.cpu(), generator).to(DEVICE)


def _bit_pattern_rows(rows: int, roll: int) -> torch.Tensor:
    """int32 ``[rows, 8]`` of the eight bit patterns rolled by ``roll`` (the per-case
    roll), the same row for every token."""
    signed = [value if value < 2**31 else value - 2**32 for value in BIT_PATTERNS]
    bits = torch.tensor(signed, dtype=torch.int32).roll(roll)
    return bits.repeat(rows, 1).contiguous().to(DEVICE)


def _routing(
    name: str,
    ids: torch.Tensor,
    weights: torch.Tensor,
    num_experts=EXPERTS,
    policy: StagePolicy = SORTED_POLICY,
):
    with kernel_override(*OPERATOR, name):
        return moe_expert_routing(
            ids, weights, num_experts, policy, solution=None, override=None
        )


def _assert_same_tables(ordered, reference, num_routes: int) -> None:
    assert ordered.route_ids.shape == reference.route_ids.shape == (EXPERTS, num_routes)
    assert ordered.counts.shape == reference.counts.shape
    assert torch.equal(
        _canonical_counts(ordered.counts, num_routes),
        _canonical_counts(reference.counts, num_routes),
    )
    assert torch.equal(
        _canonical(ordered.route_ids, ordered.counts),
        _canonical(reference.route_ids, reference.counts),
    )
    assert ordered.route_weights.dtype is reference.route_weights.dtype is torch.float32
    assert torch.equal(
        ordered.route_weights.view(torch.int32),
        reference.route_weights.view(torch.int32),
    )


@pytest.mark.parametrize(("rows", "kind"), AXIS)
def test_ordered_tables_counts_and_weights_equal_compact_on_the_17_case_axis(
    rows, kind, selection_events
):
    """The ordered kernel's flags follow the expected table (rows 1/4 ``(True, True)``,
    2/3 ``(True, False)``, 5/8/32 ``(False, False)``; the compact kernel reports
    ``(admitted, False)``) and its counts, canonicalized tables and ``route_weights`` equal
    the compact kernel's byte for byte on every case -- the tie kinds included, which pins
    the in-CTA network's tie permutation to the Torch dispatch."""
    num_routes = rows * TOP_K
    ids = _case_ids(rows, kind, SEED + rows)
    weights = _case_weights(ids, SEED + 100 + rows)
    saved = (ids.clone(), weights.clone())
    compact = _routing(COMPACT, ids, weights)
    ordered = _routing(ORDERED, ids, weights)
    torch.cuda.synchronize(DEVICE)
    expected_compact = num_routes <= MAX_COMPACT_ROUTES
    expected_ordered = num_routes in ORDERED_ROUTES
    assert expected_ordered == (rows in ROWS_ORDERED)
    assert (compact.compact, compact.ordered) == (expected_compact, False)
    assert (ordered.compact, ordered.ordered) == (expected_compact, expected_ordered)
    _assert_same_tables(ordered, compact, num_routes)
    assert ordered.route_weights.shape == weights.shape
    assert ordered.route_weights.data_ptr() != weights.data_ptr()  # a fresh tensor
    assert torch.equal(ids, saved[0]) and torch.equal(weights, saved[1])
    # The expectation is derived from the constructed ids: invalid ids are absent.
    valid = ids[(ids >= 0) & (ids < EXPERTS)]
    assert int(ordered.counts[:EXPERTS].sum()) == valid.numel()
    if kind == "all_invalid":
        assert valid.numel() == 0
    if expected_compact:
        assert int(ordered.counts[EXPERTS + num_routes]) == int(valid.unique().numel())
    # Distinct ids: one permutation, so the weights equal the plain gather.
    sorted_ids, order = ids.sort(dim=-1)
    if kind == IDS_DISTINCT:
        assert torch.equal(ordered.route_weights, weights.gather(1, order))
    # Every written slot is a position of that expert in the sorted flattened ids.
    flat = sorted_ids.reshape(-1)
    table = _canonical(ordered.route_ids, ordered.counts)
    for expert in torch.nonzero(ordered.counts[:EXPERTS] > 0).squeeze(1).tolist():
        positions = table[expert, : int(ordered.counts[expert])].to(torch.int64)
        assert torch.equal(flat[positions], torch.full_like(positions, expert))
    assert _operator_events(selection_events) == [
        (COMPACT, "override"),
        (ORDERED, "override"),
    ]


@pytest.mark.parametrize("kind", IDS_KINDS)
@pytest.mark.parametrize("rows", ROWS_ORDERED)
def test_ordered_weights_follow_the_documented_network_permutation_bit_for_bit(
    rows, kind
):
    """On the ordered rows the paired weights are ``bits.gather(1, header_perm)`` -- the
    permutation of the documented bitonic network replayed on the host -- and equal to
    ``bits.gather(1, torch_perm)``; -0, the subnormal and the NaN payload are copied bit for
    bit. A disagreement between the two references names the Torch dispatch."""
    ids = _case_ids(rows, kind, SEED + 10 + rows)
    header_keys, header_perm = _header_permutation(ids)
    torch_keys, torch_perm = ids.sort(dim=-1)
    if not (
        torch.equal(torch_keys, header_keys) and torch.equal(torch_perm, header_perm)
    ):
        pytest.fail(
            f"Torch sort dispatch differs from the documented network on {kind} rows={rows}: "
            f"torch {torch_perm.tolist()} vs network {header_perm.tolist()}"
        )
    for roll in range(len(BIT_PATTERNS)):
        bits = _bit_pattern_rows(rows, roll)
        weights = bits.view(torch.float32)
        ordered = _routing(ORDERED, ids, weights)
        torch.cuda.synchronize(DEVICE)
        assert (ordered.compact, ordered.ordered) == (True, True)
        got = ordered.route_weights.view(torch.int32)
        assert torch.equal(got, bits.gather(1, header_perm))
        assert torch.equal(got, bits.gather(1, torch_perm))
        # Every pattern survives (a permutation of the eight words per row).
        for token in range(rows):
            assert sorted(got[token].tolist()) == sorted(bits[token].tolist())
        assert torch.equal(weights.view(torch.int32), bits)  # the input is untouched
    # The NaN payload word survived (0x7FC12345 < 2**31, so int32 keeps it positive).
    assert 0x7FC12345 in bits[0].tolist()
    assert header_sort(list(TIE_IDS["duplicates"]))[1] != list(range(8))


@pytest.mark.parametrize("rows", ROWS_ORDERED)
def test_torch_sort_dispatch_on_the_device_is_the_documented_network(rows):
    """Explicit pin of the dispatch ``ids.sort(dim=-1)`` runs for ``[rows, 8]`` int32 on
    the device: keys and indices equal the network replay on the tie battery, ascending and
    descending rows, and 128 seeded rows over the seven-value alphabet."""
    cases = [list(values) for values in TIE_IDS.values()]
    cases += [list(range(8)), list(reversed(range(8)))]
    randomizer = random.Random(716)
    cases += [[randomizer.choice(SORT_ALPHABET) for _ in range(8)] for _ in range(128)]
    for values in cases:
        rotated = [values[token % 8 :] + values[: token % 8] for token in range(rows)]
        ids = torch.tensor(rotated, dtype=torch.int32, device=DEVICE)
        keys, permutation = ids.sort(dim=-1)
        expected = [header_sort(row) for row in rotated]
        assert keys.tolist() == [keys_ for keys_, _ in expected], values
        assert permutation.tolist() == [
            order for _, order in expected
        ], f"Torch sort dispatch differs from the documented network on {values}"
        assert keys.tolist() == [sorted(row) for row in rotated]


def test_ordered_kernel_takes_the_torch_sorted_path_for_16_and_24_routes_and_the_full_tables_above_32(
    selection_events,
):
    for rows in ROWS_TORCH_SORTED:
        num_routes = rows * TOP_K
        assert num_routes not in ORDERED_ROUTES and num_routes <= MAX_COMPACT_ROUTES
        ids = _case_ids(rows, IDS_DISTINCT, SEED + 20 + rows)
        weights = _case_weights(ids, SEED + 120 + rows)
        full = _routing(FULL, ids, weights)
        compact = _routing(COMPACT, ids, weights)
        ordered = _routing(ORDERED, ids, weights)
        torch.cuda.synchronize(DEVICE)
        assert (ordered.compact, ordered.ordered) == (True, False)
        assert (compact.compact, compact.ordered) == (True, False)
        assert ordered.counts.shape == (EXPERTS + num_routes + 1,)
        _assert_same_tables(ordered, compact, num_routes)
        assert torch.equal(ordered.counts[:EXPERTS], full.counts)
        assert torch.equal(
            _canonical(ordered.route_ids, ordered.counts),
            _canonical(full.route_ids, full.counts),
        )
        assert torch.equal(
            ordered.route_weights.view(torch.int32),
            full.route_weights.view(torch.int32),
        )
    for rows in (*ROWS_FULL_TABLES, 1024):
        num_routes = rows * TOP_K
        assert num_routes > MAX_COMPACT_ROUTES
        ids = _case_ids(rows, IDS_DISTINCT, SEED + 30 + rows)
        ids[0, 0] = -1  # an invalid id stays absent
        weights = _case_weights(ids, SEED + 130 + rows)
        full = _routing(FULL, ids, weights)
        compact = _routing(COMPACT, ids, weights)
        ordered = _routing(ORDERED, ids, weights)
        torch.cuda.synchronize(DEVICE)
        assert (ordered.compact, ordered.ordered) == (False, False)
        assert (compact.compact, compact.ordered) == (False, False)
        assert ordered.counts.shape == full.counts.shape == (EXPERTS,)
        _assert_same_tables(ordered, compact, num_routes)
        _assert_same_tables(ordered, full, num_routes)
        assert int(ordered.counts.sum()) == num_routes - 1
    events = _operator_events(selection_events)
    assert all(source == "override" for _, source in events)
    assert len(events) == 3 * (len(ROWS_TORCH_SORTED) + len(ROWS_FULL_TABLES) + 1)


def _stand_in_module(generator: torch.Generator) -> torch.nn.Module:
    module = torch.nn.Module()
    module.w13_weight = (
        (
            torch.randn((EXPERTS, 2 * INTERMEDIATE, HIDDEN), generator=generator)
            * WEIGHT_STD
        )
        .to(torch.bfloat16)
        .to(DEVICE)
    )
    module.w2_weight = (
        (torch.randn((EXPERTS, HIDDEN, INTERMEDIATE), generator=generator) * WEIGHT_STD)
        .to(torch.bfloat16)
        .to(DEVICE)
    )
    module.ep_size = 1
    module.activation = "silu"
    return module


def _plan() -> dict:
    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=INTERMEDIATE,
        hidden=HIDDEN,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        solution="triton",
        fast_math=True,
    )
    assert plan["apply_kernel_name"] == "triton_bf16_precomputed_moe_apply"
    return plan


def _apply_inputs(rows: int, seed: int, kind: str = IDS_DISTINCT):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, HIDDEN), generator=generator)
    if rows >= 2:
        x[0] *= 1e-3
        x[1] *= 1e3
    ids = _route_ids(rows, rows == SAME_EXPERTS_ROWS, generator)
    weights = _route_weights(ids, generator)
    if kind != IDS_DISTINCT:
        ids = _tie_ids(kind, rows)  # x and weights stay those of the distinct case
    return (
        x.to(torch.bfloat16).contiguous().to(DEVICE),
        ids.contiguous().to(DEVICE),
        weights.to(DEVICE),
    )


def _apply(name: str, plan: dict, x, module, weights, ids) -> torch.Tensor:
    with kernel_override(*OPERATOR, name):
        return tokenspeed_kernel.moe_apply(
            plan, x, module, None, topk_weights=weights, topk_ids=ids, do_finalize=True
        )


@pytest.mark.usefixtures("sorted_facade")
@pytest.mark.parametrize(("rows", "kind"), STAGE_CASES)
def test_stage_output_bytes_equal_compact_vs_ordered_for_sorted_bf16_silu(
    rows, kind, selection_events
):
    generator = torch.Generator().manual_seed(SEED)
    module = _stand_in_module(generator)
    plan = _plan()
    x, ids, weights = _apply_inputs(rows, SEED + rows, kind)
    compact_first = _apply(COMPACT, plan, x, module, weights, ids)
    compact_second = _apply(COMPACT, plan, x, module, weights, ids)
    ordered_first = _apply(ORDERED, plan, x, module, weights, ids)
    ordered_second = _apply(ORDERED, plan, x, module, weights, ids)
    torch.cuda.synchronize(DEVICE)
    for out in (compact_first, ordered_first):
        assert out.dtype is torch.bfloat16 and out.shape == (rows, HIDDEN)
        assert bool(torch.isfinite(out.float()).all())
    assert torch.equal(
        compact_first.view(torch.uint8), compact_second.view(torch.uint8)
    )
    assert torch.equal(
        ordered_first.view(torch.uint8), ordered_second.view(torch.uint8)
    )
    assert torch.equal(compact_first.view(torch.uint8), ordered_first.view(torch.uint8))
    if kind == "all_invalid":
        assert not bool((compact_first.float() != 0).any())  # no active expert
    else:
        assert bool((compact_first.float() != 0).any())
    events = _operator_events(selection_events)
    assert events == [(COMPACT, "override")] * 2 + [(ORDERED, "override")] * 2
    # The ordered kernel's flags for this row count.
    routing = _routing(ORDERED, ids, weights)
    num_routes = rows * TOP_K
    assert (routing.compact, routing.ordered) == (
        num_routes <= MAX_COMPACT_ROUTES,
        num_routes in ORDERED_ROUTES,
    )


@pytest.mark.usefixtures("sorted_facade")
@pytest.mark.parametrize("rows", ROWS_ORDERED)
def test_graph_replay_follows_live_routing_under_the_ordered_override(rows):
    """A captured ordered apply must read the live ids and weights on replay: the sort and
    the gather are in-kernel (no host read), so a different routing on the same buffers --
    a tie pattern included -- replays to the eager compact and full results on the new
    inputs."""
    generator = torch.Generator().manual_seed(SEED + 100)
    module = _stand_in_module(generator)
    plan = _plan()
    x, ids, weights = _apply_inputs(rows, SEED + 200 + rows)
    expected_first = _apply(COMPACT, plan, x, module, weights, ids)
    with kernel_override(*OPERATOR, ORDERED):
        warmup = tokenspeed_kernel.moe_apply(
            plan, x, module, None, topk_weights=weights, topk_ids=ids, do_finalize=True
        )
        torch.cuda.synchronize(DEVICE)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tokenspeed_kernel.moe_apply(
                plan,
                x,
                module,
                None,
                topk_weights=weights,
                topk_ids=ids,
                do_finalize=True,
            )
    graph.replay()
    torch.cuda.synchronize(DEVICE)
    assert torch.equal(warmup.view(torch.uint8), expected_first.view(torch.uint8))
    assert torch.equal(captured.view(torch.uint8), expected_first.view(torch.uint8))
    # New activations, the duplicates tie pattern and new weights on the same buffers.
    x_new, _, weights_new = _apply_inputs(rows, SEED + 300 + rows)
    ids_new = _tie_ids("duplicates", rows).to(DEVICE)
    assert not torch.equal(ids_new, ids)
    x.copy_(x_new)
    ids.copy_(ids_new)
    weights.copy_(weights_new)
    expected_compact = _apply(COMPACT, plan, x, module, weights, ids)
    expected_full = _apply(FULL, plan, x, module, weights, ids)
    assert torch.equal(
        expected_compact.view(torch.uint8), expected_full.view(torch.uint8)
    )
    assert not torch.equal(expected_compact, expected_first)
    graph.replay()
    torch.cuda.synchronize(DEVICE)
    assert torch.equal(captured.view(torch.uint8), expected_compact.view(torch.uint8))
    # And back to distinct ids drawn afresh.
    _, ids_third, weights_third = _apply_inputs(rows, SEED + 400 + rows)
    ids.copy_(ids_third)
    weights.copy_(weights_third)
    expected_third = _apply(COMPACT, plan, x, module, weights, ids)
    graph.replay()
    torch.cuda.synchronize(DEVICE)
    assert torch.equal(captured.view(torch.uint8), expected_third.view(torch.uint8))
    graph.reset()


def test_rejected_calls_raise_under_the_by_name_override_without_fallback(
    selection_events, monkeypatch
):
    ids = _case_ids(1, IDS_DISTINCT, SEED + 500)
    weights = _case_weights(ids, SEED + 501)

    def refused(match: str, *arguments, policy=SORTED_POLICY, num_experts=EXPERTS):
        with kernel_override(*OPERATOR, ORDERED):
            with pytest.raises(ValueError, match=match) as info:
                moe_expert_routing(
                    *arguments, num_experts, policy, solution=None, override=None
                )
        assert str(info.value).startswith(f"{ORDERED} cannot serve this call")
        return str(info.value)

    # R = 0: refused by the host guard (the compact text); served by the full kernel.
    refused("num_routes=0", ids[:, :0], weights[:, :0])
    empty = _routing(FULL, ids[:, :0], weights[:, :0])
    torch.cuda.synchronize(DEVICE)
    assert empty.route_ids.shape == (EXPERTS, 0) and empty.compact is False
    # The compact envelope: the named attribute, no fallback.
    refused("num_experts must be 256", ids, weights, num_experts=16)
    refused(
        "stage_policy must be one of",
        ids,
        weights,
        policy=StagePolicy(torch.bfloat16, "situ", True),
    )
    refused(
        "got fp16_silu_sorted",
        ids,
        weights,
        policy=StagePolicy(torch.float16, "silu", True),
    )
    # The sort clause: _moe's policy (no sort) is inside the compact envelope but has
    # nothing to fuse.
    message = refused("sort_routes must be True", ids, weights, policy=STAGE_POLICY)
    assert "CUDA" not in message and "got bf16_silu_unsorted" in message
    # The layout clauses (before the device clauses): int64 ids, strided ids, non-FP32 or
    # strided weights, another top-k.
    message = refused("topk_ids must be int32", ids.to(torch.int64), weights)
    assert "CUDA" not in message
    refused(
        "topk_ids must be contiguous",
        torch.zeros((1, 16), dtype=torch.int32, device=DEVICE)[:, ::2],
        weights,
    )
    refused("topk_weights must be float32", ids, weights.to(torch.bfloat16))
    refused(
        "topk_weights must be contiguous",
        ids,
        torch.zeros((1, 16), device=DEVICE)[:, ::2],
    )
    refused(
        "top_k must be 8",
        ids.reshape(2, 4).contiguous(),
        weights.reshape(2, 4).contiguous(),
    )
    # The compact and the full kernels serve the same int64 call, byte-equal.
    compact64 = _routing(COMPACT, ids.to(torch.int64), weights)
    full64 = _routing(FULL, ids.to(torch.int64), weights)
    torch.cuda.synchronize(DEVICE)
    assert compact64.compact is True and compact64.ordered is False
    assert torch.equal(
        _canonical(compact64.route_ids, compact64.counts),
        _canonical(full64.route_ids, full64.counts),
    )
    assert torch.equal(
        compact64.route_weights.view(torch.int32),
        full64.route_weights.view(torch.int32),
    )
    # The apply through the unpatched _moe (sort_routes=False) with the ordered override:
    # refused out of _moe, no fallback; the compact kernel serves the same apply.
    generator = torch.Generator().manual_seed(SEED)
    module = _stand_in_module(generator)
    plan = _plan()
    x, ids8, weights8 = _apply_inputs(2, SEED + 1)
    with pytest.raises(ValueError, match="sort_routes must be True"):
        _apply(ORDERED, plan, x, module, weights8, ids8)
    out = _apply(COMPACT, plan, x, module, weights8, ids8)
    torch.cuda.synchronize(DEVICE)
    assert out.shape == (2, HIDDEN)
    # With the facade bound to the sorted policy: int64 ids are refused out of _moe by
    # the alias clause; the compact kernel serves the same apply.
    monkeypatch.setattr(
        bf16,
        "moe_expert_routing",
        lambda topk_ids, topk_weights, num_experts, stage_policy, *, solution, override: (
            moe_expert_routing(
                topk_ids,
                topk_weights,
                num_experts,
                SORTED_POLICY,
                solution=solution,
                override=override,
            )
        ),
    )
    with pytest.raises(ValueError, match="topk_ids must be int32"):
        _apply(ORDERED, plan, x, module, weights8, ids8.to(torch.int64))
    out = _apply(COMPACT, plan, x, module, weights8, ids8.to(torch.int64))
    torch.cuda.synchronize(DEVICE)
    assert out.shape == (2, HIDDEN)
    events = _operator_events(selection_events)
    assert all(source == "override" for _, source in events)
    assert events.count((ORDERED, "override")) == 12
    assert events.count((FULL, "override")) == 2
    assert events.count((COMPACT, "override")) == 3


def test_ranked_selection_on_the_device_never_picks_the_ordered_kernel_and_explain_lists_three(
    selection_events,
):
    """Ranked selection lands on the full kernel for both signatures, both policies and
    every route count (one ``ranked`` event per distinct (signature, policy, R) key on
    the cleared cache, then a ``cache`` hit); ``explain_selection`` lists three
    registrations for int32 ids under the sorted policy, filters the ordered kernel for
    int64 ids and under _moe's unsorted policy, and marks it under the override."""
    route_counts = (TOP_K, 32, 40, 8192)
    for signature in (SIGNATURE_INT32, SIGNATURE_INT64):
        for policy in (SORTED_POLICY, STAGE_POLICY):
            for num_routes in route_counts:
                assert (
                    select_kernel(
                        *OPERATOR, signature, traits=_traits(num_routes, policy=policy)
                    ).name
                    == FULL
                )
    distinct_keys = 2 * 2 * len(route_counts)
    assert _operator_events(selection_events) == [(FULL, "ranked")] * distinct_keys
    ids = _case_ids(1, IDS_DISTINCT, SEED + 600)
    weights = _case_weights(ids, SEED + 601)
    routing = moe_expert_routing(
        ids, weights, EXPERTS, SORTED_POLICY, solution=None, override=None
    )
    torch.cuda.synchronize(DEVICE)
    assert routing.compact is False and routing.ordered is False
    assert routing.counts.shape == (EXPERTS,) and int(routing.counts.sum()) == TOP_K
    assert _operator_events(selection_events)[distinct_keys:] == [(FULL, "cache")]
    both = explain_selection(*OPERATOR, SIGNATURE_INT32, traits=_traits(TOP_K))
    assert "Candidates (3 matched, 3 registered)" in both
    assert f"1. {FULL}  [SELECTED]" in both and f"2. {COMPACT}" in both
    assert f"3. {ORDERED}" in both
    int64 = explain_selection(*OPERATOR, SIGNATURE_INT64, traits=_traits(TOP_K))
    assert "Candidates (2 matched, 3 registered)" in int64
    assert int64.index("Filtered out:") < int64.index(f"- {ORDERED}")
    unsorted = explain_selection(
        *OPERATOR, SIGNATURE_INT32, traits=_traits(TOP_K, policy=STAGE_POLICY)
    )
    assert "Candidates (2 matched, 3 registered)" in unsorted
    assert unsorted.index("Filtered out:") < unsorted.index(f"- {ORDERED}")
    for num_routes in route_counts:
        forced = explain_selection(
            *OPERATOR, SIGNATURE_INT32, traits=_traits(num_routes), override=ORDERED
        )
        assert f"{ORDERED}  [SELECTED (override)]" in forced
        assert f"Override: {ORDERED} (explicit override= argument)" in forced
    assert ORDERED_TOP_K == TOP_K and ORDERED_ROUTES == frozenset({8, 32})
