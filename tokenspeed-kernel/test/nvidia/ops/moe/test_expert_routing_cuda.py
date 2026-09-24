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

"""``moe.expert_routing`` on an NVIDIA device: the compact tables equal the full tables
where defined, the route rule above 32 routes, stage-output byte equality of the shared
BF16 MoE apply under the two kernels, CUDA-graph replay, rejected calls and ranking.

The in-tree kernels compile to their own Triton hashes (not a previously measured
binary's), so their correctness is established here anew. For every route count of
``ROUTES`` the facade is run once per arm inside ``kernel_override`` with a selection
listener recording the witnesses; the canonicalized tables (unwritten slots set to
``-1``), the counts and the ``route_weights`` must agree byte for byte, and the compact
tail must list the active experts ascending with their number. Above
``MAX_COMPACT_ROUTES`` the compact kernel must emit the full tables and
``compact=False``. The apply is run at a small geometry (256 experts, hidden 256,
intermediate 64, top-8) for BF16 SiLU: the BF16 outputs of the two arms must be
byte-identical, also through a captured CUDA graph whose inputs change between replays
(the compact trip count is a device-side load). ``_moe`` passes ``sort_routes=False``;
the sorted arm of the admitted envelope is run by binding the facade name in ``bf16``
to the sorted policy, so ``moe_apply`` runs the unchanged stage kernels on the
sorted-prep tables.
"""

from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.expert_routing import (
    EXPERT_ROUTING_FAMILY,
    EXPERT_ROUTING_MODE,
    MAX_COMPACT_ROUTES,
    TRITON_MOE_ROUTING_COMPACT,
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
SIGNATURE = format_signature(topk_ids=dense_tensor_format(torch.int32))
DEVICE = torch.device("cuda", 0)
SEED = 20260921
EXPERTS, TOP_K = 256, 8
# Compact-admitted route counts (one token, R slots), including the boundaries of the
# packing kernel's power-of-two block: 1, 7/8, 16/17, 24, 31/32.
ROUTES = (1, 7, 8, 16, 17, 24, 31, 32)
# Above the rule: the compact kernel emits the full tables (prefill shapes included).
ROUTES_FULL_TABLES = (33, 40, 64, 256, 8192)
# Stage byte-equality geometry: the compact guard needs 256 experts; hidden 256 and
# intermediate 64 keep the BF16 weights at about 42 MB.
HIDDEN, INTERMEDIATE = 256, 64
ROWS = (1, 2, 3, 4, 8, 32)
SAME_EXPERTS_ROWS = 32
WEIGHT_STD = 0.02
# The policy _moe passes (top-k order kept) and the sorted policy of the envelope.
STAGE_POLICY = StagePolicy(torch.bfloat16, "silu", False)
SORTED_POLICY = StagePolicy(torch.bfloat16, "silu", True)
POLICIES = (STAGE_POLICY, SORTED_POLICY)


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
def bind_facade_policy(monkeypatch):
    """Bind the policy ``_moe`` hands the facade. ``_moe`` passes ``sort_routes=False``
    (``STAGE_POLICY``): that arm runs unpatched. A caller that sorts is emulated by
    rebinding the facade name in ``bf16`` to the sorted policy, so ``moe_apply`` runs
    the unchanged stage kernels and combine on the sorted-prep tables."""

    def bind(policy: StagePolicy) -> None:
        if policy == STAGE_POLICY:
            return

        def facade(
            topk_ids, topk_weights, num_experts, stage_policy, *, solution, override
        ):
            assert stage_policy == STAGE_POLICY  # what _moe passes on this tree
            return moe_expert_routing(
                topk_ids,
                topk_weights,
                num_experts,
                policy,
                solution=solution,
                override=override,
            )

        monkeypatch.setattr(bf16, "moe_expert_routing", facade)

    return bind


def _operator_events(events: list[SelectionEvent]) -> list[tuple[str, str]]:
    return [
        (event.kernel_name, event.source)
        for event in events
        if (event.family, event.mode) == OPERATOR
    ]


def _canonical(route_ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """The table with slots at or beyond the per-expert count set to -1."""
    slots = torch.arange(route_ids.shape[1], device=route_ids.device)
    written = slots[None, :] < counts[: route_ids.shape[0], None]
    return route_ids.masked_fill(~written, -1)


def _traits(
    num_routes: int, num_experts: int = EXPERTS, policy: StagePolicy = STAGE_POLICY
) -> dict[str, int | str]:
    return {
        "num_experts": num_experts,
        "num_routes": num_routes,
        "stage_policy": policy.name,
    }


def _route_ids(
    rows: int, same_experts: bool, generator: torch.Generator, experts: int = EXPERTS
) -> torch.Tensor:
    """int32 ``[rows, 8]`` ids, distinct per token and never ascending; token 0 covers
    experts 0 and ``experts - 1``; with ``same_experts`` every token uses token 0's
    experts."""
    ids = torch.empty((rows, TOP_K), dtype=torch.int64)
    middle = torch.randperm(experts - 2, generator=generator)[: TOP_K - 2] + 1
    first = torch.cat([torch.tensor([experts - 1, 0]), middle])
    ids[0] = first[torch.randperm(TOP_K, generator=generator)]
    for token in range(1, rows):
        if same_experts:
            ids[token] = ids[0][torch.randperm(TOP_K, generator=generator)]
        else:
            ids[token] = torch.randperm(experts, generator=generator)[:TOP_K]
    for token in range(rows):
        if bool(torch.all(ids[token, 1:] > ids[token, :-1])):
            ids[token, 0], ids[token, 1] = int(ids[token, 1]), int(ids[token, 0])
    return ids.to(torch.int32)


def _route_weights(ids: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """FP32 sigmoid scores of ``1.5 + randn`` renormalized."""
    scores = torch.sigmoid(torch.randn(ids.shape, generator=generator) + 1.5)
    return (scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)).contiguous()


def _routing_case(num_routes: int, kind: str, seed: int):
    """One token with ``num_routes`` slots: distinct ids, or duplicates plus the invalid
    ids ``-1`` and ``300`` (at one route only the invalid ``-1``), or every id invalid
    (no active expert). The tests derive their expectations from the returned ids."""
    generator = torch.Generator().manual_seed(seed)
    if kind == "distinct":
        ids = torch.randperm(EXPERTS, generator=generator)[:num_routes]
    elif kind == "duplicates_and_invalid":
        ids = torch.randint(0, 4, (num_routes,), generator=generator)
        ids[0] = -1
        if num_routes > 1:
            ids[1] = 300
    else:
        ids = torch.full((num_routes,), -1)
    ids = ids.reshape(1, num_routes).to(torch.int32).to(DEVICE)
    weights = (torch.rand((1, num_routes), generator=generator) + 0.1).to(DEVICE)
    return ids, weights


def _routing(
    name: str,
    ids: torch.Tensor,
    weights: torch.Tensor,
    num_experts=EXPERTS,
    policy: StagePolicy = STAGE_POLICY,
):
    with kernel_override(*OPERATOR, name):
        return moe_expert_routing(
            ids, weights, num_experts, policy, solution=None, override=None
        )


@pytest.mark.parametrize("policy", POLICIES, ids=lambda policy: policy.name)
@pytest.mark.parametrize("kind", ["distinct", "duplicates_and_invalid", "all_invalid"])
@pytest.mark.parametrize("num_routes", ROUTES)
def test_compact_tables_equal_full_tables_where_defined(
    num_routes, kind, policy, selection_events
):
    ids, weights = _routing_case(num_routes, kind, SEED + num_routes)
    full = _routing(FULL, ids, weights, policy=policy)
    compact = _routing(COMPACT, ids, weights, policy=policy)
    torch.cuda.synchronize(DEVICE)
    assert full.compact is False and compact.compact is True
    assert full.ordered is False and compact.ordered is False
    assert full.counts.shape == (EXPERTS,)
    assert compact.counts.shape == (EXPERTS + min(EXPERTS, num_routes) + 1,)
    assert full.route_ids.shape == compact.route_ids.shape == (EXPERTS, num_routes)
    assert torch.equal(compact.counts[:EXPERTS], full.counts)
    assert torch.equal(
        _canonical(compact.route_ids, compact.counts),
        _canonical(full.route_ids, full.counts),
    )
    active = torch.nonzero(full.counts > 0).squeeze(1).to(torch.int32)
    active_count = int(compact.counts[EXPERTS + min(EXPERTS, num_routes)])
    assert active_count == active.numel()
    assert torch.equal(compact.counts[EXPERTS : EXPERTS + active_count], active)
    # The expectation is derived from the constructed ids, not from the kind label
    # (``duplicates_and_invalid`` at one route is the single invalid id).
    valid = ids[(ids >= 0) & (ids < EXPERTS)]
    expected_active = int(valid.unique().numel())
    assert active_count == expected_active
    assert int(full.counts.sum()) == valid.numel()
    if kind == "all_invalid":
        assert expected_active == 0
    elif kind == "distinct" or num_routes > 1:
        assert expected_active > 0  # the case keeps at least one valid id
    if expected_active > 1:
        assert torch.all(active[1:] > active[:-1])  # ascending
    assert torch.equal(
        compact.route_weights.view(torch.uint8), full.route_weights.view(torch.uint8)
    )
    if policy.sort_routes:
        table_ids, order = ids.sort(dim=-1)
        assert torch.equal(full.route_weights, weights.gather(1, order))
        assert full.route_weights.data_ptr() != weights.data_ptr()
    else:
        # _moe's policy: the weights are the input tensor itself.
        table_ids = ids
        assert full.route_weights is weights and compact.route_weights is weights
    # Every written slot is a position of that expert in the flattened table ids.
    flat = table_ids.reshape(-1)
    table = _canonical(full.route_ids, full.counts)
    for expert in active.tolist():
        positions = table[expert, : int(full.counts[expert])].to(torch.int64)
        assert torch.equal(flat[positions], torch.full_like(positions, expert))
    assert _operator_events(selection_events) == [
        (FULL, "override"),
        (COMPACT, "override"),
    ]


@pytest.mark.parametrize("num_routes", ROUTES_FULL_TABLES)
def test_compact_kernel_emits_full_tables_above_32_routes(num_routes, selection_events):
    assert num_routes > MAX_COMPACT_ROUTES
    generator = torch.Generator().manual_seed(SEED + num_routes)
    rows = num_routes // TOP_K if num_routes % TOP_K == 0 else 1
    top_k = num_routes // rows
    ids = torch.randint(0, EXPERTS, (rows, top_k), generator=generator)
    ids[0, 0] = -1  # an invalid id stays absent
    ids = ids.to(torch.int32).to(DEVICE)
    weights = (torch.rand((rows, top_k), generator=generator) + 0.1).to(DEVICE)
    full = _routing(FULL, ids, weights)
    compact = _routing(COMPACT, ids, weights)
    torch.cuda.synchronize(DEVICE)
    assert compact.compact is False and compact.ordered is False
    assert compact.counts.shape == (EXPERTS,) and full.counts.shape == (EXPERTS,)
    assert torch.equal(compact.counts, full.counts)
    assert torch.equal(
        _canonical(compact.route_ids, compact.counts),
        _canonical(full.route_ids, full.counts),
    )
    assert torch.equal(
        compact.route_weights.view(torch.uint8), full.route_weights.view(torch.uint8)
    )
    assert int(full.counts.sum()) == num_routes - 1
    assert _operator_events(selection_events) == [
        (FULL, "override"),
        (COMPACT, "override"),
    ]


def _stand_in_module(
    generator: torch.Generator, experts: int = EXPERTS
) -> torch.nn.Module:
    module = torch.nn.Module()
    module.w13_weight = (
        (
            torch.randn((experts, 2 * INTERMEDIATE, HIDDEN), generator=generator)
            * WEIGHT_STD
        )
        .to(torch.bfloat16)
        .to(DEVICE)
    )
    module.w2_weight = (
        (torch.randn((experts, HIDDEN, INTERMEDIATE), generator=generator) * WEIGHT_STD)
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


def _apply_inputs(rows: int, seed: int, experts: int = EXPERTS):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, HIDDEN), generator=generator)
    if rows >= 2:
        x[0] *= 1e-3
        x[1] *= 1e3
    ids = _route_ids(rows, rows == SAME_EXPERTS_ROWS, generator, experts)
    weights = _route_weights(ids, generator)
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


@pytest.mark.parametrize("policy", POLICIES, ids=lambda policy: policy.name)
@pytest.mark.parametrize("rows", ROWS)
def test_stage_output_bytes_equal_full_vs_compact_for_bf16_silu(
    rows, policy, selection_events, bind_facade_policy
):
    """The two arms of ``COMPACT_STAGE_POLICIES``: ``_moe``'s own policy runs
    unpatched; the sorted policy is bound into the facade name ``_moe`` resolves."""
    bind_facade_policy(policy)
    generator = torch.Generator().manual_seed(SEED)
    module = _stand_in_module(generator)
    plan = _plan()
    x, ids, weights = _apply_inputs(rows, SEED + rows)
    full_first = _apply(FULL, plan, x, module, weights, ids)
    full_second = _apply(FULL, plan, x, module, weights, ids)
    compact_first = _apply(COMPACT, plan, x, module, weights, ids)
    compact_second = _apply(COMPACT, plan, x, module, weights, ids)
    torch.cuda.synchronize(DEVICE)
    for out in (full_first, compact_first):
        assert out.dtype is torch.bfloat16 and out.shape == (rows, HIDDEN)
        assert bool(torch.isfinite(out.float()).all())
    assert torch.equal(full_first.view(torch.uint8), full_second.view(torch.uint8))
    assert torch.equal(
        compact_first.view(torch.uint8), compact_second.view(torch.uint8)
    )
    assert torch.equal(full_first.view(torch.uint8), compact_first.view(torch.uint8))
    assert bool((full_first.float() != 0).any())
    events = _operator_events(selection_events)
    assert events == [(FULL, "override")] * 2 + [(COMPACT, "override")] * 2
    # The compact kernel packs at rows 1-4 (R <= 32) and emits the full tables above.
    routing = _routing(COMPACT, ids, weights, policy=policy)
    assert routing.compact is (rows * TOP_K <= MAX_COMPACT_ROUTES)


@pytest.mark.parametrize("rows", (1, 4))
def test_graph_replay_follows_live_routing_under_the_compact_override(rows):
    """A captured compact apply must read the live tables on replay: the trip count of
    the stage loops is a device-side load, so a different active set on the same
    buffers replays to the eager full result on the new inputs."""
    generator = torch.Generator().manual_seed(SEED + 100)
    module = _stand_in_module(generator)
    plan = _plan()
    x, ids, weights = _apply_inputs(rows, SEED + 200 + rows)
    expected_first = _apply(FULL, plan, x, module, weights, ids)
    with kernel_override(*OPERATOR, COMPACT):
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
    # A different active expert set and different activations on the same buffers.
    x_new, ids_new, weights_new = _apply_inputs(rows, SEED + 300 + rows)
    assert not torch.equal(ids_new, ids)
    x.copy_(x_new)
    ids.copy_(ids_new)
    weights.copy_(weights_new)
    expected_second = _apply(FULL, plan, x, module, weights, ids)
    assert not torch.equal(expected_second, expected_first)
    graph.replay()
    torch.cuda.synchronize(DEVICE)
    assert torch.equal(captured.view(torch.uint8), expected_second.view(torch.uint8))
    graph.reset()


def test_rejected_calls_raise_under_the_by_name_override_without_fallback(
    selection_events,
):
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randperm(EXPERTS, generator=generator)[:TOP_K].reshape(1, TOP_K)
    ids = ids.to(torch.int32).to(DEVICE)
    weights = (torch.rand((1, TOP_K), generator=generator) + 0.1).to(DEVICE)
    # R = 0: refused by the compact host guard; served by the full kernel.
    with kernel_override(*OPERATOR, COMPACT):
        with pytest.raises(ValueError, match="num_routes=0") as info:
            moe_expert_routing(
                ids[:, :0],
                weights[:, :0],
                EXPERTS,
                STAGE_POLICY,
                solution=None,
                override=None,
            )
    assert str(info.value).startswith(f"{COMPACT} cannot serve this call")
    empty = _routing(FULL, ids[:, :0], weights[:, :0])
    torch.cuda.synchronize(DEVICE)
    assert empty.route_ids.shape == (EXPERTS, 0) and empty.compact is False
    assert torch.equal(
        empty.counts, torch.zeros(EXPERTS, dtype=torch.int32, device=DEVICE)
    )
    # Outside the envelope: the named attribute, no fallback.
    with kernel_override(*OPERATOR, COMPACT):
        with pytest.raises(ValueError, match="num_experts must be 256"):
            moe_expert_routing(
                ids, weights, 16, STAGE_POLICY, solution=None, override=None
            )
        with pytest.raises(ValueError, match="stage_policy must be one of"):
            moe_expert_routing(
                ids,
                weights,
                EXPERTS,
                StagePolicy(torch.bfloat16, "situ", False),
                solution=None,
                override=None,
            )
        with pytest.raises(ValueError, match="got fp16_silu_sorted"):
            moe_expert_routing(
                ids,
                weights,
                EXPERTS,
                StagePolicy(torch.float16, "silu", True),
                solution=None,
                override=None,
            )
    # The apply on a 16-expert module with the compact override forced: the guard
    # raises out of ``_moe`` (no fallback); the full kernel serves the same apply.
    module16 = _stand_in_module(generator, experts=16)
    x, ids16, weights16 = _apply_inputs(2, SEED + 1, experts=16)
    with pytest.raises(ValueError, match="num_experts must be 256"):
        _apply(COMPACT, _plan(), x, module16, weights16, ids16)
    out = _apply(FULL, _plan(), x, module16, weights16, ids16)
    torch.cuda.synchronize(DEVICE)
    assert out.shape == (2, HIDDEN)
    events = _operator_events(selection_events)
    assert all(source == "override" for _, source in events)
    assert (
        events.count((COMPACT, "override")) == 5
        and events.count((FULL, "override")) == 2
    )


def test_ranked_selection_on_the_device_picks_full(selection_events):
    """Ranked selection lands on the full kernel for both signatures, both policies and
    every route count. ``num_routes`` and the policy are part of the selection cache
    key, so on the cleared cache every distinct (signature, policy, R) tuple is one
    ``ranked`` event; the facade call that follows repeats the int32 / unsorted /
    ``TOP_K`` key and is served from the cache."""
    signatures = (
        SIGNATURE,
        format_signature(topk_ids=dense_tensor_format(torch.int64)),
    )
    route_counts = (TOP_K, 32, 40, 8192)
    for signature in signatures:
        for policy in POLICIES:
            for num_routes in route_counts:
                selected = select_kernel(
                    *OPERATOR, signature, traits=_traits(num_routes, policy=policy)
                )
                assert selected.name == FULL
    distinct_keys = len(signatures) * len(POLICIES) * len(route_counts)
    assert _operator_events(selection_events) == [(FULL, "ranked")] * distinct_keys
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randperm(EXPERTS, generator=generator)[:TOP_K].reshape(1, TOP_K)
    ids = ids.to(torch.int32).to(DEVICE)
    weights = (torch.rand((1, TOP_K), generator=generator) + 0.1).to(DEVICE)
    routing = moe_expert_routing(
        ids, weights, EXPERTS, STAGE_POLICY, solution=None, override=None
    )
    torch.cuda.synchronize(DEVICE)
    assert routing.compact is False and routing.counts.shape == (EXPERTS,)
    assert int(routing.counts.sum()) == TOP_K
    assert _operator_events(selection_events)[distinct_keys:] == [(FULL, "cache")]


def test_explain_selection_reports_the_override_and_the_trait_mismatch():
    # _moe's policy: the ordered kernel is filtered by its stage_policy trait.
    both = explain_selection(*OPERATOR, SIGNATURE, traits=_traits(8))
    assert "Candidates (2 matched, 3 registered)" in both
    assert f"1. {FULL}  [SELECTED]" in both and f"2. {COMPACT}" in both
    # The sorted policy: all three registrations are candidates.
    three = explain_selection(
        *OPERATOR, SIGNATURE, traits=_traits(8, policy=SORTED_POLICY)
    )
    assert "Candidates (3 matched, 3 registered)" in three
    forced = explain_selection(
        *OPERATOR, SIGNATURE, traits=_traits(8), override=COMPACT
    )
    assert f"{COMPACT}  [SELECTED (override)]" in forced
    assert f"Override: {COMPACT} (explicit override= argument)" in forced
    mismatch = explain_selection(
        *OPERATOR, SIGNATURE, traits=_traits(8, num_experts=16), override=COMPACT
    )
    assert "Candidates (1 matched, 3 registered)" in mismatch
    assert (
        f"Override selects {COMPACT}, which is not among the matched candidates"
        in mismatch
    )
