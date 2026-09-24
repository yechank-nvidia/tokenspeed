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

"""``moe.route_epilogue`` through ``select_kernel``: registration, admission, the by-name switch, byte equality with the route statements and the CPU pin of the reduction tree.

CPU only. Importing ``tokenspeed_kernel.ops.moe.route_epilogue`` registers both
kernels (the package init imports the Triton module); ``load_builtin_kernels``
re-populates the singleton when an earlier test module reset it. The torch
kernel runs on CPU, so ranked selection and the facade are exercised for real.
The Triton kernel is never launched here (``test/nvidia/ops/moe/test_route_epilogue_cuda.py``
covers it): on CPU tensors its host guard rejects the call, which is what the
override tests assert -- the by-name switch has no fallback. Its admission has
no row clause (any positive row count): only ``rows == 0`` is refused, by the
host guard alone, so that an empty grid is never launched.

The CPU pin at the end binds the reduction tree literally in the kernel source
and checks, with a ``struct``-based float32 emulation, that the literal tree
equals the descending shuffle-down loop while the sequential order does not.
This proves what the kernel encodes; whether this torch build's CUDA sum uses
the same order is the GPU pin in ``test/nvidia/ops/moe/test_route_epilogue_cuda.py``
-- a CPU ``torch.sum`` has another order and is deliberately not compared here.
"""

from __future__ import annotations

import ast
import inspect
import struct
from pathlib import Path

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.route_epilogue import (
    ROUTE_EPILOGUE_FAMILY,
    ROUTE_EPILOGUE_MODE,
    ROUTE_EPILOGUE_OVERRIDE_ENV,
    ROUTE_EPILOGUE_SIGNATURES,
    TORCH_ROUTE_EPILOGUE,
    TRITON_ROUTE_EPILOGUE,
    moe_route_epilogue,
    route_epilogue_traits,
)
from tokenspeed_kernel.ops.moe.triton import route_epilogue as triton_module
from tokenspeed_kernel.ops.moe.triton.route_epilogue import (
    k8_reduction_tree,
    reduction_order_fixture,
    reduction_order_mismatch,
    route_epilogue_rejection,
)
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

OPERATOR = (ROUTE_EPILOGUE_FAMILY, ROUTE_EPILOGUE_MODE)
SIGNATURE = next(iter(ROUTE_EPILOGUE_SIGNATURES))
# Grouped-routing example shape (256 experts in 8 groups, 4 groups kept, top-8).
EXPERTS, NUM_GROUPS, TOPK_GROUPS, TOPK = 256, 8, 4, 8
GROUP_SIZE = EXPERTS // NUM_GROUPS
# Any positive row count is admitted (decode batches, padded batches, prefill);
# only the empty grid is refused, and only by the host guard (no spec trait).
ROWS_ADMITTED = (1, 2, 3, 4, 5, 7, 8, 16, 32, 33, 64, 128, 256, 1024, 4096)
ROWS_REJECTED = (0,)
RENORMALIZE_VALUES = (True, False)
# Synthetic router-logit distribution (std 0.6 logits, std 0.1 bias).
LOGIT_STD, BIAS_STD = 0.6, 0.1
CROSS_OPERATOR_KERNEL = "torch_decode_gemv"  # registered under gemm.decode_gemv
NAN_PAYLOAD_BITS = 0x7FC01234
# The pattern rows, scattered into the selected slots.
TIE_ROW, NEGATIVE_ROW, DENORMAL_ROW, ROUNDING_ROW, UNDERFLOW_ROW, NEG_INF_ROW = range(
    1, 7
)
PATTERNS = {
    TIE_ROW: (0.5,) * 8,
    NEGATIVE_ROW: (0.9, -0.1, 0.8, -0.2, 0.7, -0.05, 0.6, -0.15),
    DENORMAL_ROW: (
        2.0**-149,
        2.0**-140,
        2.0**-130,
        2.0**-126,
        0.0,
        2.0**-145,
        2.0**-135,
        2.0**-128,
    ),
    ROUNDING_ROW: (
        1.0,
        2.0**-24,
        2.0**-25,
        2.0**-26,
        0.5,
        2.0**-23,
        2.0**-24,
        2.0**-25,
    ),
}
NEG_INF_SLOT = 3
# Kernel-test-only row (the pattern set has no such row): a ``-0.0`` selected score
# at slot 6 and a NaN-payload word at slot 7, showing that the gather and the int32
# conversion copy raw words. Kept out of row 0 on purpose: row 0 is the only
# realistic row when rows <= 4 (the decode shape rows=1 included), and a NaN word
# there would make the whole row canonical NaN under ``renormalize=True`` in every
# summation order, so the byte-equality of those cells would say nothing about the
# reduction tree.
SPECIAL_ROW = 7
NEGATIVE_ZERO_SLOT, NAN_PAYLOAD_SLOT = 6, 7
TRITON_SOURCE = Path(triton_module.__file__)
# The row on which the sequential order differs from the tree.
SEQUENTIAL_DIFFERS_ROW = (1.0,) + (2.0**-24,) * 7


@pytest.fixture(autouse=True)
def route_epilogue_registry():
    """Both kernels registered; no cached selection or override witness leaks between tests."""
    if KernelRegistry.get().get_by_name(TORCH_ROUTE_EPILOGUE) is None:
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


def _spec(name: str):
    spec = KernelRegistry.get().get_by_name(name)
    assert spec is not None, name
    return spec


def _traits(
    rows: int,
    experts: int = EXPERTS,
    topk: int = TOPK,
    renormalize: bool = True,
    contiguous: bool = True,
) -> dict[str, int | bool]:
    return {
        "rows": rows,
        "experts": experts,
        "topk": topk,
        "renormalize": renormalize,
        "contiguous": contiguous,
    }


def _layout(
    rows: int,
    experts: int = EXPERTS,
    topk: int = TOPK,
    renormalize: bool = True,
    contiguous: bool = True,
    strided: str = "scores",
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """CPU tensors carrying the metadata of one trait dict (values are irrelevant);
    ``strided`` names the tensor made non-contiguous when ``contiguous`` is False."""
    scores = torch.zeros(rows, experts)
    ids = torch.zeros(rows, topk, dtype=torch.int64)
    if not contiguous and strided == "scores":
        scores = torch.zeros(rows, 2 * experts)[:, ::2]
    if not contiguous and strided == "ids":
        ids = torch.zeros(rows, 2 * topk, dtype=torch.int64)[:, ::2]
    return scores, ids, renormalize


def _route_inputs(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The grouped router up to its epilogue on seeded inputs (the three mask
    statements inlined), then the pattern rows scattered into the selected slots
    and, in row 7 (kernel-test-only), a ``-0.0`` and a NaN-payload selected score.
    Row 0 is always a realistic sigmoid row."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, EXPERTS, generator=generator) * LOGIT_STD
    bias = torch.randn(EXPERTS, generator=generator) * BIAS_STD
    if rows > UNDERFLOW_ROW:  # every sigmoid ~2e-22: the +1e-20 path of the route
        logits[UNDERFLOW_ROW] = -50.0
    scores = logits.float().sigmoid()
    choices = scores + bias
    grouped = choices.reshape(rows, NUM_GROUPS, GROUP_SIZE)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    group_ids = group_scores.topk(TOPK_GROUPS, dim=-1, sorted=True).indices
    keep_groups = torch.zeros_like(group_scores, dtype=torch.bool)
    keep_groups.scatter_(1, group_ids, True)
    keep = keep_groups.unsqueeze(-1).expand_as(grouped).reshape_as(choices)
    ids = (
        choices.masked_fill(~keep, float("-inf"))
        .topk(TOPK, dim=-1, sorted=True)
        .indices
    )
    for row, pattern in PATTERNS.items():
        if rows > row:
            scores[row, ids[row]] = torch.tensor(pattern, dtype=torch.float32)
    if rows > NEG_INF_ROW:
        scores[NEG_INF_ROW, ids[NEG_INF_ROW, NEG_INF_SLOT]] = float("-inf")
    if rows > SPECIAL_ROW:
        scores[SPECIAL_ROW, ids[SPECIAL_ROW, NEGATIVE_ZERO_SLOT]] = -0.0
        words = scores.view(torch.int32)
        words[SPECIAL_ROW, ids[SPECIAL_ROW, NAN_PAYLOAD_SLOT]] = NAN_PAYLOAD_BITS
    return scores.contiguous(), ids.contiguous()


def _reference(
    scores: torch.Tensor, ids: torch.Tensor, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """The three statements the torch routing tail inlines."""
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights, ids.to(torch.int32)


def _words(value: torch.Tensor) -> torch.Tensor:
    return value.view(torch.int32) if value.dtype is torch.float32 else value


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _add(left: float, right: float) -> float:
    return _f32(left + right)


def _shuffle_down_tree(values) -> float:
    """Torch CUDA ``block_x_reduce`` for eight lanes, literally: ``for offset in (4, 2,
    1): lanes[lane] = lanes[lane] + lanes[lane + offset]``; lane 0 is the sum."""
    lanes = [_f32(value) for value in values]
    for offset in (4, 2, 1):
        previous = lanes[:]
        for lane in range(8 - offset):
            lanes[lane] = _add(previous[lane], previous[lane + offset])
    return lanes[0]


def _explicit_tree(values) -> float:
    """The kernel's ``even``/``odd``/``even + odd`` expression."""
    even = _add(_add(values[0], values[4]), _add(values[2], values[6]))
    odd = _add(_add(values[1], values[5]), _add(values[3], values[7]))
    return _add(even, odd)


def _sequential(values) -> float:
    total = 0.0
    for value in values:
        total = _add(total, value)
    return total


def _bits(value: float) -> bytes:
    return struct.pack("f", value)


def test_registration_lists_torch_then_triton_with_one_shared_signature():
    registry = KernelRegistry.get()
    specs = registry.list_kernels(*OPERATOR)
    assert [spec.name for spec in specs] == [
        TORCH_ROUTE_EPILOGUE,
        TRITON_ROUTE_EPILOGUE,
    ]
    torch_spec, triton_spec = specs
    assert (torch_spec.solution, torch_spec.priority) == ("torch", Priority.PORTABLE)
    assert (triton_spec.solution, triton_spec.priority) == (
        "triton",
        Priority.REFERENCE,
    )
    assert (
        torch_spec.format_signatures
        == triton_spec.format_signatures
        == ROUTE_EPILOGUE_SIGNATURES
        == frozenset({SIGNATURE})
    )
    assert torch_spec.capability == CapabilityRequirement()
    assert triton_spec.capability == CapabilityRequirement(
        vendors=frozenset({"nvidia"})
    )
    assert torch_spec.traits == {}
    # No ``rows`` trait: the row count is the grid size only. Both renormalize
    # values are compiled variants of the one kernel.
    assert triton_spec.traits == {
        "experts": frozenset({EXPERTS}),
        "topk": frozenset({TOPK}),
        "renormalize": frozenset({True, False}),
        "contiguous": frozenset({True}),
    }
    torch_text = describe_kernel(TORCH_ROUTE_EPILOGUE)
    assert "Operator: moe.route_epilogue" in torch_text
    assert "Solution: torch" in torch_text and "Priority: 4 (PORTABLE)" in torch_text
    triton_text = describe_kernel(TRITON_ROUTE_EPILOGUE)
    assert "Operator: moe.route_epilogue" in triton_text
    assert (
        "Solution: triton" in triton_text and "Priority: 0 (REFERENCE)" in triton_text
    )
    assert registry.get_impl(TORCH_ROUTE_EPILOGUE).__name__ == TORCH_ROUTE_EPILOGUE
    assert registry.get_impl(TRITON_ROUTE_EPILOGUE).__name__ == TRITON_ROUTE_EPILOGUE
    assert (
        ROUTE_EPILOGUE_OVERRIDE_ENV == "TOKENSPEED_KERNEL_OVERRIDE_MOE_ROUTE_EPILOGUE"
    )
    assert tokenspeed_kernel.moe_route_epilogue is moe_route_epilogue


@pytest.mark.parametrize("renormalize", RENORMALIZE_VALUES)
@pytest.mark.parametrize("rows", ROWS_ADMITTED)
def test_triton_spec_admits_every_positive_row_count_and_the_host_guard_agrees_up_to_the_device(
    rows, renormalize
):
    traits = _traits(rows, renormalize=renormalize)
    assert spec_matches_traits(_spec(TRITON_ROUTE_EPILOGUE), traits)
    assert spec_matches_traits(_spec(TORCH_ROUTE_EPILOGUE), traits)
    tensors = _layout(rows, renormalize=renormalize)
    assert route_epilogue_traits(*tensors) == traits
    # Lock-step: on CPU tensors of an admitted layout only the device clause fails.
    reason = route_epilogue_rejection(*tensors)
    assert "CUDA" in reason and "cpu" in reason


@pytest.mark.parametrize(
    ("traits", "strided", "attribute"),
    [
        (_traits(1, experts=128), "scores", "scores"),
        (_traits(1, topk=4), "scores", "ids"),
        (_traits(1, contiguous=False), "scores", "scores must be contiguous"),
        (_traits(1, contiguous=False), "ids", "ids must be contiguous"),
        (_traits(4096, experts=512), "scores", "scores"),
    ],
)
def test_triton_spec_rejects_off_shape_traits_and_the_host_guard_names_the_attribute(
    traits, strided, attribute
):
    assert not spec_matches_traits(_spec(TRITON_ROUTE_EPILOGUE), traits)
    assert spec_matches_traits(_spec(TORCH_ROUTE_EPILOGUE), traits)
    tensors = _layout(**traits, strided=strided)
    assert route_epilogue_traits(*tensors) == traits
    reason = route_epilogue_rejection(*tensors)
    assert attribute in reason and "CUDA" not in reason


def test_host_guard_refuses_a_non_bool_renormalize_before_the_device_clause():
    """``renormalize`` selects the compiled variant and is never coerced: the guard
    names it (``1 == True`` in Python, so the spec's ``{True, False}`` trait cannot
    tell them apart -- the facade's own ``isinstance`` check is the barrier there)."""
    scores, ids, _ = _layout(1)
    reason = route_epilogue_rejection(scores, ids, 1)
    assert reason.startswith("renormalize must be a bool") and "int" in reason
    assert "CUDA" not in reason
    assert route_epilogue_rejection(scores, ids, None).startswith(
        "renormalize must be a bool"
    )


@pytest.mark.parametrize("rows", ROWS_REJECTED)
def test_empty_grid_is_refused_by_the_host_guard_alone_and_never_launched(
    rows, selection_events
):
    """``rows == 0`` has no spec trait to filter it (ranked selection lands on the torch
    kernel anyway), so the host guard is the only barrier: the reason names ``rows``
    before any device clause, and a by-name override raises without fallback."""
    traits = _traits(rows)
    assert spec_matches_traits(_spec(TRITON_ROUTE_EPILOGUE), traits)
    assert spec_matches_traits(_spec(TORCH_ROUTE_EPILOGUE), traits)
    scores, ids, renormalize = _layout(rows)
    assert route_epilogue_traits(scores, ids, renormalize) == traits
    reason = route_epilogue_rejection(scores, ids, renormalize)
    assert reason.startswith("rows=0") and "CUDA" not in reason
    with kernel_override(*OPERATOR, TRITON_ROUTE_EPILOGUE):
        with pytest.raises(ValueError, match="rows=0") as info:
            moe_route_epilogue(
                scores, ids, renormalize=renormalize, solution=None, override=None
            )
    assert str(info.value).startswith(f"{TRITON_ROUTE_EPILOGUE} cannot serve this call")
    assert [event.kernel_name for event in selection_events] == [TRITON_ROUTE_EPILOGUE]


@pytest.mark.parametrize(
    "platform_name", ["b200_platform", "h100_platform", "mi350_platform"]
)
@pytest.mark.parametrize("renormalize", RENORMALIZE_VALUES)
@pytest.mark.parametrize("rows", (1, 2, 4, 8, 16, 4096))
def test_ranked_selection_never_picks_the_reference_kernel(
    rows, renormalize, platform_name, request
):
    platform = request.getfixturevalue(platform_name)
    selected = select_kernel(
        *OPERATOR,
        SIGNATURE,
        platform=platform,
        traits=_traits(rows, renormalize=renormalize),
    )
    assert selected.name == TORCH_ROUTE_EPILOGUE
    assert selected.impl.__name__ == TORCH_ROUTE_EPILOGUE


def test_vendor_gate_filters_the_triton_kernel_off_nvidia(
    b200_platform, mi350_platform
):
    def candidates(platform):
        return [
            spec.name
            for spec in KernelRegistry.get().get_for_operator(
                *OPERATOR, platform=platform, format_signature=SIGNATURE
            )
        ]

    assert candidates(b200_platform) == [TORCH_ROUTE_EPILOGUE, TRITON_ROUTE_EPILOGUE]
    assert candidates(mi350_platform) == [TORCH_ROUTE_EPILOGUE]


def test_explain_selection_marks_torch_and_reports_the_by_name_override(
    b200_platform,
):
    both = explain_selection(
        *OPERATOR, SIGNATURE, platform=b200_platform, traits=_traits(1)
    )
    assert "Candidates (2 matched, 2 registered)" in both
    assert f"1. {TORCH_ROUTE_EPILOGUE}  [SELECTED]" in both
    assert f"2. {TRITON_ROUTE_EPILOGUE}" in both and "Override: none" in both
    # A trait the Triton spec constrains (``experts``): the kernel is filtered out.
    one = explain_selection(
        *OPERATOR, SIGNATURE, platform=b200_platform, traits=_traits(1, experts=128)
    )
    assert "Candidates (1 matched, 2 registered)" in one
    assert f"1. {TORCH_ROUTE_EPILOGUE}  [SELECTED]" in one
    assert one.index("Filtered out:") < one.index(f"- {TRITON_ROUTE_EPILOGUE}")
    # Any row count with the example shape, either renormalize value: the override
    # marks the reference kernel among the candidates (prefill row counts included).
    for rows, renormalize in ((1, True), (2, False), (1024, True)):
        forced = explain_selection(
            *OPERATOR,
            SIGNATURE,
            platform=b200_platform,
            traits=_traits(rows, renormalize=renormalize),
            override=TRITON_ROUTE_EPILOGUE,
        )
        assert (
            f"Override: {TRITON_ROUTE_EPILOGUE} (explicit override= argument)" in forced
        )
        assert f"{TRITON_ROUTE_EPILOGUE}  [SELECTED (override)]" in forced
        assert "not among the matched candidates" not in forced
    # A rejected layout (non-contiguous): the kernel is not a candidate, so only
    # the bypass note names it.
    forced = explain_selection(
        *OPERATOR,
        SIGNATURE,
        platform=b200_platform,
        traits=_traits(1, contiguous=False),
        override=TRITON_ROUTE_EPILOGUE,
    )
    assert "[SELECTED (override)]" not in forced
    assert (
        f"Override selects {TRITON_ROUTE_EPILOGUE}, which is not among the matched "
        "candidates" in forced
    )


def test_override_by_name_ignores_traits_and_the_cpu_host_guard_raises_without_fallback(
    selection_events,
):
    with kernel_override(*OPERATOR, TRITON_ROUTE_EPILOGUE):
        # By name: neither an admitted nor a rejected trait dict changes the result.
        for traits in (_traits(1), _traits(1024), _traits(1, experts=128)):
            selected = select_kernel(*OPERATOR, SIGNATURE, traits=traits)
            assert selected.name == TRITON_ROUTE_EPILOGUE
        scores, ids = _route_inputs(1, seed=1)
        with pytest.raises(ValueError, match="CUDA") as info:
            moe_route_epilogue(
                scores, ids, renormalize=True, solution=None, override=None
            )
    assert str(info.value).startswith(f"{TRITON_ROUTE_EPILOGUE} cannot serve this call")
    assert [
        (event.kernel_name, event.source, event.override) for event in selection_events
    ] == [(TRITON_ROUTE_EPILOGUE, "override", TRITON_ROUTE_EPILOGUE)] * 4


def test_environment_override_outranks_the_context_and_the_argument(monkeypatch):
    scores, ids = _route_inputs(1, seed=2)
    expected_weights, expected_ids = _reference(scores, ids, True)
    monkeypatch.setenv(ROUTE_EPILOGUE_OVERRIDE_ENV, TORCH_ROUTE_EPILOGUE)
    with kernel_override(*OPERATOR, TRITON_ROUTE_EPILOGUE):
        assert (
            select_kernel(*OPERATOR, SIGNATURE, traits=_traits(1)).name
            == TORCH_ROUTE_EPILOGUE
        )
        weights, out_ids = moe_route_epilogue(
            scores, ids, renormalize=True, solution=None, override=None
        )
    assert torch.equal(weights.view(torch.int32), expected_weights.view(torch.int32))
    assert torch.equal(out_ids, expected_ids)
    monkeypatch.setenv(ROUTE_EPILOGUE_OVERRIDE_ENV, TRITON_ROUTE_EPILOGUE)
    with pytest.raises(ValueError, match="CUDA"):
        moe_route_epilogue(
            scores, ids, renormalize=True, solution=None, override=TORCH_ROUTE_EPILOGUE
        )


def test_cross_operator_override_is_refused_by_the_facade():
    other = KernelRegistry.get().get_by_name(CROSS_OPERATOR_KERNEL)
    assert other is not None and (other.family, other.mode) == ("gemm", "decode_gemv")
    scores, ids = _route_inputs(1, seed=3)
    with pytest.raises(ValueError, match="not registered under moe.route_epilogue"):
        moe_route_epilogue(
            scores, ids, renormalize=True, solution=None, override=CROSS_OPERATOR_KERNEL
        )
    with kernel_override(*OPERATOR, CROSS_OPERATOR_KERNEL):
        with pytest.raises(ValueError, match="not registered under moe.route_epilogue"):
            moe_route_epilogue(
                scores, ids, renormalize=True, solution=None, override=None
            )


@pytest.mark.parametrize("renormalize", RENORMALIZE_VALUES)
@pytest.mark.parametrize("rows", (1, 2, 4, 8, 33, 1024))
def test_facade_equals_the_route_statements_byte_for_byte_on_cpu(
    rows, renormalize, selection_events
):
    scores, ids = _route_inputs(rows, seed=20260921 + rows)
    before = (scores.view(torch.int32).clone(), ids.clone())
    expected_weights, expected_ids = _reference(scores, ids, renormalize)
    weights, out_ids = moe_route_epilogue(
        scores, ids, renormalize=renormalize, solution=None, override=None
    )
    assert weights.dtype is torch.float32 and weights.shape == (rows, TOPK)
    assert out_ids.dtype is torch.int32 and out_ids.shape == (rows, TOPK)
    for out in (weights, out_ids):
        assert out.is_contiguous()
        assert out.data_ptr() not in (scores.data_ptr(), ids.data_ptr())
    assert torch.equal(weights.view(torch.int32), expected_weights.view(torch.int32))
    assert torch.equal(out_ids, expected_ids) and torch.equal(
        out_ids, ids.to(torch.int32)
    )
    assert torch.equal(scores.view(torch.int32), before[0])
    assert torch.equal(ids, before[1])
    for kwargs in (
        {"solution": "torch", "override": None},
        {"solution": None, "override": TORCH_ROUTE_EPILOGUE},
    ):
        again_weights, again_ids = moe_route_epilogue(
            scores, ids, renormalize=renormalize, **kwargs
        )
        assert torch.equal(
            again_weights.view(torch.int32), expected_weights.view(torch.int32)
        )
        assert torch.equal(again_ids, expected_ids)
    if not renormalize:
        # Without renormalization the weights are the gathered scores bit for bit:
        # the -0.0 and the NaN payload of SPECIAL_ROW survive untouched.
        gathered = scores.gather(-1, ids)
        assert torch.equal(weights.view(torch.int32), gathered.view(torch.int32))
        if rows > SPECIAL_ROW:
            special = weights.view(torch.int32)[SPECIAL_ROW]
            assert (
                special[NEGATIVE_ZERO_SLOT].item()
                == torch.tensor(-0.0).view(torch.int32).item()
            )
            assert special[NAN_PAYLOAD_SLOT].item() == NAN_PAYLOAD_BITS
    else:
        # Row 0 is a realistic sigmoid row at every row count (the special words live
        # in SPECIAL_ROW): finite renormalized weights, compared bit for bit above.
        assert torch.isfinite(weights[0]).all()
        if rows > SPECIAL_ROW:
            # The NaN payload poisons the whole row's sum.
            assert torch.isnan(weights[SPECIAL_ROW]).all()
        if rows > TIE_ROW:
            assert torch.all(weights[TIE_ROW] == 0.125)
        if rows > NEG_INF_ROW:
            finite = [slot for slot in range(TOPK) if slot != NEG_INF_SLOT]
            assert torch.all(weights[NEG_INF_ROW, finite] == 0.0)
            assert torch.all(torch.signbit(weights[NEG_INF_ROW, finite]))
            assert torch.isnan(weights[NEG_INF_ROW, NEG_INF_SLOT])
    assert {event.kernel_name for event in selection_events} == {TORCH_ROUTE_EPILOGUE}


def test_metadata_validation_raises_before_selection():
    scores, ids = _route_inputs(4, seed=5)
    with pytest.raises(ValueError, match=r"scores must have shape \[rows, experts\]"):
        moe_route_epilogue(
            scores.reshape(-1), ids, renormalize=True, solution=None, override=None
        )
    with pytest.raises(ValueError, match="ids must have shape"):
        moe_route_epilogue(
            scores, ids[:2], renormalize=True, solution=None, override=None
        )
    with pytest.raises(ValueError, match="ids must have shape"):
        moe_route_epilogue(
            scores, ids[:, :0], renormalize=True, solution=None, override=None
        )
    with pytest.raises(ValueError, match="ids must have shape"):
        moe_route_epilogue(
            scores,
            ids.repeat(1, EXPERTS // TOPK + 1),
            renormalize=True,
            solution=None,
            override=None,
        )
    with pytest.raises(ValueError, match="ids must have shape"):
        moe_route_epilogue(
            scores, ids.reshape(-1), renormalize=True, solution=None, override=None
        )
    with pytest.raises(ValueError, match="renormalize must be a bool"):
        moe_route_epilogue(scores, ids, renormalize=1, solution=None, override=None)


def test_bf16_scores_have_no_registered_kernel():
    scores, ids = _route_inputs(4, seed=6)
    with pytest.raises(NoKernelFoundError, match="bfloat16") as info:
        moe_route_epilogue(
            scores.bfloat16(), ids, renormalize=True, solution=None, override=None
        )
    assert "moe.route_epilogue" in str(info.value)


@pytest.mark.parametrize("renormalize", RENORMALIZE_VALUES)
def test_zero_rows_return_empty_outputs_through_the_torch_kernel(
    renormalize, selection_events
):
    scores, ids, _ = _layout(0)
    weights, out_ids = moe_route_epilogue(
        scores, ids, renormalize=renormalize, solution=None, override=None
    )
    assert weights.dtype is torch.float32 and weights.shape == (0, TOPK)
    assert out_ids.dtype is torch.int32 and out_ids.shape == (0, TOPK)
    assert [event.kernel_name for event in selection_events] == [TORCH_ROUTE_EPILOGUE]


def test_facade_path_arguments_are_keyword_only_without_defaults():
    parameters = inspect.signature(moe_route_epilogue).parameters
    assert list(parameters) == ["scores", "ids", "renormalize", "solution", "override"]
    for name in ("renormalize", "solution", "override"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


def test_kernel_source_pins_the_torch_cuda_k8_reduction_tree_literally():
    """The tree is bound in the kernel source by AST, and a float32 emulation shows
    it equals the descending shuffle-down loop bit for bit while the sequential order
    does not. This pins what the kernel encodes; whether this torch build's CUDA sum
    uses the same order is the GPU pin in test/nvidia/ops/moe/test_route_epilogue_cuda.py
    -- a CPU torch.sum has another order and is deliberately not compared here.
    """
    source = TRITON_SOURCE.read_text()
    for literal in (
        '"add.rn.f32 $0, $1, $2;"',
        "tl.div_rn(selected, denominator)",
        "num_warps=1",
        "enable_fp_fusion=False",
        "enable_reflect_ftz=False",
        "launch_pdl=False",
        "ids.to(tl.int32)",
    ):
        assert literal in source, literal
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_route_epilogue"
    )
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }
    assert (
        assignments["even"]
        == "_add_rn(_add_rn(values[0], values[4]), _add_rn(values[2], values[6]))"
    )
    assert (
        assignments["odd"]
        == "_add_rn(_add_rn(values[1], values[5]), _add_rn(values[3], values[7]))"
    )
    assert assignments["denominator"] == "_add_rn(_add_rn(even, odd), 1e-20)"
    assert assignments["selected"] == "tl.div_rn(selected, denominator)"
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
    }
    assert {"tl.sum", "tl.fdiv", "tl.sigmoid", "tl.topk", "tl.sort"}.isdisjoint(calls)
    assert triton_module._EPSILON == 1e-20
    launch_options = triton_module._LAUNCH_OPTIONS
    assert launch_options == {
        "num_warps": 1,
        "enable_fp_fusion": False,
        "enable_reflect_ftz": False,
        "launch_pdl": False,
    }
    # Emulation: the literal tree equals the shuffle-down loop on every fixture row
    # (the module's pin rows plus the zero row), sequential differs on the pinned row.
    fixtures = [[0.0] * 8, list(SEQUENTIAL_DIFFERS_ROW)]
    fixtures += reduction_order_fixture().tolist()
    assert len(fixtures) >= 130
    for values in fixtures:
        assert _bits(_shuffle_down_tree(values)) == _bits(
            _explicit_tree(values)
        ), values
    assert _bits(_explicit_tree(list(PATTERNS[ROUNDING_ROW]))) == _bits(
        _shuffle_down_tree(list(PATTERNS[ROUNDING_ROW]))
    )
    assert _sequential(SEQUENTIAL_DIFFERS_ROW) != _explicit_tree(SEQUENTIAL_DIFFERS_ROW)
    # ``k8_reduction_tree`` (the GPU pin's reference) is the same tree, on CPU tensors.
    rows = torch.tensor(fixtures, dtype=torch.float32)
    emulated = torch.tensor(
        [_explicit_tree(values) for values in fixtures], dtype=torch.float32
    )
    assert torch.equal(
        k8_reduction_tree(rows).view(torch.int32),
        emulated.unsqueeze(-1).view(torch.int32),
    )
    assert k8_reduction_tree(rows).shape == (len(fixtures), 1)
    with pytest.raises(ValueError, match="last dimension of 8"):
        k8_reduction_tree(rows[:, :4])
    with pytest.raises(ValueError, match="float32"):
        k8_reduction_tree(rows.double())
    # The CUDA order is the contract: the pin refuses to run on a CPU device instead
    # of comparing a CPU sum whose order says nothing about the CUDA kernel.
    with pytest.raises(ValueError, match="CUDA device"):
        reduction_order_mismatch(torch.device("cpu"))


def test_reduction_order_fixture_is_deterministic_and_discriminating():
    rows = reduction_order_fixture()
    assert (
        rows.dtype is torch.float32 and rows.shape[1] == TOPK and rows.is_contiguous()
    )
    assert torch.equal(
        rows.view(torch.int32), reduction_order_fixture().view(torch.int32)
    )
    assert rows[0].tolist() == list(PATTERNS[ROUNDING_ROW])
    assert rows[1].tolist() == list(PATTERNS[DENORMAL_ROW])
    assert torch.isneginf(rows[3]).any()
    # Orders differ on the fixture: the adjacent-pairwise order and the sequential
    # order each disagree with the pinned tree on at least one row.
    lists = rows.tolist()
    pairwise = [
        _add(
            _add(_add(v[0], v[1]), _add(v[2], v[3])),
            _add(_add(v[4], v[5]), _add(v[6], v[7])),
        )
        for v in lists
    ]
    tree = [_explicit_tree(v) for v in lists]
    sequential = [_sequential(v) for v in lists]
    assert any(_bits(a) != _bits(b) for a, b in zip(pairwise, tree, strict=True))
    assert any(_bits(a) != _bits(b) for a, b in zip(sequential, tree, strict=True))
