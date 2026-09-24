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

"""``moe.group_mask`` through ``select_kernel``: registration, admission, the by-name switch and byte equality with the route statements.

CPU only. Importing ``tokenspeed_kernel.ops.moe.group_mask`` registers both
kernels (the package init imports the Triton module); ``load_builtin_kernels``
re-populates the singleton when an earlier test module reset it. The torch
kernel runs on CPU, so ranked selection and the facade are exercised for real.
The Triton kernel is never launched here (``test/nvidia/ops/moe/test_group_mask_bits.py``
covers it): on CPU tensors its host guard rejects the call, which is what the
override tests assert -- the by-name switch has no fallback. Its admission has no
row clause (any positive row count is admitted, see the Triton module doc): only
``rows == 0`` is refused, by the host guard alone, so that an empty grid is never
launched.
"""

from __future__ import annotations

import inspect

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.group_mask import (
    GROUP_MASK_FAMILY,
    GROUP_MASK_MODE,
    GROUP_MASK_OVERRIDE_ENV,
    GROUP_MASK_SIGNATURES,
    TORCH_GROUP_MASK,
    TRITON_GROUP_MASK_BITS,
    group_mask_traits,
    moe_group_mask,
)
from tokenspeed_kernel.ops.moe.triton.group_mask import group_mask_bits_rejection
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

OPERATOR = (GROUP_MASK_FAMILY, GROUP_MASK_MODE)
SIGNATURE = next(iter(GROUP_MASK_SIGNATURES))
# Grouped-routing example shape (256 experts in 8 groups, 4 groups kept).
EXPERTS, NUM_GROUPS, TOPK_GROUPS = 256, 8, 4
GROUP_SIZE = EXPERTS // NUM_GROUPS
# Any positive row count is admitted (decode batches, padded batches, prefill);
# only the empty grid is refused, and only by the host guard (no spec trait).
ROWS_ADMITTED = (1, 2, 3, 4, 5, 7, 8, 16, 32, 33, 64, 128, 256, 1024, 4096)
ROWS_REJECTED = (0,)
# Synthetic router-logit distribution (std 0.6 logits, std 0.1 bias).
LOGIT_STD, BIAS_STD = 0.6, 0.1
TIE_EXPERTS = (0, 1, 2, 32, 33, 64, 65, 96, 97)  # nine experts across four groups
CROSS_OPERATOR_KERNEL = "torch_decode_gemv"  # registered under gemm.decode_gemv
NEGATIVE_INFINITY_BITS = torch.tensor([float("-inf")]).view(torch.int32).item()
NAN_PAYLOAD_BITS = 0x7FC01234


@pytest.fixture(autouse=True)
def group_mask_registry():
    """Both kernels registered; no cached selection or override witness leaks between tests."""
    if KernelRegistry.get().get_by_name(TORCH_GROUP_MASK) is None:
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
    num_groups: int = NUM_GROUPS,
    topk_groups: int = TOPK_GROUPS,
    contiguous: bool = True,
) -> dict[str, int | bool]:
    return {
        "rows": rows,
        "experts": experts,
        "num_groups": num_groups,
        "topk_groups": topk_groups,
        "contiguous": contiguous,
    }


def _layout(
    rows: int,
    experts: int = EXPERTS,
    num_groups: int = NUM_GROUPS,
    topk_groups: int = TOPK_GROUPS,
    contiguous: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU tensors carrying the metadata of one trait dict (values are irrelevant)."""
    if contiguous:
        choices = torch.zeros(rows, experts)
    else:
        choices = torch.zeros(rows, 2 * experts)[:, ::2]
    group_ids = torch.zeros(rows, topk_groups, dtype=torch.int64)
    group_scores = torch.zeros(rows, num_groups)
    grouped = choices.reshape(rows, num_groups, experts // num_groups)
    return choices, group_ids, group_scores, grouped


def _inject_special_words(choices: torch.Tensor, group_ids: torch.Tensor) -> None:
    """Row 0: ``-inf``, ``+inf``, ``-0.0`` and a NaN payload in a kept group; ``+inf``,
    ``-inf`` and a NaN payload in an excluded group. Byte equality must survive all."""
    kept = int(group_ids[0, 0])
    excluded = next(g for g in range(NUM_GROUPS) if g not in group_ids[0].tolist())
    words = choices.view(torch.int32)
    choices[0, kept * GROUP_SIZE + 0] = float("-inf")
    choices[0, kept * GROUP_SIZE + 1] = float("inf")
    choices[0, kept * GROUP_SIZE + 2] = -0.0
    words[0, kept * GROUP_SIZE + 3] = NAN_PAYLOAD_BITS
    choices[0, excluded * GROUP_SIZE + 0] = float("inf")
    choices[0, excluded * GROUP_SIZE + 1] = float("-inf")
    words[0, excluded * GROUP_SIZE + 2] = NAN_PAYLOAD_BITS


def _route_intermediates(
    rows: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The grouped-routing intermediates on seeded inputs (a tie row, an all-negative row),
    then the special words injected into ``choices`` after the group selection."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, EXPERTS, generator=generator) * LOGIT_STD
    bias = torch.randn(EXPERTS, generator=generator) * BIAS_STD
    bias[list(TIE_EXPERTS)] = BIAS_STD
    if rows > 1:  # nine experts across four groups share one choice
        logits[1] = -3.0
        logits[1, list(TIE_EXPERTS)] = 6.0
    if rows > 2:  # every sigmoid ~2e-22: the +1e-20 path of the route
        logits[2] = -50.0
    scores = logits.float().sigmoid()
    choices = scores + bias
    grouped = choices.reshape(rows, NUM_GROUPS, GROUP_SIZE)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    group_ids = group_scores.topk(TOPK_GROUPS, dim=-1, sorted=True).indices
    if rows > 0:
        _inject_special_words(choices, group_ids)
    return choices, group_ids, group_scores, grouped


def _reference(choices, group_ids, group_scores, grouped) -> torch.Tensor:
    """The three statements the torch grouped router inlines, plus the ``masked_fill``."""
    keep_groups = torch.zeros_like(group_scores, dtype=torch.bool)
    keep_groups.scatter_(1, group_ids, True)
    keep = keep_groups.unsqueeze(-1).expand_as(grouped).reshape_as(choices)
    return choices.masked_fill(~keep, float("-inf"))


def _words(value: torch.Tensor) -> torch.Tensor:
    return value.view(torch.int32) if value.dtype is torch.float32 else value


def test_registration_lists_torch_then_triton_with_one_shared_signature():
    registry = KernelRegistry.get()
    specs = registry.list_kernels(*OPERATOR)
    assert [spec.name for spec in specs] == [TORCH_GROUP_MASK, TRITON_GROUP_MASK_BITS]
    torch_spec, triton_spec = specs
    assert (torch_spec.solution, torch_spec.priority) == ("torch", Priority.PORTABLE)
    assert (triton_spec.solution, triton_spec.priority) == (
        "triton",
        Priority.REFERENCE,
    )
    assert (
        torch_spec.format_signatures
        == triton_spec.format_signatures
        == GROUP_MASK_SIGNATURES
        == frozenset({SIGNATURE})
    )
    assert torch_spec.capability == CapabilityRequirement()
    assert triton_spec.capability == CapabilityRequirement(
        vendors=frozenset({"nvidia"})
    )
    assert torch_spec.traits == {}
    # No ``rows`` trait: the row count is the grid size only.
    assert triton_spec.traits == {
        "experts": frozenset({EXPERTS}),
        "num_groups": frozenset({NUM_GROUPS}),
        "topk_groups": frozenset({TOPK_GROUPS}),
        "contiguous": frozenset({True}),
    }
    torch_text = describe_kernel(TORCH_GROUP_MASK)
    assert "Operator: moe.group_mask" in torch_text
    assert "Solution: torch" in torch_text and "Priority: 4 (PORTABLE)" in torch_text
    triton_text = describe_kernel(TRITON_GROUP_MASK_BITS)
    assert "Operator: moe.group_mask" in triton_text
    assert (
        "Solution: triton" in triton_text and "Priority: 0 (REFERENCE)" in triton_text
    )
    assert registry.get_impl(TORCH_GROUP_MASK).__name__ == TORCH_GROUP_MASK
    assert registry.get_impl(TRITON_GROUP_MASK_BITS).__name__ == TRITON_GROUP_MASK_BITS
    assert GROUP_MASK_OVERRIDE_ENV == "TOKENSPEED_KERNEL_OVERRIDE_MOE_GROUP_MASK"
    assert tokenspeed_kernel.moe_group_mask is moe_group_mask


@pytest.mark.parametrize("rows", ROWS_ADMITTED)
def test_triton_spec_admits_every_positive_row_count_and_the_host_guard_agrees_up_to_the_device(
    rows,
):
    traits = _traits(rows)
    assert spec_matches_traits(_spec(TRITON_GROUP_MASK_BITS), traits)
    assert spec_matches_traits(_spec(TORCH_GROUP_MASK), traits)
    tensors = _layout(rows)
    assert group_mask_traits(*tensors) == traits
    # Lock-step: on CPU tensors of an admitted layout only the device clause fails.
    reason = group_mask_bits_rejection(*tensors)
    assert "CUDA" in reason and "cpu" in reason


@pytest.mark.parametrize(
    ("traits", "attribute"),
    [
        (_traits(1, experts=128), "choices"),
        (_traits(1, num_groups=4), "group_scores"),
        (_traits(1, topk_groups=2), "group_ids"),
        (_traits(1, contiguous=False), "contiguous"),
        (_traits(4096, experts=512), "choices"),
    ],
)
def test_triton_spec_rejects_off_shape_traits_and_the_host_guard_names_the_attribute(
    traits, attribute
):
    assert not spec_matches_traits(_spec(TRITON_GROUP_MASK_BITS), traits)
    assert spec_matches_traits(_spec(TORCH_GROUP_MASK), traits)
    tensors = _layout(**traits)
    assert group_mask_traits(*tensors) == traits
    reason = group_mask_bits_rejection(*tensors)
    assert attribute in reason and "CUDA" not in reason


@pytest.mark.parametrize("rows", ROWS_REJECTED)
def test_empty_grid_is_refused_by_the_host_guard_alone_and_never_launched(
    rows, selection_events
):
    """``rows == 0`` has no spec trait to filter it (ranked selection lands on the torch
    kernel anyway), so the host guard is the only barrier: the reason names ``rows``
    before any device clause, and a by-name override raises without fallback."""
    traits = _traits(rows)
    assert spec_matches_traits(_spec(TRITON_GROUP_MASK_BITS), traits)
    assert spec_matches_traits(_spec(TORCH_GROUP_MASK), traits)
    tensors = _layout(rows)
    assert group_mask_traits(*tensors) == traits
    reason = group_mask_bits_rejection(*tensors)
    assert reason.startswith("rows=0") and "CUDA" not in reason
    with kernel_override(*OPERATOR, TRITON_GROUP_MASK_BITS):
        with pytest.raises(ValueError, match="rows=0") as info:
            moe_group_mask(*tensors, solution=None, override=None)
    assert str(info.value).startswith(
        f"{TRITON_GROUP_MASK_BITS} cannot serve this call"
    )
    assert [event.kernel_name for event in selection_events] == [TRITON_GROUP_MASK_BITS]


@pytest.mark.parametrize(
    "platform_name", ["b200_platform", "h100_platform", "mi350_platform"]
)
@pytest.mark.parametrize("rows", (1, 2, 4, 8, 16, 4096))
def test_ranked_selection_never_picks_the_reference_kernel(
    rows, platform_name, request
):
    platform = request.getfixturevalue(platform_name)
    selected = select_kernel(
        *OPERATOR, SIGNATURE, platform=platform, traits=_traits(rows)
    )
    assert selected.name == TORCH_GROUP_MASK
    assert selected.impl.__name__ == TORCH_GROUP_MASK


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

    assert candidates(b200_platform) == [TORCH_GROUP_MASK, TRITON_GROUP_MASK_BITS]
    assert candidates(mi350_platform) == [TORCH_GROUP_MASK]


def test_explain_selection_marks_torch_and_reports_the_by_name_override(
    b200_platform,
):
    both = explain_selection(
        *OPERATOR, SIGNATURE, platform=b200_platform, traits=_traits(1)
    )
    assert "Candidates (2 matched, 2 registered)" in both
    assert f"1. {TORCH_GROUP_MASK}  [SELECTED]" in both
    assert f"2. {TRITON_GROUP_MASK_BITS}" in both and "Override: none" in both
    # A trait the Triton spec constrains (``experts``): the kernel is filtered out.
    one = explain_selection(
        *OPERATOR, SIGNATURE, platform=b200_platform, traits=_traits(1, experts=128)
    )
    assert "Candidates (1 matched, 2 registered)" in one
    assert f"1. {TORCH_GROUP_MASK}  [SELECTED]" in one
    assert one.index("Filtered out:") < one.index(f"- {TRITON_GROUP_MASK_BITS}")
    # Any row count with the example shape: the override marks the reference kernel
    # among the candidates (prefill row counts included).
    for rows in (1, 2, 1024):
        forced = explain_selection(
            *OPERATOR,
            SIGNATURE,
            platform=b200_platform,
            traits=_traits(rows),
            override=TRITON_GROUP_MASK_BITS,
        )
        assert (
            f"Override: {TRITON_GROUP_MASK_BITS} (explicit override= argument)"
            in forced
        )
        assert f"{TRITON_GROUP_MASK_BITS}  [SELECTED (override)]" in forced
        assert "not among the matched candidates" not in forced
    # A rejected layout (non-contiguous): the kernel is not a candidate, so only
    # the bypass note names it.
    forced = explain_selection(
        *OPERATOR,
        SIGNATURE,
        platform=b200_platform,
        traits=_traits(1, contiguous=False),
        override=TRITON_GROUP_MASK_BITS,
    )
    assert "[SELECTED (override)]" not in forced
    assert (
        f"Override selects {TRITON_GROUP_MASK_BITS}, which is not among the matched "
        "candidates" in forced
    )


def test_override_by_name_ignores_traits_and_the_cpu_host_guard_raises_without_fallback(
    selection_events,
):
    with kernel_override(*OPERATOR, TRITON_GROUP_MASK_BITS):
        # By name: neither an admitted nor a rejected trait dict changes the result.
        for traits in (_traits(1), _traits(1024), _traits(1, experts=128)):
            selected = select_kernel(*OPERATOR, SIGNATURE, traits=traits)
            assert selected.name == TRITON_GROUP_MASK_BITS
        with pytest.raises(ValueError, match="CUDA") as info:
            moe_group_mask(
                *_route_intermediates(1, seed=1), solution=None, override=None
            )
    assert str(info.value).startswith(
        f"{TRITON_GROUP_MASK_BITS} cannot serve this call"
    )
    assert [
        (event.kernel_name, event.source, event.override) for event in selection_events
    ] == [(TRITON_GROUP_MASK_BITS, "override", TRITON_GROUP_MASK_BITS)] * 4


def test_environment_override_outranks_the_context_and_the_argument(monkeypatch):
    tensors = _route_intermediates(1, seed=2)
    expected = _reference(*tensors)
    monkeypatch.setenv(GROUP_MASK_OVERRIDE_ENV, TORCH_GROUP_MASK)
    with kernel_override(*OPERATOR, TRITON_GROUP_MASK_BITS):
        assert (
            select_kernel(*OPERATOR, SIGNATURE, traits=_traits(1)).name
            == TORCH_GROUP_MASK
        )
        out = moe_group_mask(*tensors, solution=None, override=None)
    assert torch.equal(out.view(torch.int32), expected.view(torch.int32))
    monkeypatch.setenv(GROUP_MASK_OVERRIDE_ENV, TRITON_GROUP_MASK_BITS)
    with pytest.raises(ValueError, match="CUDA"):
        moe_group_mask(*tensors, solution=None, override=TORCH_GROUP_MASK)


def test_cross_operator_override_is_refused_by_the_facade():
    other = KernelRegistry.get().get_by_name(CROSS_OPERATOR_KERNEL)
    assert other is not None and (other.family, other.mode) == ("gemm", "decode_gemv")
    tensors = _route_intermediates(1, seed=3)
    with pytest.raises(ValueError, match="not registered under moe.group_mask"):
        moe_group_mask(*tensors, solution=None, override=CROSS_OPERATOR_KERNEL)
    with kernel_override(*OPERATOR, CROSS_OPERATOR_KERNEL):
        with pytest.raises(ValueError, match="not registered under moe.group_mask"):
            moe_group_mask(*tensors, solution=None, override=None)


@pytest.mark.parametrize("rows", (1, 2, 4, 8, 33, 1024))
def test_facade_equals_the_route_statements_byte_for_byte_on_cpu(
    rows, selection_events
):
    tensors = _route_intermediates(rows, seed=20260921 + rows)
    before = [value.clone() for value in tensors]
    expected = _reference(*tensors)
    out = moe_group_mask(*tensors, solution=None, override=None)
    assert out.dtype is torch.float32 and out.shape == (rows, EXPERTS)
    assert out.is_contiguous() and out.data_ptr() != tensors[0].data_ptr()
    assert torch.equal(out.view(torch.int32), expected.view(torch.int32))
    for value, saved in zip(tensors, before, strict=True):
        assert torch.equal(_words(value), _words(saved))
    for kwargs in (
        {"solution": "torch", "override": None},
        {"solution": None, "override": TORCH_GROUP_MASK},
    ):
        again = moe_group_mask(*tensors, **kwargs)
        assert torch.equal(again.view(torch.int32), expected.view(torch.int32))
    # Kept words (including -inf, +inf, -0.0 and the NaN payload) are copied bit for
    # bit; every excluded word is the -inf pattern.
    choices, group_ids = tensors[0], tensors[1]
    kept = int(group_ids[0, 0])
    excluded = next(g for g in range(NUM_GROUPS) if g not in group_ids[0].tolist())
    kept_slice = slice(kept * GROUP_SIZE, (kept + 1) * GROUP_SIZE)
    excluded_slice = slice(excluded * GROUP_SIZE, (excluded + 1) * GROUP_SIZE)
    assert torch.equal(
        out.view(torch.int32)[0, kept_slice], choices.view(torch.int32)[0, kept_slice]
    )
    assert torch.all(out.view(torch.int32)[0, excluded_slice] == NEGATIVE_INFINITY_BITS)
    assert torch.isnan(out[0, kept * GROUP_SIZE + 3])
    assert {event.kernel_name for event in selection_events} == {TORCH_GROUP_MASK}


def test_metadata_validation_raises_before_selection():
    choices, group_ids, group_scores, grouped = _route_intermediates(4, seed=5)
    with pytest.raises(ValueError, match=r"choices must have shape \[rows, experts\]"):
        moe_group_mask(
            choices.reshape(-1),
            group_ids,
            group_scores,
            grouped,
            solution=None,
            override=None,
        )
    with pytest.raises(ValueError, match="group_scores must have shape"):
        moe_group_mask(
            choices, group_ids, group_scores[:2], grouped, solution=None, override=None
        )
    with pytest.raises(ValueError, match="multiple of num_groups"):
        moe_group_mask(
            choices[:, :250],
            group_ids,
            group_scores,
            grouped,
            solution=None,
            override=None,
        )
    with pytest.raises(ValueError, match="group_ids must have shape"):
        moe_group_mask(
            choices,
            group_ids.repeat(1, 3),
            group_scores,
            grouped,
            solution=None,
            override=None,
        )
    with pytest.raises(ValueError, match="group_ids must have shape"):
        moe_group_mask(
            choices,
            group_ids[:, :0],
            group_scores,
            grouped,
            solution=None,
            override=None,
        )
    with pytest.raises(ValueError, match="grouped must have shape"):
        moe_group_mask(
            choices,
            group_ids,
            group_scores,
            grouped.reshape(4, 4, 64),
            solution=None,
            override=None,
        )


def test_bf16_choices_have_no_registered_kernel():
    choices, group_ids, group_scores, grouped = _route_intermediates(4, seed=6)
    with pytest.raises(NoKernelFoundError, match="bfloat16") as info:
        moe_group_mask(
            choices.bfloat16(),
            group_ids,
            group_scores,
            grouped.bfloat16(),
            solution=None,
            override=None,
        )
    assert "moe.group_mask" in str(info.value)


def test_zero_rows_return_an_empty_mask_through_the_torch_kernel(selection_events):
    out = moe_group_mask(*_layout(0), solution=None, override=None)
    assert out.dtype is torch.float32 and out.shape == (0, EXPERTS)
    assert [event.kernel_name for event in selection_events] == [TORCH_GROUP_MASK]


def test_facade_path_arguments_are_keyword_only_without_defaults():
    parameters = inspect.signature(moe_group_mask).parameters
    assert list(parameters) == [
        "choices",
        "group_ids",
        "group_scores",
        "grouped",
        "solution",
        "override",
    ]
    for name in ("solution", "override"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty
