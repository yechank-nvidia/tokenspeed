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

"""``triton_route_epilogue`` on an NVIDIA device: the Torch K8 reduction-order pin,
byte equality with ``torch_route_epilogue`` under the by-name switch, the ranked
default, rejected shapes and ``explain_selection``.

The kernel's bitwise contract rests on this Torch build's CUDA ``sum(dim=-1)``
over a contiguous FP32 ``[rows, 8]`` reducing in the shuffle-down order
``((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))``. A module-scoped autouse fixture runs
that pin first (``reduction_order_mismatch``) and, on a mismatch, fails every
test of the module with the reason -- a failure, never a skip, so a Torch
upgrade that changes the order cannot pass silently. Two negative controls (the
sequential and the adjacent-pairwise orders) prove the pin rows discriminate.

The in-tree kernel compiles to its own Triton hash (not a previously measured
binary's), so its correctness is established here anew: for every row count in
``ROWS`` (the row axis 1..4096 plus the non-powers of two 3, 5, 7 and 33) and
both ``renormalize`` values the facade is run once per arm inside
``kernel_override`` with a selection listener recording the witnesses, and the
two outputs must agree bit for bit -- ties, negatives, subnormals, the rounding
row, the underflow row, a ``-inf`` selected score, ``-0.0`` and a NaN payload
included. The only row count the kernel refuses is zero (an empty grid), by its
host guard.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe.route_epilogue import (
    ROUTE_EPILOGUE_FAMILY,
    ROUTE_EPILOGUE_MODE,
    ROUTE_EPILOGUE_SIGNATURES,
    TORCH_ROUTE_EPILOGUE,
    TRITON_ROUTE_EPILOGUE,
    moe_route_epilogue,
    route_epilogue_traits,
)
from tokenspeed_kernel.ops.moe.triton.route_epilogue import (
    k8_reduction_tree,
    reduction_order_fixture,
    reduction_order_mismatch,
)
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

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not Platform.get().is_nvidia,
    reason="requires an NVIDIA CUDA device",
)
pytestmark = requires_cuda

OPERATOR = (ROUTE_EPILOGUE_FAMILY, ROUTE_EPILOGUE_MODE)
SIGNATURE = next(iter(ROUTE_EPILOGUE_SIGNATURES))
DEVICE = torch.device("cuda", 0)
SEED = 20260921
# Row counts under test: the row axis (1..4096, decode through prefill) plus
# non-powers of two; every one must be admitted and byte-identical to the torch arm.
ROWS = (1, 2, 3, 4, 5, 7, 8, 16, 32, 33, 64, 128, 256, 1024, 4096)
RENORMALIZE_VALUES = (True, False)
# Grouped-routing example shape (256 experts in 8 groups, 4 groups kept, top-8).
EXPERTS, NUM_GROUPS, TOPK_GROUPS, TOPK = 256, 8, 4, 8
GROUP_SIZE = EXPERTS // NUM_GROUPS
LOGIT_STD, BIAS_STD = 0.6, 0.1
NAN_PAYLOAD_BITS = 0x7FC01234
NEGATIVE_ZERO_BITS = torch.tensor([-0.0]).view(torch.int32).item()
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
EPSILON = 1e-20


@pytest.fixture(scope="module", autouse=True)
def torch_k8_reduction_order():
    """The Torch side of the bitwise contract, checked before any byte-equality test.

    A non-empty reason fails every test of this module (``pytest.fail``, not a
    skip): the Triton kernel cannot be used on a Torch build whose CUDA K8 sum does
    not follow the pinned tree, and that must be loud.
    """
    if not torch.cuda.is_available() or not Platform.get().is_nvidia:
        return
    reason = reduction_order_mismatch(DEVICE)
    if reason:
        pytest.fail(reason)


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


def _route_inputs(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The grouped router up to its epilogue on seeded inputs on the device (the
    three mask statements inlined), then the pattern rows scattered into the
    selected slots and, in row 7, a ``-0.0`` and a NaN-payload selected score.

    Row 1 (when ``rows > 1``) has eight equal selected scores; row 2 negatives
    without cancellation; row 3 subnormals; row 4 the rounding row whose sum's bits
    depend on the summation order; row 5 all-underflow logits (every sigmoid about
    2e-22, the ``+1e-20`` path); row 6 a ``-inf`` selected score at slot 3; row 7
    (kernel-test-only) a ``-0.0`` word at slot 6 and a NaN payload at slot 7. Every
    other row, row 0 in particular, is drawn from the synthetic distribution, so every
    row count has at least one realistic sigmoid row whose renormalized bits are
    compared across the arms. ``rows == 0`` yields empty tensors for the
    rejected-shape test.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, EXPERTS, generator=generator) * LOGIT_STD
    bias = torch.randn(EXPERTS, generator=generator) * BIAS_STD
    if rows > UNDERFLOW_ROW:
        logits[UNDERFLOW_ROW] = -50.0
    logits, bias = logits.to(DEVICE), bias.to(DEVICE)
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
            scores[row, ids[row]] = torch.tensor(
                pattern, dtype=torch.float32, device=DEVICE
            )
    if rows > NEG_INF_ROW:
        scores[NEG_INF_ROW, ids[NEG_INF_ROW, NEG_INF_SLOT]] = float("-inf")
    if rows > SPECIAL_ROW:
        scores[SPECIAL_ROW, ids[SPECIAL_ROW, NEGATIVE_ZERO_SLOT]] = -0.0
        words = scores.view(torch.int32)
        words[SPECIAL_ROW, ids[SPECIAL_ROW, NAN_PAYLOAD_SLOT]] = NAN_PAYLOAD_BITS
    return scores.contiguous(), ids.contiguous()


def _strided(value: torch.Tensor) -> torch.Tensor:
    """A non-contiguous view carrying ``value``'s values (row stride doubled)."""
    rows, columns = value.shape
    base = torch.zeros(rows, 2 * columns, dtype=value.dtype, device=DEVICE)
    base[:, :columns] = value
    strided = torch.as_strided(base, (rows, columns), (2 * columns, 1))
    assert not strided.is_contiguous()
    return strided


def _reference(
    scores: torch.Tensor, ids: torch.Tensor, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights, ids.to(torch.int32)


def _operator_events(events: list[SelectionEvent]) -> list[tuple[str, str]]:
    return [
        (event.kernel_name, event.source)
        for event in events
        if (event.family, event.mode) == OPERATOR
    ]


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().view(torch.int32)


def test_torch_k8_sum_matches_the_pinned_reduction_tree():
    """``torch.sum(dim=-1, keepdim=True)`` over the pin rows equals the tree bit for
    bit; the sequential and adjacent-pairwise orders each differ on at least one
    row, so the agreement is not vacuous. The ``+ 1e-20`` add and the division are
    plain elementwise round-to-nearest operations (no fused kernel intervenes)."""
    rows = reduction_order_fixture().to(DEVICE)
    torch_sum = rows.sum(dim=-1, keepdim=True)
    tree = k8_reduction_tree(rows)
    assert torch.equal(_bits(torch_sum), _bits(tree)), (
        f"torch {torch.__version__}: CUDA K8 sum differs from the pinned tree on rows "
        f"{(_bits(torch_sum) != _bits(tree)).flatten().nonzero().flatten().tolist()}"
    )
    assert reduction_order_mismatch(DEVICE) == ""
    v = rows.unbind(-1)
    sequential = v[0]
    for value in v[1:]:
        sequential = sequential + value
    pairwise = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))
    assert not torch.equal(_bits(sequential.unsqueeze(-1)), _bits(torch_sum))
    assert not torch.equal(_bits(pairwise.unsqueeze(-1)), _bits(torch_sum))
    # The epsilon add and the division are the same single FP32 operations the
    # kernel spells: elementwise torch on the device equals them bit for bit.
    denominator = torch_sum + EPSILON
    expected_denominator = torch_sum + torch.tensor(EPSILON, device=DEVICE)
    assert torch.equal(_bits(denominator), _bits(expected_denominator))
    finite = torch.isfinite(denominator).flatten()
    weights = rows[finite] / denominator[finite]
    expected = torch.div(rows[finite], denominator[finite])
    assert torch.equal(_bits(weights), _bits(expected))


@pytest.mark.parametrize("renormalize", RENORMALIZE_VALUES)
@pytest.mark.parametrize("rows", ROWS)
def test_epilogue_equals_the_torch_statements_byte_for_byte(
    rows, renormalize, selection_events
):
    scores, ids = _route_inputs(rows, SEED + rows)
    before = (_bits(scores).clone(), ids.clone())
    input_ptrs = {scores.untyped_storage().data_ptr(), ids.untyped_storage().data_ptr()}
    with kernel_override(*OPERATOR, TORCH_ROUTE_EPILOGUE):
        on_weights, on_ids = moe_route_epilogue(
            scores, ids, renormalize=renormalize, solution=None, override=None
        )
    with kernel_override(*OPERATOR, TRITON_ROUTE_EPILOGUE):
        off_weights, off_ids = moe_route_epilogue(
            scores, ids, renormalize=renormalize, solution=None, override=None
        )
    torch.cuda.synchronize()
    assert _operator_events(selection_events) == [
        (TORCH_ROUTE_EPILOGUE, "override"),
        (TRITON_ROUTE_EPILOGUE, "override"),
    ]
    for weights, out_ids in ((on_weights, on_ids), (off_weights, off_ids)):
        assert weights.dtype is torch.float32 and weights.shape == (rows, TOPK)
        assert out_ids.dtype is torch.int32 and out_ids.shape == (rows, TOPK)
        for out in (weights, out_ids):
            assert out.is_contiguous() and out.device == scores.device
            assert out.untyped_storage().data_ptr() not in input_ptrs
    assert torch.equal(_bits(off_weights), _bits(on_weights))
    assert torch.equal(off_ids, on_ids)
    expected_weights, expected_ids = _reference(scores, ids, renormalize)
    assert torch.equal(_bits(on_weights), _bits(expected_weights))
    assert torch.equal(on_ids, expected_ids) and torch.equal(
        on_ids, ids.to(torch.int32)
    )
    assert torch.equal(_bits(scores), before[0]) and torch.equal(ids, before[1])
    if renormalize:
        # Row 0 is a realistic sigmoid row at every row count (the special words live
        # in SPECIAL_ROW): its renormalized weights are finite, so the byte equality
        # above compares a real K8 sum even at rows=1.
        assert torch.isfinite(off_weights[0]).all()
        if rows > TIE_ROW:
            assert torch.all(off_weights[TIE_ROW] == 0.125)
        if rows > NEG_INF_ROW:
            # The -inf slot makes the sum and the denominator -inf: finite slots give
            # -0.0 and the -inf slot NaN, canonical in both arms (bytes compared above).
            finite = [slot for slot in range(TOPK) if slot != NEG_INF_SLOT]
            assert torch.all(
                _bits(off_weights)[NEG_INF_ROW, finite] == NEGATIVE_ZERO_BITS
            )
            assert torch.isnan(off_weights[NEG_INF_ROW, NEG_INF_SLOT])
            assert torch.isnan(on_weights[NEG_INF_ROW, NEG_INF_SLOT])
        if rows > SPECIAL_ROW:
            # The NaN payload poisons the whole row's sum.
            assert torch.isnan(off_weights[SPECIAL_ROW]).all()
    else:
        assert torch.equal(_bits(off_weights), _bits(scores.gather(-1, ids)))
        if rows > SPECIAL_ROW:
            special = _bits(off_weights)[SPECIAL_ROW]
            assert special[NEGATIVE_ZERO_SLOT].item() == NEGATIVE_ZERO_BITS
            assert special[NAN_PAYLOAD_SLOT].item() == NAN_PAYLOAD_BITS


@pytest.mark.parametrize("rows", ROWS)
def test_ranked_selection_on_the_device_picks_torch(rows, selection_events):
    scores, ids = _route_inputs(rows, SEED)
    selected = select_kernel(
        *OPERATOR, SIGNATURE, traits=route_epilogue_traits(scores, ids, True)
    )
    assert selected.name == TORCH_ROUTE_EPILOGUE
    weights, out_ids = moe_route_epilogue(
        scores, ids, renormalize=True, solution=None, override=None
    )
    expected_weights, expected_ids = _reference(scores, ids, True)
    assert torch.equal(_bits(weights), _bits(expected_weights))
    assert torch.equal(out_ids, expected_ids)
    assert {name for name, _ in _operator_events(selection_events)} == {
        TORCH_ROUTE_EPILOGUE
    }
    assert {source for _, source in _operator_events(selection_events)} <= {
        "ranked",
        "cache",
    }


def test_rejected_shapes_raise_under_the_by_name_override():
    with kernel_override(*OPERATOR, TRITON_ROUTE_EPILOGUE):
        # Zero rows: no spec trait filters it (ranked selection would land on the
        # torch kernel), the override forces the Triton kernel and its host guard
        # refuses the empty grid -- there is no fallback to the torch statements.
        empty_scores, empty_ids = _route_inputs(0, SEED)
        assert empty_scores.shape == (0, EXPERTS) and empty_ids.shape == (0, TOPK)
        with pytest.raises(ValueError, match="rows=0") as info:
            moe_route_epilogue(
                empty_scores, empty_ids, renormalize=True, solution=None, override=None
            )
        assert str(info.value).startswith(f"{TRITON_ROUTE_EPILOGUE} cannot serve")
        scores, ids = _route_inputs(4, SEED)
        with pytest.raises(ValueError, match="scores must be contiguous"):
            moe_route_epilogue(
                _strided(scores), ids, renormalize=True, solution=None, override=None
            )
        with pytest.raises(ValueError, match="ids must be contiguous"):
            moe_route_epilogue(
                scores, _strided(ids), renormalize=True, solution=None, override=None
            )
        # A BF16 ``scores`` resolves by name through ``override=`` (no signature
        # filter on that path); the host guard rejects it.
        with pytest.raises(ValueError, match="scores must be torch.float32"):
            moe_route_epilogue(
                scores.bfloat16(),
                ids,
                renormalize=True,
                solution=None,
                override=TRITON_ROUTE_EPILOGUE,
            )
        with pytest.raises(ValueError, match="ids must be torch.int64"):
            moe_route_epilogue(
                scores,
                ids.to(torch.int32),
                renormalize=True,
                solution=None,
                override=None,
            )
        with pytest.raises(ValueError, match="ids must have shape"):
            moe_route_epilogue(
                scores,
                ids[:, :4].contiguous(),
                renormalize=True,
                solution=None,
                override=None,
            )
        with pytest.raises(ValueError, match="scores must have shape"):
            moe_route_epilogue(
                scores[:, :128].contiguous(),
                ids % 128,
                renormalize=True,
                solution=None,
                override=None,
            )
        # The facade refuses a non-bool before selection; the host guard would too.
        with pytest.raises(ValueError, match="renormalize must be a bool"):
            moe_route_epilogue(scores, ids, renormalize=1, solution=None, override=None)


def test_explain_selection_reports_the_override_and_the_trait_mismatch():
    scores, ids = _route_inputs(2, SEED)
    # Two rows with the example shape: both kernels are candidates, the override marks
    # the Triton kernel as selected.
    with kernel_override(*OPERATOR, TRITON_ROUTE_EPILOGUE):
        admitted = explain_selection(
            *OPERATOR, SIGNATURE, traits=route_epilogue_traits(scores, ids, True)
        )
    assert f"Override: {TRITON_ROUTE_EPILOGUE} (kernel_override())" in admitted
    assert "Candidates (2 matched, 2 registered)" in admitted
    assert f"{TRITON_ROUTE_EPILOGUE}  [SELECTED (override)]" in admitted
    # A non-contiguous ``scores`` fails the ``contiguous`` trait: the Triton kernel is
    # filtered out and only the bypass note names it.
    traits = route_epilogue_traits(_strided(scores), ids, True)
    assert traits["contiguous"] is False and traits["rows"] == 2
    with kernel_override(*OPERATOR, TRITON_ROUTE_EPILOGUE):
        text = explain_selection(*OPERATOR, SIGNATURE, traits=traits)
    assert f"Override: {TRITON_ROUTE_EPILOGUE} (kernel_override())" in text
    assert "Candidates (1 matched, 2 registered)" in text
    assert (
        f"Override selects {TRITON_ROUTE_EPILOGUE}, which is not among the matched "
        "candidates" in text
    )
