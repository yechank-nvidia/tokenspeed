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

"""``triton_group_mask_bits`` on an NVIDIA device: byte equality with ``torch_group_mask``
under the by-name switch, the ranked default, rejected shapes and ``explain_selection``.

The in-tree kernel compiles to its own Triton hash (not a previously measured
binary's), so its correctness is established here anew: for every row count in
``ROWS`` (the row axis 1..4096 plus the non-powers of two 3, 5, 7 and 33; every
positive row count is admitted, see the Triton module doc) the facade is run once
per arm inside
``kernel_override`` with a selection listener recording the witnesses, and the two
outputs must agree bit for bit -- NaN payloads, ``-0.0`` and infinities included --
while the downstream ``topk`` of the route sees identical ids. The only row count
the kernel refuses is zero (an empty grid), by its host guard.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe.group_mask import (
    GROUP_MASK_FAMILY,
    GROUP_MASK_MODE,
    GROUP_MASK_SIGNATURES,
    TORCH_GROUP_MASK,
    TRITON_GROUP_MASK_BITS,
    group_mask_traits,
    moe_group_mask,
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

OPERATOR = (GROUP_MASK_FAMILY, GROUP_MASK_MODE)
SIGNATURE = next(iter(GROUP_MASK_SIGNATURES))
DEVICE = torch.device("cuda", 0)
SEED = 20260921
# Row counts under test: the row axis (1..4096, decode through prefill) plus
# non-powers of two; every one must be admitted and byte-identical to the torch arm.
ROWS = (1, 2, 3, 4, 5, 7, 8, 16, 32, 33, 64, 128, 256, 1024, 4096)
# Grouped-routing example shape (256 experts in 8 groups, 4 groups kept, top-8).
EXPERTS, NUM_GROUPS, TOPK_GROUPS, TOPK = 256, 8, 4, 8
GROUP_SIZE = EXPERTS // NUM_GROUPS
LOGIT_STD, BIAS_STD = 0.6, 0.1
TIE_EXPERTS = (0, 1, 2, 32, 33, 64, 65, 96, 97)
NEGATIVE_INFINITY_BITS = torch.tensor([float("-inf")]).view(torch.int32).item()
NAN_PAYLOAD_BITS = 0x7FC01234


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


def _route_intermediates(rows: int, seed: int):
    """The grouped-routing intermediates on seeded inputs on the device, then special words
    injected into ``choices`` after the group selection: ``-inf``/``+inf``/``-0.0``/a
    NaN payload in a kept group, ``+inf``/``-inf``/a NaN payload in an excluded group.

    Row 1 (when ``rows > 1``) is the tie row: nine experts across four groups share
    one choice. Row 2 (when ``rows > 2``) is the all-negative row: every sigmoid is
    about 2e-22, the ``+1e-20`` path of the route. Every other row is drawn from the
    synthetic distribution. Cost is linear in ``rows`` (one batched sigmoid, two batched
    ``topk``, the special words on row 0 only); ``rows == 0`` yields empty tensors
    (nothing to inject) for the rejected-shape test.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, EXPERTS, generator=generator) * LOGIT_STD
    bias = torch.randn(EXPERTS, generator=generator) * BIAS_STD
    bias[list(TIE_EXPERTS)] = BIAS_STD
    if rows > 1:
        logits[1] = -3.0
        logits[1, list(TIE_EXPERTS)] = 6.0
    if rows > 2:
        logits[2] = -50.0
    logits, bias = logits.to(DEVICE), bias.to(DEVICE)
    scores = logits.float().sigmoid()
    choices = scores + bias
    grouped = choices.reshape(rows, NUM_GROUPS, GROUP_SIZE)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    group_ids = group_scores.topk(TOPK_GROUPS, dim=-1, sorted=True).indices
    if rows > 0:
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
    return choices, group_ids, group_scores, grouped


def _strided_choices(choices: torch.Tensor) -> torch.Tensor:
    """A non-contiguous view carrying ``choices``' values (row stride doubled)."""
    rows = choices.shape[0]
    base = torch.zeros(rows, 2 * EXPERTS, device=DEVICE)
    base[:, :EXPERTS] = choices
    strided = torch.as_strided(base, (rows, EXPERTS), (2 * EXPERTS, 1))
    assert not strided.is_contiguous()
    return strided


def _reference(choices, group_ids, group_scores, grouped) -> torch.Tensor:
    keep_groups = torch.zeros_like(group_scores, dtype=torch.bool)
    keep_groups.scatter_(1, group_ids, True)
    keep = keep_groups.unsqueeze(-1).expand_as(grouped).reshape_as(choices)
    return choices.masked_fill(~keep, float("-inf"))


def _words(value: torch.Tensor) -> torch.Tensor:
    return value.view(torch.int32) if value.dtype is torch.float32 else value


def _operator_events(events: list[SelectionEvent]) -> list[tuple[str, str]]:
    return [
        (event.kernel_name, event.source)
        for event in events
        if (event.family, event.mode) == OPERATOR
    ]


@pytest.mark.parametrize("rows", ROWS)
def test_bit_select_equals_the_torch_statements_byte_for_byte(rows, selection_events):
    tensors = _route_intermediates(rows, SEED + rows)
    before = [value.clone() for value in tensors]
    input_ptrs = {value.untyped_storage().data_ptr() for value in tensors}
    with kernel_override(*OPERATOR, TORCH_GROUP_MASK):
        on = moe_group_mask(*tensors, solution=None, override=None)
    with kernel_override(*OPERATOR, TRITON_GROUP_MASK_BITS):
        off = moe_group_mask(*tensors, solution=None, override=None)
    torch.cuda.synchronize()
    assert _operator_events(selection_events) == [
        (TORCH_GROUP_MASK, "override"),
        (TRITON_GROUP_MASK_BITS, "override"),
    ]
    for out in (on, off):
        assert out.dtype is torch.float32 and out.shape == (rows, EXPERTS)
        assert out.is_contiguous() and out.device == tensors[0].device
        assert out.untyped_storage().data_ptr() not in input_ptrs
    assert torch.equal(off.view(torch.int32), on.view(torch.int32))
    assert torch.equal(on.view(torch.int32), _reference(*tensors).view(torch.int32))
    for value, saved in zip(tensors, before, strict=True):
        assert torch.equal(_words(value), _words(saved))
    # The downstream statement of the route sees identical ids from both arms.
    assert torch.equal(
        off.topk(TOPK, dim=-1, sorted=True).indices,
        on.topk(TOPK, dim=-1, sorted=True).indices,
    )
    # Every row carries at least the 128 excluded words (four groups of 32) as the
    # -inf pattern; kept words may add to the count (row 0 has one -inf kept word).
    excluded_words = off.view(torch.int32) == NEGATIVE_INFINITY_BITS
    assert torch.all(
        excluded_words.sum(dim=-1) >= (NUM_GROUPS - TOPK_GROUPS) * GROUP_SIZE
    )
    choices, group_ids = tensors[0], tensors[1]
    kept = int(group_ids[0, 0])
    excluded = next(g for g in range(NUM_GROUPS) if g not in group_ids[0].tolist())
    kept_slice = slice(kept * GROUP_SIZE, (kept + 1) * GROUP_SIZE)
    excluded_slice = slice(excluded * GROUP_SIZE, (excluded + 1) * GROUP_SIZE)
    assert torch.equal(
        off.view(torch.int32)[0, kept_slice], choices.view(torch.int32)[0, kept_slice]
    )
    assert torch.all(off.view(torch.int32)[0, excluded_slice] == NEGATIVE_INFINITY_BITS)
    assert torch.isnan(off[0, kept * GROUP_SIZE + 3])


@pytest.mark.parametrize("rows", ROWS)
def test_ranked_selection_on_the_device_picks_torch(rows, selection_events):
    tensors = _route_intermediates(rows, SEED)
    selected = select_kernel(*OPERATOR, SIGNATURE, traits=group_mask_traits(*tensors))
    assert selected.name == TORCH_GROUP_MASK
    out = moe_group_mask(*tensors, solution=None, override=None)
    assert torch.equal(out.view(torch.int32), _reference(*tensors).view(torch.int32))
    assert {name for name, _ in _operator_events(selection_events)} == {
        TORCH_GROUP_MASK
    }
    assert {source for _, source in _operator_events(selection_events)} <= {
        "ranked",
        "cache",
    }


def test_rejected_shapes_raise_under_the_by_name_override():
    with kernel_override(*OPERATOR, TRITON_GROUP_MASK_BITS):
        # Zero rows: no spec trait filters it (ranked selection would land on the
        # torch kernel), the override forces the Triton kernel and its host guard
        # refuses the empty grid -- there is no fallback to the torch statements.
        empty = _route_intermediates(0, SEED)
        assert empty[0].shape == (0, EXPERTS) and empty[1].shape == (0, TOPK_GROUPS)
        with pytest.raises(ValueError, match="rows=0") as info:
            moe_group_mask(*empty, solution=None, override=None)
        assert str(info.value).startswith(f"{TRITON_GROUP_MASK_BITS} cannot serve")
        choices, group_ids, group_scores, grouped = _route_intermediates(4, SEED)
        strided = _strided_choices(choices)
        with pytest.raises(ValueError, match="choices must be contiguous"):
            moe_group_mask(
                strided,
                group_ids,
                group_scores,
                strided.reshape(4, NUM_GROUPS, GROUP_SIZE),
                solution=None,
                override=None,
            )
        # A BF16 ``choices`` resolves by name through ``override=`` (no signature
        # filter on that path); the host guard rejects it.
        with pytest.raises(ValueError, match="choices must be torch.float32"):
            moe_group_mask(
                choices.bfloat16(),
                group_ids,
                group_scores,
                grouped.bfloat16(),
                solution=None,
                override=TRITON_GROUP_MASK_BITS,
            )
        with pytest.raises(ValueError, match="group_ids must be torch.int64"):
            moe_group_mask(
                choices,
                group_ids.to(torch.int32),
                group_scores,
                grouped,
                solution=None,
                override=None,
            )


def test_explain_selection_reports_the_override_and_the_trait_mismatch():
    choices, group_ids, group_scores, grouped = _route_intermediates(2, SEED)
    # Two rows with the example shape: both kernels are candidates, the override marks
    # the Triton kernel as selected.
    with kernel_override(*OPERATOR, TRITON_GROUP_MASK_BITS):
        admitted = explain_selection(
            *OPERATOR,
            SIGNATURE,
            traits=group_mask_traits(choices, group_ids, group_scores, grouped),
        )
    assert f"Override: {TRITON_GROUP_MASK_BITS} (kernel_override())" in admitted
    assert "Candidates (2 matched, 2 registered)" in admitted
    assert f"{TRITON_GROUP_MASK_BITS}  [SELECTED (override)]" in admitted
    # A non-contiguous ``choices`` fails the ``contiguous`` trait: the Triton kernel is
    # filtered out and only the bypass note names it.
    strided = _strided_choices(choices)
    traits = group_mask_traits(
        strided, group_ids, group_scores, strided.reshape(2, NUM_GROUPS, GROUP_SIZE)
    )
    assert traits["contiguous"] is False and traits["rows"] == 2
    with kernel_override(*OPERATOR, TRITON_GROUP_MASK_BITS):
        text = explain_selection(*OPERATOR, SIGNATURE, traits=traits)
    assert f"Override: {TRITON_GROUP_MASK_BITS} (kernel_override())" in text
    assert "Candidates (1 matched, 2 registered)" in text
    assert (
        f"Override selects {TRITON_GROUP_MASK_BITS}, which is not among the matched "
        "candidates" in text
    )
