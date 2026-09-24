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

"""Expert routing tables of the Triton BF16 MoE apply: the ``moe.expert_routing`` operator.

``_moe`` (``ops/moe/triton/bf16.py``) prepares its stage launches from the
top-k routing in one step: it builds the per-expert route tables (``[E, R]``
route ids and ``[E]`` counts) the stage kernels traverse. That step is the
operator ``("moe", "expert_routing")``: :func:`moe_expert_routing` takes the
top-k ids and weights, the expert count and the :class:`StagePolicy` the stage
kernels run under, and returns an :class:`ExpertRouting` -- the tables, the
route weights ``_combine`` consumes and the ``compact`` flag ``_moe`` forwards
to both stage kernels as their ``COMPACT_EXPERTS`` constexpr. The policy's
``sort_routes`` asks for each token's expert ids to be sorted ascending with
the route weights gathered along, so the combine accumulates routes in
ascending expert order; ``_moe`` passes ``sort_routes=False`` (its existing
behaviour: the ids are used in top-k order). Three kernels are registered in
:mod:`tokenspeed_kernel.ops.moe.triton.expert_routing`:

* ``triton_moe_routing_full`` (solution ``triton``, PORTABLE, any vendor): the
  optional sort + gather and the canonical one-program-per-expert table kernel
  verbatim; every expert is visited by the stage kernels (``compact=False``).
  The ranked default and the production path.
* ``triton_moe_routing_compact`` (solution ``triton``, REFERENCE, NVIDIA): the
  same optional sort + gather, then a one-CTA packing kernel that appends the
  ascending list of active experts to the counts so the stage kernels visit
  only those (``compact=True``). Never auto-selected; reached only by name
  through the ordinary registry override (``override=``, ``kernel_override("moe",
  "expert_routing", ...)``, ``TOKENSPEED_KERNEL_OVERRIDE_MOE_EXPERT_ROUTING``,
  or ``--kernel-override moe.expert_routing=triton_moe_routing_compact``). It
  admits 256 experts under BF16 SiLU (sorted or unsorted routes) on CUDA
  tensors and raises on any other call, an empty route set included: no
  in-function fallback to another registered kernel. Its one documented shape
  rule is on the return value: for more than ``MAX_COMPACT_ROUTES`` (32) routes
  it emits the full tables and ``compact=False``, so one process-wide override
  runs decode and prefill alike.
* ``triton_moe_routing_compact_ordered`` (solution ``triton``, REFERENCE,
  NVIDIA, int32 ids only): the compact packing kernel with the per-token sort
  and the paired FP32 weight gather moved into the CTA (a 32-lane bitonic
  network replaying the Torch dispatch for ``[N, 8]`` int32 sorts, tie
  permutation included). Reached only by name (``--kernel-override
  moe.expert_routing=triton_moe_routing_compact_ordered``); one override names
  one kernel, so it replaces the compact override rather than stacking on it.
  Admits the compact envelope plus ``sort_routes=True`` (the in-CTA sort is
  the sorted prep; an unsorted caller has nothing to fuse), contiguous int32
  ids, contiguous FP32 weights and top-k 8. Because ``_moe`` passes
  ``sort_routes=False``, this kernel is reachable through the facade (tests,
  callers that sort) and is refused, without fallback, when the override is
  applied to the default MoE path. Shape rule on the return value: 8 or 32
  routes -> the in-kernel sort, ``compact=True, ordered=True``; 16 or 24
  routes -> the torch sort and the compact tables, ``ordered=False``; more
  than 32 -> the full tables, ``compact=False``; an empty route set raises.

The three kernels agree byte for byte on the tables where defined (slots below
the per-expert count) and on ``route_weights`` (tie rows included), which
``test/nvidia/ops/moe/test_expert_routing_cuda.py`` and
``test_expert_routing_ordered_cuda.py`` prove.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "COMPACT_NUM_EXPERTS",
    "COMPACT_STAGE_POLICIES",
    "EXPERT_ROUTING_FAMILY",
    "EXPERT_ROUTING_MODE",
    "EXPERT_ROUTING_OVERRIDE_ENV",
    "EXPERT_ROUTING_SIGNATURES",
    "MAX_COMPACT_ROUTES",
    "ORDERED_ROUTES",
    "ORDERED_STAGE_POLICY",
    "ORDERED_TOP_K",
    "TRITON_MOE_ROUTING_COMPACT",
    "TRITON_MOE_ROUTING_COMPACT_ORDERED",
    "TRITON_MOE_ROUTING_FULL",
    "ExpertRouting",
    "StagePolicy",
    "expert_routing_traits",
    "moe_expert_routing",
]

EXPERT_ROUTING_FAMILY = "moe"
EXPERT_ROUTING_MODE = "expert_routing"
# Same derivation as ``select_kernel``'s environment override key.
EXPERT_ROUTING_OVERRIDE_ENV = (
    f"TOKENSPEED_KERNEL_OVERRIDE_{EXPERT_ROUTING_FAMILY.upper()}"
    f"_{EXPERT_ROUTING_MODE.upper()}"
)
# Registered names: the production default, the compact packing (REFERENCE) and
# the compact packing with the in-CTA sort (REFERENCE); the last two are the
# opt-in alternatives.
TRITON_MOE_ROUTING_FULL = "triton_moe_routing_full"
TRITON_MOE_ROUTING_COMPACT = "triton_moe_routing_compact"
TRITON_MOE_ROUTING_COMPACT_ORDERED = "triton_moe_routing_compact_ordered"
# The compact kernel's shape rule: at most this many routes are packed; above it
# the kernel emits the full tables with ``compact=False`` (module doc).
MAX_COMPACT_ROUTES = 32
# The ordered kernel's shape rule: at these route counts (rows 1 and 4 at top-8)
# the in-CTA sort network runs and ``ordered=True``; at every other admitted count
# the torch sort + gather run and the compact kernel's rule applies (module doc).
ORDERED_ROUTES = frozenset({8, 32})
# The sort network addresses ``row * 8 + lane``: eight ids per token, structural.
ORDERED_TOP_K = 8
# The compact kernel's admitted envelope, declared as spec traits and re-stated
# by its host guard: 256 experts under BF16 SiLU, with or without the per-token
# sort. This is the envelope the stage byte-equality test covers; widen it after
# running that test at other dtypes / activations / expert counts.
COMPACT_NUM_EXPERTS = 256
COMPACT_STAGE_POLICIES = frozenset({"bf16_silu_unsorted", "bf16_silu_sorted"})
# The ordered kernel's envelope: the compact envelope restricted to the sorted
# prep (the in-CTA sort is that prep; an unsorted caller has nothing to fuse).
ORDERED_STAGE_POLICY = "bf16_silu_sorted"
_DTYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16"}
# The activations ``_validate`` admits (ops/moe/triton/bf16.py).
_ACTIVATIONS = frozenset({"silu", "situ", "swiglu"})
# The two id dtypes ``_validate`` admits; both kernels declare both, so a
# by-name switch never changes the facade's filtering.
EXPERT_ROUTING_SIGNATURES = frozenset(
    {
        format_signature(topk_ids=dense_tensor_format(torch.int32)),
        format_signature(topk_ids=dense_tensor_format(torch.int64)),
    }
)


@dataclass(frozen=True)
class StagePolicy:
    """The stage-kernel policy the routing kernels are qualified for: the values
    ``_moe`` receives (``x.dtype``, ``activation``) and whether the routes are
    sorted before the tables are built.

    ``sort_routes`` asks the kernel to sort each token's expert ids ascending
    and gather the route weights along, so the combine accumulates routes in
    ascending expert order; ``False`` keeps the top-k order (what ``_moe``
    passes). ``name`` is the ``stage_policy`` trait value, for example
    ``"bf16_silu_unsorted"`` or ``"fp16_situ_sorted"``.
    """

    input_dtype: torch.dtype
    activation: str
    sort_routes: bool

    def __post_init__(self) -> None:
        if self.input_dtype not in _DTYPE_NAMES:
            raise ValueError(
                f"StagePolicy.input_dtype must be one of {sorted(map(str, _DTYPE_NAMES))}, "
                f"got {self.input_dtype}"
            )
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"StagePolicy.activation must be one of {sorted(_ACTIVATIONS)}, "
                f"got {self.activation!r}"
            )
        if not isinstance(self.sort_routes, bool):
            raise TypeError(
                "StagePolicy.sort_routes must be a bool, got "
                f"{type(self.sort_routes).__name__}"
            )

    @property
    def name(self) -> str:
        """The trait value: ``<dtype>_<activation>_<sorted|unsorted>``."""
        ordering = "sorted" if self.sort_routes else "unsorted"
        return f"{_DTYPE_NAMES[self.input_dtype]}_{self.activation}_{ordering}"


@dataclass(frozen=True)
class ExpertRouting:
    """What a ``moe.expert_routing`` kernel returns to ``_moe``.

    ``route_ids`` is int32 ``[E, R]``; slots at or beyond ``counts[e]`` are
    unwritten. ``counts`` is int32 ``[E]`` for the full tables or ``[E + min(E,
    R) + 1]`` for the compact tables (tail layout ``[E counts | min(E, R)
    ascending active ids | active count]``). ``route_weights`` is the top-k
    weights gathered with the sort order under ``sort_routes``, or the input
    tensor itself otherwise. ``compact`` is the stage kernels'
    ``COMPACT_EXPERTS`` constexpr; ``ordered`` is ``True`` when the tables and
    ``route_weights`` came from the in-CTA sort of
    ``triton_moe_routing_compact_ordered`` (8 or 32 routes at top-8) and
    ``False`` when the torch sort ran or no sort was asked for (every other
    kernel and route count).
    """

    route_ids: torch.Tensor
    counts: torch.Tensor
    route_weights: torch.Tensor
    compact: bool
    ordered: bool


def expert_routing_traits(
    topk_ids: torch.Tensor, num_experts: int, stage_policy: StagePolicy
) -> dict[str, int | str]:
    """The trait dict :func:`moe_expert_routing` selects with (metadata only).

    ``num_experts`` and ``stage_policy`` (its ``name``) are the values the
    ``triton_moe_routing_compact`` and ``triton_moe_routing_compact_ordered``
    specs constrain; ``num_routes`` (``topk_ids.numel()``) is carried for the
    shape capture and the kernel scope (no registered spec constrains it: the
    compact kernels' route rules are their documented return-value rules, and
    ``spec_matches_traits`` ignores traits a spec does not declare);
    ``triton_moe_routing_full`` declares no traits and admits every dict. No
    ``contiguous`` trait: the kernels make the int32 ids contiguous themselves.
    """
    return {
        "num_experts": int(num_experts),
        "num_routes": int(topk_ids.numel()),
        "stage_policy": stage_policy.name,
    }


def _operator_spec(kernel_name: str) -> KernelSpec:
    """The registry spec of ``kernel_name`` if it belongs to ``moe.expert_routing``.

    Raises:
        ValueError: ``kernel_name`` is unregistered or registered under another
            operator. ``select_kernel`` resolves an override by name alone, so
            the operator check is made here before the kernel is called with
            the routing tensors.
    """
    spec = KernelRegistry.get().get_by_name(kernel_name)
    if spec is None or (spec.family, spec.mode) != (
        EXPERT_ROUTING_FAMILY,
        EXPERT_ROUTING_MODE,
    ):
        raise ValueError(
            f"{kernel_name!r} is not registered under "
            f"{EXPERT_ROUTING_FAMILY}.{EXPERT_ROUTING_MODE}"
        )
    return spec


def moe_expert_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    stage_policy: StagePolicy,
    *,
    solution: str | None,
    override: str | None,
) -> ExpertRouting:
    """Build the expert route tables of the Triton MoE apply through ``moe.expert_routing``.

    Args:
        topk_ids: ``[tokens, top_k]`` int32 or int64 expert ids (the dtypes
            ``_validate`` admits; the kernels cast to int32). Out-of-range ids
            are absent from the tables.
        topk_weights: ``[tokens, top_k]`` floating-point route weights.
        num_experts: Number of experts ``E`` (the tables' first dimension).
        stage_policy: The :class:`StagePolicy` the stage kernels run under; its
            ``sort_routes`` decides whether the ids are sorted per token and
            the weights gathered along, its ``name`` is a trait.
        solution: Restrict ranked selection to one solution (``"triton"``) or
            ``None`` for any.
        override: Exact kernel name to force, or ``None`` for ranked selection;
            the ``kernel_override`` context and
            ``TOKENSPEED_KERNEL_OVERRIDE_MOE_EXPERT_ROUTING`` outrank it (see
            ``select_kernel``).

    Returns:
        The :class:`ExpertRouting` the selected kernel produced: fresh tables,
        the route weights ``_combine`` consumes (a fresh gathered tensor under
        ``sort_routes``, the input itself otherwise), ``compact`` and
        ``ordered``; the inputs are not modified.

    Raises:
        TypeError: ``stage_policy`` is not a :class:`StagePolicy`, or the
            kernel returned something other than an :class:`ExpertRouting`.
        ValueError: The tensors do not form a top-k routing, ``num_experts`` is
            not positive, the tensors are on different devices, or the resolved
            kernel is not registered under ``moe.expert_routing``. The compact
            and the compact-ordered kernels raise their own ``ValueError`` on
            a call outside their envelopes (no fallback).
        NoKernelFoundError: No registered kernel serves the ``topk_ids`` dtype.
    """
    if not isinstance(stage_policy, StagePolicy):
        raise TypeError(
            f"stage_policy must be a StagePolicy, got {type(stage_policy).__name__}"
        )
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k]")
    if num_experts < 1:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if topk_ids.device != topk_weights.device:
        raise ValueError("topk_ids and topk_weights must share one device")
    # An unregistered id dtype makes ``select_kernel`` raise ``NoKernelFoundError``
    # naming the signature.
    signature = format_signature(topk_ids=dense_tensor_format(topk_ids.dtype))
    traits = expert_routing_traits(topk_ids, num_experts, stage_policy)
    kernel = select_kernel(
        EXPERT_ROUTING_FAMILY,
        EXPERT_ROUTING_MODE,
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    _operator_spec(kernel.name)
    with kernel_scope(
        EXPERT_ROUTING_FAMILY,
        EXPERT_ROUTING_MODE,
        topk_ids.dtype,
        kernel_name=kernel.name,
        num_experts=traits["num_experts"],
        num_routes=traits["num_routes"],
    ):
        routing = kernel(topk_ids, topk_weights, num_experts, stage_policy)
    if not isinstance(routing, ExpertRouting):
        raise TypeError(
            f"{kernel.name} returned {type(routing).__name__}, not an ExpertRouting"
        )
    # Recorded after the call so the kernel's flags are captured per call site.
    ShapeCapture.get().record(
        EXPERT_ROUTING_FAMILY,
        EXPERT_ROUTING_MODE,
        kernel.name,
        topk_ids.dtype,
        {
            "num_experts": traits["num_experts"],
            "num_routes": traits["num_routes"],
            "stage_policy": traits["stage_policy"],
            "compact": routing.compact,
            "ordered": routing.ordered,
        },
    )
    return routing
