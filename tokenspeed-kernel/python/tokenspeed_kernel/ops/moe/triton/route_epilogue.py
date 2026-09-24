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

"""Triton row epilogue for ``moe.route_epilogue`` (``triton_route_epilogue``).

One program per row gathers the row's eight selected FP32 scores and, under
``NORMALIZE``, divides them by their sum plus ``1e-20`` with the sum spelled in
the order Torch's CUDA reduction uses for a contiguous FP32 ``[rows, 8]``
``sum(dim=-1)``: ``((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))``, then ``+ 1e-20``,
then a correctly rounded division -- every step one round-to-nearest FP32
operation (``add.rn.f32`` by inline PTX, ``tl.div_rn``). The int64 ids are stored
as int32. Without ``NORMALIZE`` no floating-point arithmetic runs. The output
equals the three torch statements of the routing tail --
``torch_route_epilogue`` (:mod:`tokenspeed_kernel.ops.moe.route_epilogue`) --
byte for byte on a Torch build whose CUDA K8 sum follows that tree.

The Torch side of the contract (torch ``2.13.0+cu130``,
``ATen/native/cuda/Reduce.cuh``): the sum over the fastest-striding dimension
of 8 elements takes no input vectorization (that needs ``dim0 >= 128``), gets
``block.x = 8`` lanes per output row with one element per lane, folds the four
per-lane accumulators as ``((v + 0) + 0) + 0`` (exact), and combines the lanes
with ``for (offset = dim_x >> 1; offset > 0; offset >>= 1) value =
combine(value, shfl_down(value, offset))`` -- offsets 4, 2, 1 -- so lane 0
holds ``((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))``. ``+ 1e-20`` is a separate
elementwise add and ``/`` is ``DivFunctor``'s ``a / b`` (``div.rn.f32``,
non-FTZ). Nothing in that derivation depends on the row count (``block.y``
only indexes outputs), which is why every positive row count is admitted from
the start.

Pin rule. A Torch change to the shuffle-down order, the
vectorization threshold, the block sizing, the accumulator count or an
FTZ/fast-division build flag voids the contract without touching this kernel.
The GPU test ``test/nvidia/ops/moe/test_route_epilogue_cuda.py`` therefore runs
:func:`reduction_order_mismatch` first and fails every byte-equality test on a
mismatch (never a skip); the CPU test ``test/ops/moe/test_route_epilogue.py``
pins the tree literally in this source by AST. Nothing probes the order at run
time: the route runs under CUDA-graph capture, where a device probe with a host
sync is illegal, and the kernel is opt-in by name.

The kernel body is the straightforward per-row gather and pinned-order sum;
correctness is established by the byte-equality tests, not by a binary hash.

Registered in the REFERENCE band, so ranked selection never picks it while
``torch_route_epilogue`` is registered; it is reached only by name
(``override=``, ``kernel_override("moe", "route_epilogue", ...)``,
``TOKENSPEED_KERNEL_OVERRIDE_MOE_ROUTE_EPILOGUE``). Its admission -- any
positive row count, contiguous FP32 ``[rows, 256]`` scores and int64
``[rows, 8]`` ids on one CUDA device, ``renormalize`` True or False -- is
declared twice on purpose: as spec traits, so ranked selection and callers that
re-check traits reject it, and as the host guard :func:`route_epilogue_rejection`, so
a by-name override on a rejected call raises instead of falling back. Zero
rows are rejected by the host guard alone (an empty grid is never launched;
``torch_route_epilogue`` serves that call).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "TRITON_ROUTE_EPILOGUE",
    "k8_reduction_tree",
    "reduction_order_fixture",
    "reduction_order_mismatch",
    "route_epilogue_rejection",
    "triton_route_epilogue",
]

# Registered name of the row epilogue kernel; the OFF target of the
# ``moe.route_epilogue`` switch.
TRITON_ROUTE_EPILOGUE = "triton_route_epilogue"
# Compile-time constants of the example shape: 256 experts, 8 selected slots
# (the reduction tree is 8-wide by construction). ``rows`` is the grid size only
# (no ``rows`` constant or trait: see the module doc).
_EXPERTS = 256
_TOPK = 8
# Host-side witness of the epsilon literal in the kernel (documentation and the
# CPU source pin only; the kernel carries the literal itself).
_EPSILON = 1e-20
# Launch options, made explicit. ``enable_reflect_ftz`` was the
# tokenspeed_triton default (True) at qualification; it only sets the NVVM
# reflect flag that libdevice reads, and this kernel links no libdevice
# (``tl.div_rn`` is an LLVM fdiv -> ``div.rn.f32``; ``_add_rn`` is inline PTX),
# so the compiled PTX carries no ``.ftz`` and pinning the flag to False, as the
# other opt-in MoE kernels do, leaves the PTX unchanged. Torch's division is
# non-FTZ, so the pin states the semantics we want.
_LAUNCH_OPTIONS = dict(
    num_warps=1,
    enable_fp_fusion=False,
    enable_reflect_ftz=False,
    launch_pdl=False,
)

# Fixture rows of the reduction-order pin (section "Pin rule" above): the
# rounding row (its sum's bits depend on the summation order), the subnormal
# row, an all-equal row, a row with ``-inf`` and seeded rows of log-uniform
# magnitudes ``2^U[-149, 0] * rand`` with random signs.
_PIN_ROUNDING_ROW = (
    1.0,
    2.0**-24,
    2.0**-25,
    2.0**-26,
    0.5,
    2.0**-23,
    2.0**-24,
    2.0**-25,
)
_PIN_SUBNORMAL_ROW = (
    2.0**-149,
    2.0**-140,
    2.0**-130,
    2.0**-126,
    0.0,
    2.0**-145,
    2.0**-135,
    2.0**-128,
)
_PIN_EQUAL_ROW = (0.25,) * _TOPK
_PIN_NEG_INF_ROW = (1.0, 2.0, float("-inf"), 4.0, 0.5, 8.0, 16.0, 32.0)
_PIN_SEED = 913
_PIN_RANDOM_ROWS = 128


@triton.jit
def _add_rn(left, right):
    """One correctly rounded FP32 add (``add.rn.f32``) that the compiler can
    neither reassociate nor fuse."""
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _route_epilogue(
    Scores, Ids, WeightsOut, IdsOut, EXPERTS: tl.constexpr, NORMALIZE: tl.constexpr
):
    """One row: gather the eight selected scores; under ``NORMALIZE`` divide them
    by the pinned Torch CUDA K8 sum plus ``1e-20``; store the FP32 weights and the
    ids as int32."""
    row = tl.program_id(0)
    slots = tl.arange(0, 8)
    ids = tl.load(Ids + row * 8 + slots)
    selected = tl.load(Scores + row * EXPERTS + ids)
    if NORMALIZE:
        # Pinned Torch CUDA Reduce.cuh: K8 contiguous input, block.x=8,
        # one input per lane, no vectorization, shuffle-down 4 -> 2 -> 1.
        values = ()
        for slot in tl.static_range(8):
            index = tl.load(Ids + row * 8 + slot)
            values += (tl.load(Scores + row * EXPERTS + index),)
        even = _add_rn(_add_rn(values[0], values[4]), _add_rn(values[2], values[6]))
        odd = _add_rn(_add_rn(values[1], values[5]), _add_rn(values[3], values[7]))
        denominator = _add_rn(_add_rn(even, odd), 1e-20)
        selected = tl.div_rn(selected, denominator)
    tl.store(WeightsOut + row * 8 + slots, selected)
    tl.store(IdsOut + row * 8 + slots, ids.to(tl.int32))


def route_epilogue_rejection(
    scores: torch.Tensor, ids: torch.Tensor, renormalize: bool
) -> str:
    """Why ``triton_route_epilogue`` cannot serve this call; empty when admitted.

    Metadata only (no device read): the row count must be positive (an empty
    grid is not launched); ``scores`` FP32 ``[rows, 256]`` and ``ids`` int64
    ``[rows, 8]``, each contiguous (the kernel's pointer arithmetic has no
    stride arguments); ``renormalize`` a ``bool`` (it selects the compiled
    variant and is never coerced); both CUDA tensors on one device. The reason
    names the first rejecting attribute; device clauses come last so that shape
    and layout findings are reported on any device.

    Args:
        scores: FP32 sigmoid scores the kernel gathers from.
        ids: Selected expert ids the kernel reads through an int64 pointer.
        renormalize: Whether the gathered scores are divided by their K8 sum.

    Returns:
        The rejection reason, or ``""`` when the kernel admits the call.
    """
    if scores.ndim != 2:
        return f"scores must be 2-D, got {scores.ndim}-D"
    rows = int(scores.shape[0])
    if rows == 0:
        return "rows=0: an empty grid is not launched (rows must be positive)"
    layouts = (
        ("scores", scores, torch.float32, (rows, _EXPERTS)),
        ("ids", ids, torch.int64, (rows, _TOPK)),
    )
    for name, value, dtype, shape in layouts:
        if value.dtype != dtype:
            return f"{name} must be {dtype}, got {value.dtype}"
        if tuple(value.shape) != shape:
            return f"{name} must have shape {shape}, got {tuple(value.shape)}"
        if not value.is_contiguous():
            return f"{name} must be contiguous, got strides {tuple(value.stride())}"
    if not isinstance(renormalize, bool):
        return f"renormalize must be a bool, got {type(renormalize).__name__}"
    for name, value, _, _ in layouts:
        if not value.is_cuda:
            return f"{name} must be a CUDA tensor, got device {value.device}"
    if scores.device != ids.device:
        return "scores and ids must share one device"
    return ""


@register_kernel(
    "moe",
    "route_epilogue",
    name=TRITON_ROUTE_EPILOGUE,
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    # The same FP32/int64 pair as ``torch_route_epilogue`` (ops/moe/route_epilogue.py),
    # so a name override between the two never changes the facade's filtering.
    signatures=frozenset(
        {
            format_signature(
                scores=dense_tensor_format(torch.float32),
                ids=dense_tensor_format(torch.int64),
            )
        }
    ),
    # No ``rows`` trait: every positive row count is admitted (module doc); the
    # facade still passes ``rows`` in its trait dict, which no spec constrains.
    # Both ``renormalize`` values are compiled variants of the one kernel.
    traits={
        "experts": frozenset({_EXPERTS}),
        "topk": frozenset({_TOPK}),
        "renormalize": frozenset({True, False}),
        "contiguous": frozenset({True}),
    },
    # REFERENCE band: never auto-selected while torch_route_epilogue is registered;
    # reached only by name (``override=``, ``kernel_override()``,
    # ``TOKENSPEED_KERNEL_OVERRIDE_MOE_ROUTE_EPILOGUE``).
    priority=Priority.REFERENCE,
)
def triton_route_epilogue(
    scores: torch.Tensor, ids: torch.Tensor, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the row epilogue on an admitted call; raise otherwise.

    Args:
        scores: ``[rows, 256]`` contiguous FP32 CUDA sigmoid scores.
        ids: ``[rows, 8]`` contiguous int64 CUDA selected expert ids in
            ``[0, 256)`` (a producer precondition, never read on the host).
        renormalize: ``True`` divides the gathered scores by their pinned K8 sum
            plus ``1e-20``; ``False`` returns them as gathered.

    Returns:
        Fresh contiguous ``[rows, 8]`` tensors: the FP32 weights and the int32
        ids, byte-identical to the torch tail statements on a Torch build whose
        CUDA K8 sum follows the pinned tree.

    Raises:
        ValueError: :func:`route_epilogue_rejection` rejects the call (zero rows
            included). There is no fallback to the torch statements.
    """
    rejection = route_epilogue_rejection(scores, ids, renormalize)
    if rejection:
        raise ValueError(f"{TRITON_ROUTE_EPILOGUE} cannot serve this call: {rejection}")
    # Fresh, contiguous, disjoint from the inputs.
    weights = torch.empty(ids.shape, device=scores.device, dtype=torch.float32)
    out_ids = torch.empty(ids.shape, device=scores.device, dtype=torch.int32)
    _route_epilogue[(scores.shape[0],)](
        scores,
        ids,
        weights,
        out_ids,
        EXPERTS=_EXPERTS,
        NORMALIZE=renormalize,
        **_LAUNCH_OPTIONS,
    )
    return weights, out_ids


def reduction_order_fixture() -> torch.Tensor:
    """The contiguous FP32 ``[n, 8]`` CPU rows the reduction-order pin compares on.

    The rounding row and the mixed-magnitude seeded rows are the ones on which
    different summation orders give different bits; the subnormal, all-equal and
    ``-inf`` rows cover the special-value paths. Deterministic (seed
    ``_PIN_SEED``); the caller moves the rows to the device under test.

    Returns:
        A fresh contiguous FP32 tensor of ``4 + _PIN_RANDOM_ROWS`` rows.
    """
    generator = torch.Generator().manual_seed(_PIN_SEED)
    exponents = torch.randint(
        -149, 1, (_PIN_RANDOM_ROWS, _TOPK), generator=generator
    ).to(torch.float64)
    mantissas = torch.rand(
        _PIN_RANDOM_ROWS, _TOPK, generator=generator, dtype=torch.float64
    )
    negative = torch.rand(_PIN_RANDOM_ROWS, _TOPK, generator=generator) < 0.5
    magnitudes = mantissas * torch.exp2(exponents)
    random_rows = torch.where(negative, -magnitudes, magnitudes).to(torch.float32)
    fixed_rows = torch.tensor(
        [_PIN_ROUNDING_ROW, _PIN_SUBNORMAL_ROW, _PIN_EQUAL_ROW, _PIN_NEG_INF_ROW],
        dtype=torch.float32,
    )
    return torch.cat([fixed_rows, random_rows]).contiguous()


def k8_reduction_tree(values: torch.Tensor) -> torch.Tensor:
    """Sum the last dimension (size 8) in the pinned Torch CUDA order.

    ``((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))`` with one FP32 round-to-nearest add
    per ``+`` (torch elementwise adds: a lone add is never fused, so the order
    is the point and the device is not). This is the reference the GPU pin
    compares ``torch.sum`` against and the tree the kernel spells in
    ``add.rn.f32``.

    Args:
        values: FP32 tensor whose last dimension has size 8.

    Returns:
        FP32 tensor of shape ``values.shape[:-1] + (1,)``.

    Raises:
        ValueError: ``values`` is not FP32 or its last dimension is not 8.
    """
    if values.dtype != torch.float32:
        raise ValueError(f"k8_reduction_tree needs float32 values, got {values.dtype}")
    if values.ndim == 0 or values.shape[-1] != _TOPK:
        raise ValueError(
            f"k8_reduction_tree needs a last dimension of {_TOPK}, "
            f"got shape {tuple(values.shape)}"
        )
    v = values.unbind(-1)
    even = (v[0] + v[4]) + (v[2] + v[6])
    odd = (v[1] + v[5]) + (v[3] + v[7])
    return (even + odd).unsqueeze(-1)


def reduction_order_mismatch(device: torch.device) -> str:
    """Run ``torch.sum`` over the pinned fixture rows on ``device`` and compare it
    bit for bit with :func:`k8_reduction_tree`.

    An operator can run this by hand before an eval under the Triton override;
    the GPU test runs it first and fails every byte-equality test on a mismatch.
    Never called by the kernel or the facade (the route runs under CUDA-graph
    capture, where the host sync this needs is illegal).

    Args:
        device: The CUDA device whose ``torch.sum`` order is the contract.

    Returns:
        ``""`` when every row agrees, else a reason naming ``torch.__version__``
        and the first differing row.

    Raises:
        ValueError: ``device`` is not a CUDA device (a CPU ``torch.sum`` reduces
            in another order and is not the contract).
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            "reduction_order_mismatch needs a CUDA device: the contract is the CUDA "
            f"K8 reduction order, a CPU torch.sum reduces differently (got {device})"
        )
    rows = reduction_order_fixture().to(device)
    torch_sum = rows.sum(dim=-1, keepdim=True).contiguous()
    tree = k8_reduction_tree(rows).contiguous()
    same = torch_sum.view(torch.int32) == tree.view(torch.int32)
    if bool(same.all()):
        return ""
    first = int((~same).flatten().nonzero()[0].item())
    return (
        f"torch {torch.__version__}: the CUDA K8 sum does not follow the pinned "
        "shuffle-down tree ((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7)); "
        f"{TRITON_ROUTE_EPILOGUE}'s bitwise contract is void on this build "
        f"(first differing row {first}: values {rows[first].tolist()}, "
        f"torch.sum {torch_sum[first].item()!r}, tree {tree[first].item()!r})"
    )
