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

"""Selected-weight epilogue of top-k expert routing: the ``moe.route_epilogue`` operator.

A top-k routing callback ends by gathering the FP32 scores of the selected
experts, dividing them (under ``renormalize``) by their sum plus ``1e-20`` and
converting the int64 ids to int32. That tail is the operator
``("moe", "route_epilogue")``:
:func:`moe_route_epilogue` takes the route's ``scores`` and ``ids`` and returns
the ``(weights, ids)`` pair the route returns. Two kernels are registered:

* ``torch_route_epilogue`` (solution ``torch``, PORTABLE, any device): the three
  torch tail statements of the route verbatim. Admits every shape and both
  ``renormalize`` values; the ranked default and the production path.
* ``triton_route_epilogue`` (solution ``triton``, REFERENCE, NVIDIA;
  :mod:`tokenspeed_kernel.ops.moe.triton.route_epilogue`): one program per row
  gathers the eight scores and spells the K8 sum in the order Torch's CUDA
  reduction uses (``((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))``, then ``+1e-20``, then
  a correctly rounded division). Never auto-selected; reached only by name
  through the ordinary registry override (``--kernel-override
  moe.route_epilogue=triton_route_epilogue``, mirrored to
  ``TOKENSPEED_KERNEL_OVERRIDE_MOE_ROUTE_EPILOGUE``, or a
  :func:`tokenspeed_kernel.selection.kernel_override` context). It admits any
  positive row count of the example shape (contiguous FP32 ``[rows, 256]``
  scores, int64 ``[rows, 8]`` ids, one CUDA device, ``renormalize`` True or
  False) and raises on any other call, zero rows included: no in-function
  fallback.

The two kernels agree byte for byte as long as this Torch build's CUDA K8 sum
follows the pinned tree, which the GPU test
``test/nvidia/ops/moe/test_route_epilogue_cuda.py`` checks first (a mismatch fails
the suite, it never skips).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.moe.triton.route_epilogue import TRITON_ROUTE_EPILOGUE
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import (
    KernelRegistry,
    KernelSpec,
    Priority,
    register_kernel,
)
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "ROUTE_EPILOGUE_FAMILY",
    "ROUTE_EPILOGUE_MODE",
    "ROUTE_EPILOGUE_OVERRIDE_ENV",
    "ROUTE_EPILOGUE_SIGNATURES",
    "TORCH_ROUTE_EPILOGUE",
    "TRITON_ROUTE_EPILOGUE",
    "moe_route_epilogue",
    "route_epilogue_traits",
    "torch_route_epilogue",
]

ROUTE_EPILOGUE_FAMILY = "moe"
ROUTE_EPILOGUE_MODE = "route_epilogue"
# Same derivation as ``select_kernel``'s environment override key.
ROUTE_EPILOGUE_OVERRIDE_ENV = (
    f"TOKENSPEED_KERNEL_OVERRIDE_{ROUTE_EPILOGUE_FAMILY.upper()}"
    f"_{ROUTE_EPILOGUE_MODE.upper()}"
)
# Registered name of the production default (the ON target of the switch).
TORCH_ROUTE_EPILOGUE = "torch_route_epilogue"
# The one dtype pair the route produces: FP32 sigmoid scores
# (``gating_output.float().sigmoid()``) and int64 ``topk(...).indices``.
ROUTE_EPILOGUE_SIGNATURES = frozenset(
    {
        format_signature(
            scores=dense_tensor_format(torch.float32),
            ids=dense_tensor_format(torch.int64),
        )
    }
)


def route_epilogue_traits(
    scores: torch.Tensor, ids: torch.Tensor, renormalize: bool
) -> dict[str, int | bool]:
    """The trait dict :func:`moe_route_epilogue` selects with (metadata only).

    ``rows``/``experts`` come from ``scores``, ``topk`` from ``ids``,
    ``renormalize`` is the flag itself and ``contiguous`` is whether both
    tensors are contiguous. ``experts``, ``topk``, ``renormalize`` and
    ``contiguous`` are the values the ``triton_route_epilogue`` spec constrains;
    ``rows`` is carried for the shape capture and the kernel scope (no
    registered spec constrains it since the Triton kernel admits every positive
    row count, and ``spec_matches_traits`` ignores traits a spec does not
    declare); ``torch_route_epilogue`` declares no traits and admits every dict.
    """
    return {
        "rows": int(scores.shape[0]),
        "experts": int(scores.shape[1]),
        "topk": int(ids.shape[1]),
        "renormalize": renormalize,
        "contiguous": scores.is_contiguous() and ids.is_contiguous(),
    }


def _operator_spec(kernel_name: str) -> KernelSpec:
    """The registry spec of ``kernel_name`` if it belongs to ``moe.route_epilogue``.

    Raises:
        ValueError: ``kernel_name`` is unregistered or registered under another
            operator. ``select_kernel`` resolves an override by name alone, so
            the operator check is made here before the kernel is called with
            the route tensors.
    """
    spec = KernelRegistry.get().get_by_name(kernel_name)
    if spec is None or (spec.family, spec.mode) != (
        ROUTE_EPILOGUE_FAMILY,
        ROUTE_EPILOGUE_MODE,
    ):
        raise ValueError(
            f"{kernel_name!r} is not registered under "
            f"{ROUTE_EPILOGUE_FAMILY}.{ROUTE_EPILOGUE_MODE}"
        )
    return spec


def moe_route_epilogue(
    scores: torch.Tensor,
    ids: torch.Tensor,
    *,
    renormalize: bool,
    solution: str | None,
    override: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather, renormalize and convert the route's selection through ``moe.route_epilogue``.

    Args:
        scores: ``[rows, experts]`` FP32 sigmoid scores
            (``gating_output.float().sigmoid()``).
        ids: ``[rows, topk]`` int64 selected expert ids from
            ``masked_choices.topk(topk).indices``; values in ``[0, experts)`` are a
            producer precondition (never read on the host).
        renormalize: ``True`` divides the gathered scores by their per-row sum
            plus ``1e-20``; ``False`` returns them as gathered. Selects the
            arithmetic branch, so it is keyword-only and must be a ``bool``.
        solution: Restrict ranked selection to one solution (``"torch"``/``"triton"``)
            or ``None`` for any.
        override: Exact kernel name to force, or ``None`` for ranked selection; the
            ``kernel_override`` context and
            ``TOKENSPEED_KERNEL_OVERRIDE_MOE_ROUTE_EPILOGUE`` outrank it (see
            ``select_kernel``).

    Returns:
        ``(weights, ids)``: a fresh ``[rows, topk]`` FP32 tensor equal to
        ``scores.gather(-1, ids)``, divided by ``(sum + 1e-20)`` per row when
        ``renormalize``, and the ids as a fresh ``[rows, topk]`` int32 tensor;
        inputs are not modified.

    Raises:
        ValueError: The tensors do not form the route's ``(scores, ids)`` pair,
            ``renormalize`` is not a ``bool``, or the resolved kernel is not
            registered under ``moe.route_epilogue``.
        NoKernelFoundError: No registered kernel serves the dtype pair.
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape [rows, experts]")
    rows, experts = (int(size) for size in scores.shape)
    if ids.ndim != 2 or ids.shape[0] != rows or not 0 < ids.shape[1] <= experts:
        raise ValueError(
            f"ids must have shape [rows, topk] with 0 < topk <= experts ({experts})"
        )
    if scores.device != ids.device:
        raise ValueError("scores and ids must share one device")
    if not isinstance(renormalize, bool):
        raise ValueError(
            f"renormalize must be a bool, got {type(renormalize).__name__}"
        )
    # An unregistered dtype pair makes ``select_kernel`` raise ``NoKernelFoundError``
    # naming the signature.
    signature = format_signature(
        scores=dense_tensor_format(scores.dtype),
        ids=dense_tensor_format(ids.dtype),
    )
    traits = route_epilogue_traits(scores, ids, renormalize)
    kernel = select_kernel(
        ROUTE_EPILOGUE_FAMILY,
        ROUTE_EPILOGUE_MODE,
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    _operator_spec(kernel.name)
    shape_params = {
        "rows": traits["rows"],
        "experts": traits["experts"],
        "topk": traits["topk"],
        "renormalize": traits["renormalize"],
    }
    ShapeCapture.get().record(
        ROUTE_EPILOGUE_FAMILY,
        ROUTE_EPILOGUE_MODE,
        kernel.name,
        scores.dtype,
        shape_params,
    )
    with kernel_scope(
        ROUTE_EPILOGUE_FAMILY,
        ROUTE_EPILOGUE_MODE,
        scores.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(scores, ids, renormalize)


@register_kernel(
    "moe",
    "route_epilogue",
    name=TORCH_ROUTE_EPILOGUE,
    solution="torch",
    signatures=ROUTE_EPILOGUE_SIGNATURES,
    # Admits every shape and both renormalize values: the statements are
    # general. While this is the only non-REFERENCE registration under the
    # operator, ranked selection lands here for every call, so the route's
    # behaviour is byte-identical to the inlined statements it replaced.
    traits={},
    priority=Priority.PORTABLE,
)
def torch_route_epilogue(
    scores: torch.Tensor, ids: torch.Tensor, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """The three torch statements of the routing tail verbatim; runs on any device."""
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights, ids.to(torch.int32)
