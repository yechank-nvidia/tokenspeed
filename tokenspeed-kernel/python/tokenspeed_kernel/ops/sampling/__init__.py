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

"""Sampling kernel entry points.

``sampling.topk_topp_renorm`` operator
=======================================

Per-row top-k then top-p renormalization of ``[bs, V]`` float32 probabilities,
as the FlashInfer sampling backends call it in ``verify()``. Two kernels are
registered under ``("sampling", "topk_topp_renorm")``:

* ``fused_topk_topp_renorm`` (solution ``cuda``, PERFORMANT, NVIDIA): the
  fused native kernel, registered by :mod:`tokenspeed_kernel.ops.sampling.cuda`
  when its wrapper imports. The default winner.
* ``flashinfer_topk_topp_renorm`` (solution ``flashinfer``, PORTABLE, NVIDIA):
  ``top_k_renorm_prob`` followed by ``top_p_renorm_prob(is_deterministic=True)``,
  registered by :mod:`tokenspeed_kernel.ops.sampling.flashinfer`.

Both take ``(probs, top_ks, top_ps)`` and return a fresh ``[bs, V]`` float32
tensor. The arm is switched by kernel name through the ordinary registry
override (the ``TOKENSPEED_KERNEL_OVERRIDE_SAMPLING_TOPK_TOPP_RENORM``
environment variable or a :func:`tokenspeed_kernel.selection.kernel_override`
context). The deprecated ``TS_DISABLE_FUSED_TOPK_TOPP=1`` switch is an alias
of the FlashInfer name; :func:`resolve_topk_topp_renorm_override` maps it with
a one-time warning and refuses to silently outrank a different explicit
override.

Every kernel registered under the operator declares exactly one value of the
``rank_deterministic`` trait, which the sampler reads through
:func:`topk_topp_renorm_rank_deterministic` to decide whether the TP-rank
broadcast of its verify outputs is required (``True``: identical inputs give
bit-identical rows on every rank while PDL is disabled; ``False``: ranks may
round differently, broadcast always). Kernels with pre-capture device state
(the fused kernel's side stream) are set up through
:func:`prepare_topk_topp_renorm`, keyed on the selected kernel name, because
``cudaStreamCreate`` is illegal inside CUDA-graph capture.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, MutableMapping

import torch
from tokenspeed_kernel.ops.sampling.cuda import (
    FUSED_TOPK_TOPP_RENORM,
    fused_topk_topp_prepare,
)
from tokenspeed_kernel.ops.sampling.flashinfer import FLASHINFER_TOPK_TOPP_RENORM
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectedKernel,
    select_kernel,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

logger = logging.getLogger(__name__)

__all__ = [
    "DISABLE_FUSED_TOPK_TOPP_ENV",
    "TOPK_TOPP_RENORM_FAMILY",
    "TOPK_TOPP_RENORM_MODE",
    "TOPK_TOPP_RENORM_OVERRIDE_ENV",
    "TOPK_TOPP_RENORM_RANK_DETERMINISTIC_TRAIT",
    "argmax",
    "prepare_topk_topp_renorm",
    "resolve_topk_topp_renorm_override",
    "select_topk_topp_renorm",
    "topk_topp_renorm",
    "topk_topp_renorm_rank_deterministic",
]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_OUT_DTYPES = (torch.int32, torch.int64)

TOPK_TOPP_RENORM_FAMILY = "sampling"
TOPK_TOPP_RENORM_MODE = "topk_topp_renorm"
# Same derivation as ``select_kernel``'s environment override key.
TOPK_TOPP_RENORM_OVERRIDE_ENV = (
    f"TOKENSPEED_KERNEL_OVERRIDE_{TOPK_TOPP_RENORM_FAMILY.upper()}"
    f"_{TOPK_TOPP_RENORM_MODE.upper()}"
)
TOPK_TOPP_RENORM_RANK_DETERMINISTIC_TRAIT = "rank_deterministic"
# Deprecated import-time switch, now an alias of
# ``TOPK_TOPP_RENORM_OVERRIDE_ENV=flashinfer_topk_topp_renorm``.
DISABLE_FUSED_TOPK_TOPP_ENV = "TS_DISABLE_FUSED_TOPK_TOPP"

# One deprecation warning per process for the alias.
_alias_warned: bool = False

# Per-kernel pre-capture device setup, keyed on the registry name. Kernels
# absent from this table need none.
_TOPK_TOPP_RENORM_PREPARE: dict[str, Callable[[torch.device | str | int], None]] = {
    FUSED_TOPK_TOPP_RENORM: fused_topk_topp_prepare,
}


def _validate_argmax_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    if out.shape != (logits.shape[0],):
        raise ValueError(
            f"out must have shape (M,)={(logits.shape[0],)}, got {tuple(out.shape)}"
        )
    if out.dtype not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(f"out must be int32 or int64; got {out.dtype}")
    if out.device != logits.device:
        raise ValueError("out must be on the same device as logits")


def _argmax_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        _validate_argmax_out(logits, out)
    result = torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def argmax(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    solution: str | None = None,
    override: str | None = None,
) -> torch.Tensor:
    """Return row-wise argmax indices over the last logits dimension.

    Args:
        logits: Input logits with shape ``(M, N)``. The argmax is taken
            over the last dimension.
        out: Optional output buffer with shape ``(M,)`` and dtype int32 or
            int64 on the same device as ``logits``.
        solution: Optional kernel solution to force through normal selection.
        override: Optional exact kernel-name or solution override.

    NaN handling:
        Kernel-backed sampling paths treat NaNs as invalid candidates.
        Rows with at least one non-NaN value return the index of the
        maximum non-NaN value, with ties broken toward the lowest index.
        Rows with no valid non-NaN values return the ``-1`` sentinel.
        Unsupported inputs fall back to ``torch.argmax`` semantics.

    Returns:
        A tensor containing argmax indices for each row of ``logits``.
    """
    if (
        logits.dim() != 2
        or logits.shape[0] == 0
        or not logits.is_cuda
        or logits.dtype not in _SUPPORTED_DTYPES
    ):
        return _argmax_torch_fallback(logits, out=out)

    signature = format_signature(logits=dense_tensor_format(logits.dtype))
    try:
        kernel = select_kernel(
            "sampling",
            "argmax",
            signature,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return _argmax_torch_fallback(logits, out=out)

    shape_params = {
        "M": logits.shape[0],
        "N": logits.shape[1],
        "has_out": out is not None,
    }
    ShapeCapture.get().record(
        "sampling", "argmax", kernel.name, logits.dtype, shape_params
    )
    with kernel_scope(
        "sampling", "argmax", logits.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(logits, out=out)


def resolve_topk_topp_renorm_override(
    environ: MutableMapping[str, str],
) -> str | None:
    """Map the deprecated ``TS_DISABLE_FUSED_TOPK_TOPP=1`` switch onto the registry override.

    ``TS_DISABLE_FUSED_TOPK_TOPP=1`` used to flip an import-time availability
    flag. It is now an alias of
    ``TOKENSPEED_KERNEL_OVERRIDE_SAMPLING_TOPK_TOPP_RENORM=flashinfer_topk_topp_renorm``;
    any other value of the legacy variable is ignored, as before.

    Args:
        environ: The process environment (``os.environ``) or a test double.
            When the alias applies and the registry key is unset, the key is
            written so that every selection in this process (sampling
            backend, :func:`topk_topp_renorm`, ``explain_selection``) sees
            the same override; each TP rank runs this on its own environment.

    Returns:
        ``"flashinfer_topk_topp_renorm"`` when the alias applies -- pass it as
        ``override`` to :func:`select_topk_topp_renorm` -- otherwise ``None``
        (an explicit registry key, if any, is read by ``select_kernel``
        itself).

    Raises:
        ValueError: The registry key already names a different kernel. The
            two surfaces never silently outrank each other; unset one.

    Warns once per process (``DeprecationWarning`` plus a warning log line)
    when the alias applies.
    """
    global _alias_warned
    if environ.get(DISABLE_FUSED_TOPK_TOPP_ENV) != "1":
        return None
    explicit = environ.get(TOPK_TOPP_RENORM_OVERRIDE_ENV)
    if explicit and explicit != FLASHINFER_TOPK_TOPP_RENORM:
        raise ValueError(
            f"{DISABLE_FUSED_TOPK_TOPP_ENV}=1 is an alias of "
            f"{TOPK_TOPP_RENORM_OVERRIDE_ENV}={FLASHINFER_TOPK_TOPP_RENORM}, but "
            f"the environment holds {TOPK_TOPP_RENORM_OVERRIDE_ENV}={explicit!r}; "
            "unset one of them"
        )
    if not explicit:
        environ[TOPK_TOPP_RENORM_OVERRIDE_ENV] = FLASHINFER_TOPK_TOPP_RENORM
    if not _alias_warned:
        _alias_warned = True
        message = (
            f"{DISABLE_FUSED_TOPK_TOPP_ENV}=1 is deprecated; it now selects the "
            f"{TOPK_TOPP_RENORM_FAMILY}.{TOPK_TOPP_RENORM_MODE} kernel "
            f"{FLASHINFER_TOPK_TOPP_RENORM}. Use the registry override "
            f"{TOPK_TOPP_RENORM_FAMILY}.{TOPK_TOPP_RENORM_MODE}="
            f"{FLASHINFER_TOPK_TOPP_RENORM} (environment "
            f"{TOPK_TOPP_RENORM_OVERRIDE_ENV}={FLASHINFER_TOPK_TOPP_RENORM}) instead"
        )
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        logger.warning(message)
    return FLASHINFER_TOPK_TOPP_RENORM


def select_topk_topp_renorm(
    *,
    probs_dtype: torch.dtype,
    solution: str | None,
    override: str | None,
) -> SelectedKernel:
    """Select the ``sampling.topk_topp_renorm`` kernel for probabilities of ``probs_dtype``.

    Args:
        probs_dtype: Storage dtype of the ``[bs, V]`` probability rows; the
            registered kernels take ``torch.float32``.
        solution: Restrict ranked selection to one registered solution
            (``"cuda"`` or ``"flashinfer"``), or ``None`` for any.
        override: Exact kernel name to force -- for example the result of
            :func:`resolve_topk_topp_renorm_override` -- or ``None`` for
            ranked selection. A :func:`tokenspeed_kernel.selection.kernel_override`
            context and ``TOKENSPEED_KERNEL_OVERRIDE_SAMPLING_TOPK_TOPP_RENORM``
            outrank this argument (see ``select_kernel``).

    Returns:
        The selected kernel; call it as ``kernel(probs, top_ks, top_ps)`` and
        read its registry name from ``.name``.

    Raises:
        NoKernelFoundError: No registered kernel matches, for example off
            NVIDIA or for an unregistered override name.
        ValueError: An override named a kernel registered under another
            operator. ``select_kernel`` resolves an override by name alone,
            so the operator check is made here before the kernel can be
            called with ``(probs, top_ks, top_ps)``.
    """
    signature = format_signature(probs=dense_tensor_format(probs_dtype))
    selected = select_kernel(
        TOPK_TOPP_RENORM_FAMILY,
        TOPK_TOPP_RENORM_MODE,
        signature,
        solution=solution,
        override=override,
    )
    _operator_spec(selected.name)
    return selected


def _operator_spec(kernel_name: str) -> KernelSpec:
    """The registry spec of ``kernel_name`` if it belongs to ``sampling.topk_topp_renorm``.

    Raises:
        ValueError: ``kernel_name`` is unregistered or registered under
            another operator.
    """
    spec = KernelRegistry.get().get_by_name(kernel_name)
    if spec is None or (spec.family, spec.mode) != (
        TOPK_TOPP_RENORM_FAMILY,
        TOPK_TOPP_RENORM_MODE,
    ):
        raise ValueError(
            f"{kernel_name!r} is not registered under "
            f"{TOPK_TOPP_RENORM_FAMILY}.{TOPK_TOPP_RENORM_MODE}"
        )
    return spec


def topk_topp_renorm_rank_deterministic(kernel_name: str) -> bool:
    """Read the ``rank_deterministic`` trait of a ``sampling.topk_topp_renorm`` kernel.

    ``True``: identical inputs give bit-identical rows on every TP rank while
    PDL is disabled, so a sampler may skip broadcasting rank 0's verify
    outputs. ``False``: ranks may round differently and the broadcast is
    load-bearing. Every kernel registered under the operator must declare
    exactly one value.

    Args:
        kernel_name: Registry name, normally ``SelectedKernel.name``.

    Returns:
        The declared trait value.

    Raises:
        ValueError: ``kernel_name`` is not registered under the operator or
            does not declare exactly one trait value.
    """
    spec = _operator_spec(kernel_name)
    values = spec.traits.get(TOPK_TOPP_RENORM_RANK_DETERMINISTIC_TRAIT)
    if values is None or len(values) != 1:
        raise ValueError(
            f"{kernel_name!r} must declare exactly one "
            f"{TOPK_TOPP_RENORM_RANK_DETERMINISTIC_TRAIT!r} trait value; "
            f"got {values!r}"
        )
    (value,) = values
    return bool(value)


def prepare_topk_topp_renorm(
    kernel_name: str, device: torch.device | str | int
) -> None:
    """Run the selected kernel's one-time device setup outside CUDA-graph capture.

    ``fused_topk_topp_renorm`` overlaps its top-p radix on a per-device side
    stream; ``cudaStreamCreate`` is illegal inside capture, so the stream is
    created here (idempotent) before the first captured call. Kernels without
    such state are a no-op.

    Args:
        kernel_name: Registry name of the selected kernel.
        device: Device the kernel will run on. An unindexed ``"cuda"`` (the
            runtime's default ``device`` string) is resolved to the current
            CUDA device, because the kernel looks its stream up by
            ``probs.device`` at call time, which always carries the index.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    hook = _TOPK_TOPP_RENORM_PREPARE.get(kernel_name)
    if hook is not None:
        hook(device)


def topk_topp_renorm(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    solution: str | None,
    override: str | None,
) -> torch.Tensor:
    """Per-row top-k then top-p renormalization through ``sampling.topk_topp_renorm``.

    Selects on every call (cached; overrides honoured) and records the shape
    and kernel like :func:`argmax`. Callers that replay from a CUDA graph
    should instead bind once with :func:`select_topk_topp_renorm`, run
    :func:`prepare_topk_topp_renorm` for the bound name, and call the bound
    kernel.

    Args:
        probs: ``[bs, V]`` float32 probabilities; each row sums to 1.
        top_ks: ``[bs]`` int32 per-row K; the sentinel ``1 << 30`` disables
            the top-k cap for the row.
        top_ps: ``[bs]`` float32 per-row P in ``(0, 1]``.
        solution: See :func:`select_topk_topp_renorm`.
        override: See :func:`select_topk_topp_renorm`.

    Returns:
        A fresh ``[bs, V]`` float32 tensor: positions outside the per-row
        top-k then top-p set are 0, kept positions are renormalized so each
        row sums to 1.
    """
    kernel = select_topk_topp_renorm(
        probs_dtype=probs.dtype, solution=solution, override=override
    )
    shape_params = {"M": probs.shape[0], "N": probs.shape[1]}
    ShapeCapture.get().record(
        TOPK_TOPP_RENORM_FAMILY,
        TOPK_TOPP_RENORM_MODE,
        kernel.name,
        probs.dtype,
        shape_params,
    )
    with kernel_scope(
        TOPK_TOPP_RENORM_FAMILY,
        TOPK_TOPP_RENORM_MODE,
        probs.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(probs, top_ks, top_ps)


# Backend registration (side-effect imports).
import tokenspeed_kernel.ops.sampling.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.sampling.gluon  # noqa: E402,F401
