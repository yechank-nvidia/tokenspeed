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

"""CUDA sampling kernels.

Registers ``fused_topk_topp_renorm`` under the registry operator
``sampling.topk_topp_renorm`` (the fused ON arm) whenever the native wrapper
imports; the FlashInfer arm is registered by
:mod:`tokenspeed_kernel.ops.sampling.flashinfer`. The operator contract lives
in :mod:`tokenspeed_kernel.ops.sampling`.
"""

import os as _os

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Registry name of the fused CUDA arm of ``sampling.topk_topp_renorm``.
FUSED_TOPK_TOPP_RENORM = "fused_topk_topp_renorm"

chain_speculative_sampling_target_only = error_fn
# Prepare is a side-stream registration helper. On non-NVIDIA the fused
# kernel doesn't exist, so callers should never reach the renorm path —
# but prepare is dispatched unconditionally through
# ``tokenspeed_kernel.ops.sampling.prepare_topk_topp_renorm``, so we keep it
# as a silent no-op rather than error_fn.
fused_topk_topp_prepare = lambda *_args, **_kwargs: None  # noqa: E731
fused_topk_topp_renorm = error_fn
fused_topk_topp_workspace_size = error_fn
verify_chain_greedy = error_fn

# Legacy availability flag: True when the fused wrapper imported and the
# deprecated ``TS_DISABLE_FUSED_TOPK_TOPP=1`` switch is not set. Kept for the
# callers that have not moved to the ``sampling.topk_topp_renorm`` operator
# (the flashinfer_full sampling backend and other legacy callers). The
# registration below does not read it: disabling the fused arm goes through
# the registry override that
# ``tokenspeed_kernel.ops.sampling.resolve_topk_topp_renorm_override`` derives
# from the same switch. The 2026-07-14 misaligned-address quarantine was lifted
# after upstream #683 fixed the kernel's graph-capture addressing without
# losing float4 throughput.
fused_topk_topp_available = False

if current_platform().is_nvidia:
    try:
        from tokenspeed_kernel.thirdparty.cuda.sampling_chain import (
            chain_speculative_sampling_target_only,
            verify_chain_greedy,
        )
    except ImportError:
        pass

    try:
        from tokenspeed_kernel.thirdparty.cuda.fused_topk_topp import (
            fused_topk_topp_renorm as _fused_topk_topp_renorm_impl,
        )
        from tokenspeed_kernel.thirdparty.cuda.fused_topk_topp import (
            fused_topk_topp_workspace_size,
        )
        from tokenspeed_kernel.thirdparty.cuda.fused_topk_topp import (
            prepare_for_device as fused_topk_topp_prepare,
        )
    except ImportError:
        pass
    else:
        fused_topk_topp_available = (
            _os.environ.get("TS_DISABLE_FUSED_TOPK_TOPP", "0") != "1"
        )

        @register_kernel(
            "sampling",
            "topk_topp_renorm",
            name=FUSED_TOPK_TOPP_RENORM,
            solution="cuda",
            capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
            signatures=format_signatures("probs", "dense", {torch.float32}),
            # Fixed per-row reduction order: identical inputs give bit-identical
            # rows on every TP rank while PDL is off, so the sampler may skip
            # the rank-0 broadcast of its verify outputs.
            traits={"rank_deterministic": frozenset({True})},
            priority=Priority.PERFORMANT,
        )
        def fused_topk_topp_renorm(
            probs: torch.Tensor,
            top_ks: torch.Tensor,
            top_ps: torch.Tensor,
            workspace: torch.Tensor | None = None,
            out: torch.Tensor | None = None,
            enable_pdl: bool | None = None,
        ) -> torch.Tensor:
            """Fused top-k + top-p renormalization in one launch sequence.

            Matches ``flashinfer_topk_topp_renorm`` (``top_k_renorm_prob`` then
            ``top_p_renorm_prob(is_deterministic=True)``). Call
            ``fused_topk_topp_prepare(device)`` outside CUDA-graph capture
            before the first captured call: the top-p radix overlaps the top-k
            radix on a per-device side stream that cannot be created inside
            capture.

            Args:
                probs: ``[bs, V]`` float32 probabilities; each row sums to 1.
                top_ks: ``[bs]`` int32 per-row K in ``[1, V)``, or the sentinel
                    ``1 << 30`` which routes the row through the top-p only
                    path.
                top_ps: ``[bs]`` float32 per-row P in ``(0, 1]``.
                workspace: Optional pre-allocated uint8 scratch buffer of
                    ``fused_topk_topp_workspace_size(bs, V)`` bytes; allocated
                    per call when omitted.
                out: Optional pre-allocated ``[bs, V]`` float32 output;
                    allocated per call when omitted.
                enable_pdl: Programmatic Dependent Launch attribute; ``None``
                    uses the platform default.

            Returns:
                ``out`` (or the fresh ``[bs, V]`` float32 tensor). Non-kept
                positions are 0; kept positions are renormalized so each row
                sums to 1.
            """
            return _fused_topk_topp_renorm_impl(
                probs,
                top_ks,
                top_ps,
                workspace=workspace,
                out=out,
                enable_pdl=enable_pdl,
            )


__all__ = [
    "FUSED_TOPK_TOPP_RENORM",
    "chain_speculative_sampling_target_only",
    "fused_topk_topp_available",
    "fused_topk_topp_prepare",
    "fused_topk_topp_renorm",
    "fused_topk_topp_workspace_size",
    "verify_chain_greedy",
]
