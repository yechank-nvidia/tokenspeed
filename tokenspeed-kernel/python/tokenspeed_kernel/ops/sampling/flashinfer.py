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

"""FlashInfer sampling kernels.

Besides re-exporting the FlashInfer sampling entry points, this module
registers ``flashinfer_topk_topp_renorm`` under the registry operator
``sampling.topk_topp_renorm``: the reference (OFF) arm of the fused CUDA
renormalizer registered by :mod:`tokenspeed_kernel.ops.sampling.cuda`. The
operator contract (selection, broadcast rule, pre-capture setup, the
deprecated ``TS_DISABLE_FUSED_TOPK_TOPP`` alias) lives in
:mod:`tokenspeed_kernel.ops.sampling`.
"""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Registry name of the FlashInfer arm of ``sampling.topk_topp_renorm``; also
# the target of the deprecated ``TS_DISABLE_FUSED_TOPK_TOPP=1`` alias.
FLASHINFER_TOPK_TOPP_RENORM = "flashinfer_topk_topp_renorm"

flashinfer_topk_topp_renorm = error_fn
min_p_sampling_from_probs = error_fn
softmax = error_fn
top_k_renorm_prob = error_fn
top_k_top_p_sampling_from_probs = error_fn
top_k_top_p_sampling_from_logits = error_fn
top_p_renorm_prob = error_fn
top_p_renorm_probs = error_fn

if current_platform().is_nvidia:
    try:
        from flashinfer.sampling import (
            min_p_sampling_from_probs,
            top_k_renorm_prob,
            top_k_top_p_sampling_from_logits,
            top_k_top_p_sampling_from_probs,
            top_p_renorm_prob,
            top_p_renorm_probs,
        )
    except ImportError:
        pass

    try:
        from tokenspeed_kernel.thirdparty.cuda.flashinfer_softmax import softmax
    except ImportError:
        pass

if top_k_renorm_prob is not error_fn and top_p_renorm_prob is not error_fn:

    @register_kernel(
        "sampling",
        "topk_topp_renorm",
        name=FLASHINFER_TOPK_TOPP_RENORM,
        solution="flashinfer",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=format_signatures("probs", "dense", {torch.float32}),
        # Rows wider than one CTA's chunk reduce the kept mass with a float
        # atomicAdd, so TP ranks fed identical inputs may round differently:
        # the sampler must broadcast rank 0's verify outputs.
        traits={"rank_deterministic": frozenset({False})},
        priority=Priority.PORTABLE,
    )
    def flashinfer_topk_topp_renorm(
        probs: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
    ) -> torch.Tensor:
        """Top-k then top-p renormalization as two FlashInfer launches.

        Args:
            probs: ``[bs, V]`` float32 probabilities; each row sums to 1.
            top_ks: ``[bs]`` int32 per-row K. The sentinel ``1 << 30`` keeps
                the whole row for the top-k stage.
            top_ps: ``[bs]`` float32 per-row P in ``(0, 1]``.

        Returns:
            A fresh ``[bs, V]`` float32 tensor. Positions outside the per-row
            top-k then top-p set are 0; kept positions are renormalized so
            each row sums to 1. ``top_p_renorm_prob`` runs with
            ``is_deterministic=True``; ``top_k_renorm_prob`` has no such knob.
        """
        renormed = top_k_renorm_prob(probs, top_ks)
        return top_p_renorm_prob(renormed, top_ps, is_deterministic=True)


__all__ = [
    "FLASHINFER_TOPK_TOPP_RENORM",
    "flashinfer_topk_topp_renorm",
    "min_p_sampling_from_probs",
    "softmax",
    "top_k_renorm_prob",
    "top_k_top_p_sampling_from_logits",
    "top_k_top_p_sampling_from_probs",
    "top_p_renorm_prob",
    "top_p_renorm_probs",
]
