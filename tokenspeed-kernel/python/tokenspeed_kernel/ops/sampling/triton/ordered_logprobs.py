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

"""Registered ordered selected-token logprob implementation."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.sampling import _supports_selected_token_logprobs
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@register_kernel(
    "sampling",
    "gather_token_logprobs",
    name="triton_gather_token_logprobs",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 0),
    ),
    signatures=frozenset(
        {
            format_signature(
                logits=dense_tensor_format(torch.float32),
                tokens=dense_tensor_format(torch.int32),
            )
        }
    ),
    traits={"rows": frozenset({1}), "vocab_size": frozenset({151936})},
    priority=Priority.SPECIALIZED,
    tags={"latency", "cuda_graph"},
)
def triton_gather_token_logprobs(
    logits: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """Return a fresh selected-token logprob for admitted FP32 inputs.

    Args:
        logits: Current-device contiguous FP32 logits shaped [1, 151936].
        tokens: Colocated contiguous INT32 token indices shaped [1].

    Returns:
        A fresh FP32 [1] tensor. Three independent scratch tensors are local
        to this invocation; graph capture owns their storage through replay.
        Unsupported inputs raise before allocating or importing the kernel.
    """
    if not _supports_selected_token_logprobs(logits, tokens):
        raise ValueError("Unsupported ordered selected-token logprob inputs")
    from tokenspeed_kernel.thirdparty.triton.ordered_logprobs import (
        launch_ordered_logprobs,
    )

    lane_max = torch.empty((1024,), dtype=torch.float32, device=logits.device)
    row_max = torch.empty((1,), dtype=torch.float32, device=logits.device)
    lane_sum = torch.empty((1024,), dtype=torch.float32, device=logits.device)
    output = torch.empty((1,), dtype=torch.float32, device=logits.device)
    launch_ordered_logprobs(
        logits, tokens, lane_max, row_max, lane_sum, output, enable_pdl=True
    )
    return output
