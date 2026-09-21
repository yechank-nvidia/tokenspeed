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

"""Position-dependent query scaling for attention."""

import torch


def apply_ssmax_root10(
    query: torch.Tensor,
    positions: torch.Tensor,
    offset: int,
    denominator: float,
) -> torch.Tensor:
    """Scale Q by ``(1 + (position + offset) / denominator) ** 0.1``.

    Args:
        query: FP32, BF16, or FP16 queries shaped ``[..., heads, head_dim]``.
        positions: Positions on the query device, shaped ``query.shape[:-2]``.
            The caller must ensure ``positions + offset > -denominator``.
        offset: Position offset before scaling.
        denominator: Positive normalization constant, chosen by the model.

    Returns:
        A new tensor with the query's shape and dtype. Scaling and multiplication
        use FP32; the result is cast once to the query dtype.
    """
    if denominator <= 0:
        raise ValueError(f"SSMax denominator must be positive, got {denominator}")
    if query.dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise ValueError(f"unsupported SSMax query dtype: {query.dtype}")
    if query.ndim < 2 or positions.shape != query.shape[:-2]:
        raise ValueError("SSMax requires one position per query head group")
    # Keep position values on device so this path also runs during CUDA capture.
    ratio = (positions.float() + float(offset)) / float(denominator)
    scale = torch.exp(0.1 * torch.log1p(ratio))
    return (query * scale[..., None, None]).to(query.dtype)
