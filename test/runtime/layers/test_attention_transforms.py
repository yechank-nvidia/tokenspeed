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

"""Reference math and CUDA replay checks for shared attention transforms."""

import math
from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch

from tokenspeed.runtime.layers.attention.ssmax import apply_ssmax_root10

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("shape", ((0, 3, 4), (5, 3, 4), (2, 5, 3, 4)))
def test_ssmax_root10_matches_reference(dtype, shape):
    # Strided head-width views also cover outputs sliced from fused projections.
    query = torch.linspace(-2, 2, math.prod(shape) * 2).to(dtype)
    query = query.reshape(*shape[:-1], 2 * shape[-1])[..., ::2]
    original = query.clone()
    positions = torch.arange(math.prod(shape[:-2])).reshape(shape[:-2]) * 4096
    expected = (
        query.double()
        * (1 + (positions.double() + 3) / 2048.0).pow(0.1)[..., None, None]
    )

    actual = apply_ssmax_root10(query, positions, 3, 2048.0)

    assert actual.dtype == dtype
    tolerance = 1e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected.to(dtype), rtol=tolerance, atol=0)
    torch.testing.assert_close(query, original, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ssmax_root10_replays_with_updated_positions():
    query = torch.linspace(-2, 2, 48, device="cuda").reshape(4, 3, 4).bfloat16()
    positions = torch.arange(4, device="cuda")
    for _ in range(3):
        apply_ssmax_root10(query, positions, 1, 2048.0)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply_ssmax_root10(query, positions, 1, 2048.0)

    positions.add_(8192)
    graph.replay()
    expected = apply_ssmax_root10(query, positions, 1, 2048.0)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
