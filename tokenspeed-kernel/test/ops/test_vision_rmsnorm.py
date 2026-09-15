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

"""Contract and numerical tests for the experimental vision RMSNorm kernel."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tokenspeed_kernel.ops import vision_rmsnorm as norm


def tensors(n=16):
    return torch.ones((n, 1024), dtype=torch.bfloat16), torch.ones(
        1024, dtype=torch.bfloat16
    )


@pytest.mark.parametrize("n", [16, 17, 31, 32, 65, 780, 864, 2816])
def test_valid_metadata(n):
    x, w = tensors(n)
    assert norm.supports_layout(x, w, 1e-6)


@pytest.mark.parametrize(
    "kind",
    [
        "small",
        "rank",
        "width",
        "xdtype",
        "wdtype",
        "weight-shape",
        "column-gap",
        "row-overlap",
        "weight-stride",
        "eps",
        "eps-type",
    ],
)
def test_rejected_metadata(kind):
    x, w = tensors()
    eps = 1e-6
    if kind == "small":
        x = x[:15]
    if kind == "rank":
        x = x.unsqueeze(0)
    if kind == "width":
        x = x[:, :1023]
    if kind == "xdtype":
        x = x.float()
    if kind == "wdtype":
        w = w.float()
    if kind == "weight-shape":
        w = w.unsqueeze(0)
    if kind == "column-gap":
        x = torch.ones((16, 2048), dtype=x.dtype)[:, ::2]
    if kind == "row-overlap":
        x = x.as_strided((16, 1024), (0, 1))
    if kind == "weight-stride":
        w = torch.ones(2048, dtype=w.dtype)[::2]
    if kind == "eps":
        eps = 1e-5
    if kind == "eps-type":
        eps = torch.tensor(1e-6)
    assert not norm.supports_layout(x, w, eps)


def test_row_gap_and_nonzero_offsets():
    x = torch.ones(17 * 1041 + 7, dtype=torch.bfloat16).as_strided(
        (17, 1024), (1041, 1), 7
    )
    w = torch.ones(1027, dtype=torch.bfloat16)[3:]
    assert norm.supports_layout(x, w, 1e-6)


@pytest.mark.parametrize("parameter", ["input", "weight"])
def test_autograd_falls_back_and_has_exact_gradients(parameter):
    x, w = tensors()
    (x if parameter == "input" else w).requires_grad_(True)
    assert not norm.supports_layout(x, w, 1e-6)
    actual = norm.apply_vision_rmsnorm(x, w)
    expected = norm.reference(x, w)
    ga = torch.autograd.grad(actual.float().sum(), x if parameter == "input" else w)
    gb = torch.autograd.grad(expected.float().sum(), x if parameter == "input" else w)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(ga[0], gb[0])
    with torch.inference_mode():
        assert norm.supports_layout(x, w, 1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_cpu_fallback_does_not_import_dsl(dtype):
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((17, 1024), generator=generator).to(dtype)
    w = torch.randn(1024, generator=generator).to(dtype)
    with patch.object(
        norm, "_implementation", side_effect=AssertionError("CPU imported optional DSL")
    ):
        actual = norm.apply_vision_rmsnorm(x, w)
    a = x.float()
    expected = (
        w.float() * (a * torch.rsqrt(a.square().mean(-1, keepdim=True) + 1e-6))
    ).to(dtype)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert actual.data_ptr() != x.data_ptr()


def test_reduction_schedule_covers_each_column_once():
    # Independent schedule reconstruction used in reviewing the explicit kernel.
    lanes = [
        [[4 * lane + 128 * k + j for k in range(8)] for j in range(4)]
        for lane in range(32)
    ]
    assert sorted(
        index for lane in lanes for component in lane for index in component
    ) == list(range(1024))
    support = [{index for component in lane for index in component} for lane in lanes]
    for step in (16, 8, 4, 2, 1):
        previous = support
        support = [
            (
                previous[lane] | previous[lane + step]
                if lane + step < 32
                else previous[lane]
            )
            for lane in range(32)
        ]
    assert support[0] == set(range(1024))


def test_pinned_scope_and_explicit_rounding():
    source = Path(norm.__file__).with_name("cute_dsl.py").read_text()
    assert "add.rn.f32" in source and "mul.rn.f32" in source
    assert "fma." not in source and ".ftz." not in source
    assert "shuffle_sync_down(total, 16 >> level)" in source
    assert "range_constexpr(8)" in source
    assert norm.TORCH_VERSION == "2.13.0+cu130"


@pytest.mark.parametrize("rows", [16, 17, 780, 864, 2816])
def test_cuda_outputs_and_intermediate_means_are_exact(rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if (
        torch.cuda.get_device_capability() != (10, 0)
        or torch.__version__ != norm.TORCH_VERSION
    ):
        pytest.skip("requires the pinned sm_100 and PyTorch implementation")
    from tokenspeed_kernel.ops.vision_rmsnorm.cute_dsl import diagnostic_mean

    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn((rows, 1024), generator=generator, device="cuda").bfloat16()
    weight = torch.randn(1024, generator=generator, device="cuda").bfloat16()
    with torch.inference_mode():
        expected = norm.reference(x, weight)
        actual = norm.apply_vision_rmsnorm(x, weight)
        expected_mean = x.float().square().mean(-1, keepdim=True)
        actual_mean = diagnostic_mean(x)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_mean.view(torch.uint8), expected_mean.view(torch.uint8))


def test_cuda_vision_adapter_uses_the_kernel_boundary(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from tokenspeed.runtime.models.deepseek_v4_vision import DeepseekV4VisionRMSNorm

    layer = DeepseekV4VisionRMSNorm(1024).to(device="cuda", dtype=torch.bfloat16)
    x = torch.ones((17, 1024), device="cuda", dtype=torch.bfloat16)
    calls = []
    original = norm.apply_vision_rmsnorm

    def observed(value, weight, eps):
        calls.append((tuple(value.shape), eps))
        return original(value, weight, eps)

    monkeypatch.setattr(norm, "apply_vision_rmsnorm", observed)
    with torch.inference_mode():
        actual = layer(x)
        expected = norm.reference(x, layer.weight)
    assert calls == [((17, 1024), 1e-6)]
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
