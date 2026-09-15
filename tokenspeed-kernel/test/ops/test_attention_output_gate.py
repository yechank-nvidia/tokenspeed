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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.activation import attention_gate_mul
from tokenspeed_kernel.registry import KernelRegistry

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _reference(output, gate, bias, floor, temperature):
    shaped = gate.reshape(output.shape[0], bias.numel(), -1).float()
    multiplier = floor + (1.0 - floor) * torch.sigmoid(
        (shaped + bias.float()[None, :, None]) / temperature
    )
    return (output.float() * multiplier.reshape_as(output)).to(output.dtype)


def test_attention_gate_registration() -> None:
    registry = KernelRegistry.get()
    spec = registry.get_by_name("triton_attention_gate_mul")
    assert spec is not None
    assert (spec.family, spec.mode) == ("activation", "attention_gate_mul")
    assert registry.get_impl(spec.name) is attention_gate_mul


@requires_gpu
@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (1, 3, 17), (7, 5, 32), (33, 7, 128), (129, 8, 64)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "floor,temperature,bias_dtype",
    [(0.0, 1.0, torch.bfloat16), (0.2, 0.7, torch.float32), (1.0, 2.0, torch.float32)],
)
def test_attention_gate_matches_reference(
    shape, dtype, strided, floor, temperature, bias_dtype
):
    tokens, heads, dim = shape
    torch.manual_seed(173 + tokens + heads)
    output = torch.randn((tokens, heads * dim), device="cuda", dtype=dtype)
    if strided:
        storage = torch.randn((tokens, heads + 1, 2 * dim), device="cuda", dtype=dtype)
        gate = storage[:, :heads, dim:]
    else:
        gate = torch.randn_like(output)
    bias = torch.randn(heads, device="cuda", dtype=bias_dtype)
    original_gate, original_bias = gate.clone(), bias.clone()
    expected = _reference(output, gate, bias, floor, temperature)
    actual = attention_gate_mul(output, gate, bias, floor, temperature)
    assert actual is output
    tolerance = {torch.bfloat16: 0.016, torch.float16: 0.002, torch.float32: 2e-6}[
        dtype
    ]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.equal(gate, original_gate)
    assert torch.equal(bias, original_bias)


@requires_gpu
def test_attention_gate_broadcast_rows_and_storage_offsets():
    storage = torch.full((8, 96), 77.0, device="cuda")
    output = storage[1:]
    output.normal_()
    gate = torch.randn((1, 96), device="cuda").expand(7, -1)
    bias = torch.randn(4, device="cuda")[1:]
    expected = _reference(output, gate, bias, 0.15, 0.8)
    attention_gate_mul(output, gate, bias, 0.15, 0.8)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-6)
    assert (storage[0] == 77.0).all()


@requires_gpu
@pytest.mark.parametrize("strided", [False, True])
def test_attention_gate_graph_replay_reads_updated_inputs(strided):
    output = torch.randn((9, 96), dtype=torch.bfloat16, device="cuda")
    gate_storage = torch.randn((9, 3, 64), dtype=output.dtype, device="cuda")
    gate = gate_storage[:, :, 32:] if strided else torch.randn_like(output)
    bias = torch.randn(3, device="cuda")
    attention_gate_mul(output, gate, bias, 0.2, 0.7)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = attention_gate_mul(output, gate, bias, 0.2, 0.7)
    assert result is output
    for shift in (0.0, 0.25):
        output.normal_()
        gate.add_(shift)
        bias.add_(shift)
        expected = _reference(output, gate, bias, 0.2, 0.7)
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0.016, rtol=0.016)


@requires_gpu
def test_attention_gate_saturated_values():
    output = torch.tensor([[2.0, -3.0, 4.0, 5.0]], device="cuda")
    gate = torch.tensor([[-100.0, 100.0, float("-inf"), float("inf")]], device="cuda")
    bias = torch.tensor([0.5, -0.5], device="cuda")
    expected = torch.tensor([[0.5, -3.0, 1.0, 5.0]], device="cuda")
    attention_gate_mul(output, gate, bias, 0.25, 1.0)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@requires_gpu
@pytest.mark.parametrize("shape", [(0, 96), (3, 0)])
def test_attention_gate_empty(shape):
    output = torch.empty(shape, device="cuda")
    gate = torch.empty_like(output)
    bias = torch.zeros(3, device="cuda")
    assert attention_gate_mul(output, gate, bias, 0.0, 1.0) is output


@requires_gpu
@pytest.mark.parametrize(
    "floor,temperature",
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
    ],
)
def test_attention_gate_rejects_invalid_parameters(floor, temperature):
    output = torch.empty((2, 32), device="cuda")
    with pytest.raises(ValueError, match="finite"):
        attention_gate_mul(
            output,
            torch.empty_like(output),
            torch.zeros(2, device="cuda"),
            floor,
            temperature,
        )


def test_attention_gate_rejects_cpu():
    output = torch.empty((2, 32))
    with pytest.raises(ValueError, match="same GPU"):
        attention_gate_mul(output, torch.empty_like(output), torch.zeros(2), 0.0, 1.0)


@requires_gpu
@pytest.mark.parametrize(
    "invalid",
    [
        "output_rank",
        "output_stride",
        "output_dtype",
        "gate_rank",
        "gate_dtype",
        "gate_shape",
        "gate_stride",
        "gate_device",
        "bias_shape",
        "bias_dtype",
        "bias_stride",
        "bias_device",
        "bias_empty",
        "bias_width",
    ],
)
def test_attention_gate_rejects_invalid_buffers(invalid):
    output = torch.empty((7, 96), device="cuda")
    gate = torch.empty_like(output)
    bias = torch.zeros(3, device="cuda")
    if invalid == "output_rank":
        output = output.flatten()
    elif invalid == "output_stride":
        output = torch.empty((7, 192), device="cuda")[:, ::2]
    elif invalid == "output_dtype":
        output = output.double()
    elif invalid == "gate_rank":
        gate = gate.view(7, 3, 4, 8)
    elif invalid == "gate_dtype":
        gate = gate.half()
    elif invalid == "gate_shape":
        gate = gate[:, :64]
    elif invalid == "gate_stride":
        gate = torch.empty((7, 192), device="cuda")[:, ::2]
    elif invalid == "gate_device":
        gate = gate.cpu()
    elif invalid == "bias_shape":
        bias = bias[None, :]
    elif invalid == "bias_dtype":
        bias = bias.half()
    elif invalid == "bias_stride":
        bias = torch.zeros(6, device="cuda")[::2]
    elif invalid == "bias_device":
        bias = bias.cpu()
    elif invalid == "bias_empty":
        bias = bias[:0]
    elif invalid == "bias_width":
        bias = torch.zeros(5, device="cuda")
    with pytest.raises((ValueError, TypeError)):
        attention_gate_mul(output, gate, bias, 0.0, 1.0)


@requires_gpu
@pytest.mark.parametrize("alias", ["gate", "bias", "disjoint_view"])
def test_attention_gate_rejects_shared_output_storage(alias):
    output = torch.empty((7, 96), device="cuda")
    gate = torch.empty_like(output)
    bias = torch.zeros(3, device="cuda")
    if alias == "gate":
        gate = output
    elif alias == "bias":
        bias = output[0, :3]
    else:
        storage = torch.empty((14, 96), device="cuda")
        output, gate = storage[:7], storage[7:]
    with pytest.raises(ValueError, match="share storage"):
        attention_gate_mul(output, gate, bias, 0.0, 1.0)
