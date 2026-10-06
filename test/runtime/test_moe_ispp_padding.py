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

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.moe import expert as expert_mod
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(est_time=5, suite="runtime-1gpu")

# The alignment declared by the flashinfer_trtllm unquant MoE kernels
# (tokenspeed_kernel/ops/moe/flashinfer/trtllm_unquant.py, ispp_alignment).
_UNQUANT_ALIGNMENT = 128


@dataclass
class _SpecStub:
    intermediate_size: int
    gated: bool


def _padding_stub(intermediate_size: int, tp_size: int = 2, gated: bool = True):
    stub = SimpleNamespace(
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        prefix="model.layers.0.mlp.experts",
        _spec=_SpecStub(intermediate_size=intermediate_size, gated=gated),
    )
    apply = MoELayer._apply_trtllm_ispp_padding.__get__(stub)
    return stub, apply


def test_trtllm_ispp_padding_rounds_up_unaligned(monkeypatch):
    monkeypatch.setattr(
        expert_mod,
        "get_moe_backend",
        lambda: SimpleNamespace(value="flashinfer_trtllm"),
    )
    # ispp = 1000 -> padded to 1024 with the unquant kernel's 128 alignment.
    stub, apply = _padding_stub(intermediate_size=2000, tp_size=2)

    apply(_UNQUANT_ALIGNMENT, "test")

    assert stub.intermediate_size == 1024 * 2
    assert stub._spec.intermediate_size == 1024 * 2


def test_trtllm_ispp_padding_noop_when_aligned(monkeypatch):
    monkeypatch.setattr(
        expert_mod,
        "get_moe_backend",
        lambda: SimpleNamespace(value="flashinfer_trtllm"),
    )
    stub, apply = _padding_stub(intermediate_size=1024 * 2, tp_size=2)

    apply(_UNQUANT_ALIGNMENT, "test")

    assert stub.intermediate_size == 1024 * 2
    assert stub._spec.intermediate_size == 1024 * 2


def test_trtllm_ispp_padding_noop_for_other_backends(monkeypatch):
    monkeypatch.setattr(
        expert_mod,
        "get_moe_backend",
        lambda: SimpleNamespace(value="auto"),
    )
    stub, apply = _padding_stub(intermediate_size=2000, tp_size=2)

    apply(_UNQUANT_ALIGNMENT, "test")

    assert stub.intermediate_size == 2000
    assert stub._spec.intermediate_size == 2000


def test_auto_backend_pads_non_gated_experts_for_trtllm(monkeypatch):
    monkeypatch.setattr(
        expert_mod, "get_moe_backend", lambda: SimpleNamespace(value="auto")
    )
    # Nemotron-3 Super at TP2: ispp = 2688 / 2 = 1344 -> 1408, the relu2 kernels' alignment.
    relu2, apply_relu2 = _padding_stub(intermediate_size=2688, tp_size=2, gated=False)
    swiglu, apply_swiglu = _padding_stub(intermediate_size=2688, tp_size=2)

    apply_relu2(_UNQUANT_ALIGNMENT, "test")
    apply_swiglu(_UNQUANT_ALIGNMENT, "test")

    assert relu2.intermediate_size == 1408 * 2
    assert relu2._spec.intermediate_size == 1408 * 2
    assert swiglu.intermediate_size == 2688


# 768 intermediate channels at TP4: 192 (1.5 FP8 blocks) per rank.
_FP8_BLOCK, _FP8_TP, _FP8_INTERMEDIATE, _FP8_HIDDEN = 128, 4, 768, 256


def _fp8_block_layer(monkeypatch, backend: str, tp_rank: int):
    plans = []

    def fake_plan(weight_dtype, **kwargs):
        plans.append(kwargs)
        return {"solution": "fake", "apply_kernel_name": "fake"}

    monkeypatch.setattr(
        expert_mod, "get_moe_backend", lambda: SimpleNamespace(value=backend)
    )
    monkeypatch.setattr(expert_mod.tokenspeed_kernel, "moe_plan", fake_plan)
    monkeypatch.setitem(global_server_args_dict, "moe_mxfp4_fp8_activation", False)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    layer = MoELayer(
        top_k=1,
        num_experts=1,
        hidden_size=_FP8_HIDDEN,
        intermediate_size=_FP8_INTERMEDIATE,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[_FP8_BLOCK, _FP8_BLOCK],
        ),
        layer_index=0,
        prefix="model.layers.0.mlp",
        tp_rank=tp_rank,
        tp_size=_FP8_TP,
    )
    (plan,) = plans
    return layer, plan


def _channel_ids(shape: tuple[int, int], dim: int) -> torch.Tensor:
    """FP8 weights whose first two entries across ``dim`` spell channel + 1.

    Digits stay below 100: finite, non-negative E4M3 bytes.
    """
    ids = torch.arange(1, shape[dim] + 1)
    data = torch.zeros(shape, dtype=torch.uint8)
    index = [slice(None), slice(None)]
    for position, digit in enumerate((ids % 100, ids // 100)):
        index[1 - dim] = position
        data[tuple(index)] = digit.to(torch.uint8)
    return data.view(torch.float8_e4m3fn)


def _loaded_channels(weight: torch.Tensor, dim: int) -> torch.Tensor:
    """Checkpoint channels held by the leading slots of ``dim`` (zero tail)."""
    data = weight.view(torch.uint8).to(torch.int64)
    ids = data.select(1 - dim, 0) + 100 * data.select(1 - dim, 1) - 1
    return ids[ids >= 0]


@pytest.mark.parametrize(
    "backend", ["auto", "flashinfer_trtllm", "flashinfer_cutlass", "triton"]
)
def test_fp8_block_scales_follow_each_ranks_channels(monkeypatch, backend):
    """Every channel a rank holds reads its checkpoint block's scale, and the
    ranks hold every channel once, whichever backend runs the layer."""
    blocks, hidden_blocks = (
        _FP8_INTERMEDIATE // _FP8_BLOCK,
        _FP8_HIDDEN // _FP8_BLOCK,
    )
    # Checkpoint tensors of one expert: distinct scales per 128x128 block.
    w1_scale = torch.arange(1.0, 1 + blocks * hidden_blocks).view(blocks, -1)
    w3_scale = w1_scale + 1000
    w2_scale = -w1_scale.T.contiguous()
    loaded, ispps = [], []
    for tp_rank in range(_FP8_TP):
        layer, plan = _fp8_block_layer(monkeypatch, backend, tp_rank)
        w13, w13_scale = layer.w13_weight, layer.w13_weight_scale_inv
        w2, w2_scale_inv = layer.w2_weight, layer.w2_weight_scale_inv
        shape = (_FP8_INTERMEDIATE, _FP8_HIDDEN)
        for shard_id, scale in (("w1", w1_scale), ("w3", w3_scale)):
            w13.weight_loader(w13, _channel_ids(shape, 0), shard_id, 0)
            w13_scale.weight_loader(w13_scale, scale, shard_id, 0)
        w2.weight_loader(w2, _channel_ids(shape[::-1], 1), "w2", 0)
        w2_scale_inv.weight_loader(w2_scale_inv, w2_scale, "w2", 0)

        half = w13.shape[1] // 2
        half_blocks = w13_scale.shape[1] // 2
        for offset, block_offset, scale in (
            (0, 0, w1_scale),
            (half, half_blocks, w3_scale),
        ):
            channels = _loaded_channels(w13.data[0, offset : offset + half], 0)
            local = torch.arange(len(channels))
            assert torch.equal(
                w13_scale.data[0, block_offset + local // _FP8_BLOCK],
                scale[channels // _FP8_BLOCK],
            )
        channels = _loaded_channels(w2.data[0], 1)
        local = torch.arange(len(channels))
        assert torch.equal(
            w2_scale_inv.data[0][:, local // _FP8_BLOCK],
            w2_scale[:, channels // _FP8_BLOCK],
        )
        loaded.append(channels)
        ispps.append((plan["ispp"], layer.intermediate_size // _FP8_TP))
    # The ranks hold every checkpoint channel once, padded to whole blocks.
    assert torch.equal(torch.cat(loaded), torch.arange(_FP8_INTERMEDIATE))
    assert ispps == [(256, 256)] * _FP8_TP


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
