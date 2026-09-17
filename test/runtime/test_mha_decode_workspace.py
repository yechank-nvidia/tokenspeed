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

"""CPU-tensor lifecycle checks for immutable, leaf-owned MHA decode workspace.

Importing the real runtime may still require a CUDA-capable vendor environment;
the tests themselves do not launch kernels or capture CUDA graphs.
"""

import importlib
import inspect
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.fixture
def mha_module():
    return importlib.import_module(
        "tokenspeed.runtime.layers.attention.backends.paged.mha"
    )


def _leaf(mha_module, *, block_decode, spec_width):
    config = SimpleNamespace(
        device="cpu",
        dtype=torch.bfloat16,
        is_draft=block_decode,
        speculative_num_draft_tokens=spec_width,
        context_len=2048,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_mxfp8=False,
        draft_block_decode=block_decode,
        max_bs=8,
        kernel_page_size=None,
    )
    spec = SimpleNamespace(
        num_attention_heads=6,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=128,
        backend_name="triton",
        skip_softmax_threshold=0.0,
    )
    return mha_module.MHAAttnBackend(config, spec, kernel_page_size=64)


def _inputs(leaf, *, bs, actual_bs):
    lengths = torch.full((bs,), 1151, dtype=torch.int32)
    lengths[actual_bs:] = 1
    table = torch.arange(1, bs * leaf.max_num_pages + 1, dtype=torch.int32).reshape(
        bs, leaf.max_num_pages
    )
    table[actual_bs:] = 0
    return lengths, table


def _refresh(leaf, *, bs, actual_bs, replay):
    lengths, table = _inputs(leaf, bs=bs, actual_bs=actual_bs)
    leaf.refresh_decode_metadata(
        bs,
        actual_bs,
        lengths,
        table,
        num_extends=0,
        for_graph_replay=replay,
    )
    return leaf.forward_decode_metadata


@pytest.mark.parametrize(
    "block_decode,spec_width,expansion", [(False, 1, 1), (False, 4, 1), (True, 4, 4)]
)
def test_workspace_owner_covers_full_expanded_capacity(
    mha_module, block_decode, spec_width, expansion
):
    leaf = _leaf(mha_module, block_decode=block_decode, spec_width=spec_width)
    assert leaf.decode_workspace_buf is None
    leaf.init_cuda_graph_state(max_bs=8)
    workspace = leaf.decode_workspace_buf
    assert workspace.shape == leaf.seq_lens_buf.shape == (8 * expansion,)
    assert workspace.dtype == torch.int32 and workspace.device.type == "cpu"
    assert workspace.tolist() == [1] * (8 * expansion)
    # Above a hypothetical capture ladder of two requests, eager uses the
    # same full-capacity owner and does not allocate a second metadata path.
    metadata = _refresh(leaf, bs=8, actual_bs=8, replay=False)
    assert metadata.decode_workspace.shape == (8 * expansion,)
    assert metadata.decode_workspace.data_ptr() == workspace.data_ptr()


@pytest.mark.parametrize("block_decode,spec_width", [(False, 1), (True, 4)])
def test_cached_workspace_views_survive_eager_capture_and_replay_refresh(
    mha_module, monkeypatch, block_decode, spec_width
):
    leaf = _leaf(mha_module, block_decode=block_decode, spec_width=spec_width)
    leaf.init_cuda_graph_state(max_bs=8)
    owner = leaf.decode_workspace_buf
    monkeypatch.setattr(
        mha_module,
        "prepare_mha_decode_workspace",
        Mock(side_effect=AssertionError("Refresh must not allocate workspace")),
    )
    small = _refresh(leaf, bs=2, actual_bs=2, replay=False)
    padded = _refresh(leaf, bs=4, actual_bs=2, replay=True)
    assert small is leaf._decode_views(2)
    assert padded is leaf._decode_views(4)
    assert small.decode_workspace.shape == (2 * leaf.block_decode_expansion,)
    assert padded.decode_workspace.shape == (4 * leaf.block_decode_expansion,)
    assert small.decode_workspace.data_ptr() == padded.decode_workspace.data_ptr()

    # Exercise the inherited capture-metadata hook, not an actual CUDA graph.
    lengths, table = _inputs(leaf, bs=4, actual_bs=0)
    leaf.init_forward_metadata_capture_cuda_graph(4, lengths, table)
    assert leaf.forward_decode_metadata is padded
    assert _refresh(leaf, bs=4, actual_bs=3, replay=True) is padded
    assert _refresh(leaf, bs=2, actual_bs=2, replay=False) is small
    assert leaf.decode_workspace_buf is owner
    assert owner.tolist() == [1] * owner.numel()


def test_pool_rebind_drops_owner_and_views_then_reinitializes(mha_module):
    leaf = _leaf(mha_module, block_decode=False, spec_width=1)
    first_pool, second_pool = object(), object()
    leaf.set_cache_pool(first_pool)
    leaf.init_cuda_graph_state(max_bs=8)
    old_owner = leaf.decode_workspace_buf
    old_metadata = _refresh(leaf, bs=2, actual_bs=2, replay=False)
    leaf.forward_extend_metadata = object()

    leaf.set_cache_pool(second_pool)
    assert leaf.cache_pool is second_pool
    assert leaf.decode_workspace_buf is None
    assert leaf.page_table_buf is None and leaf.seq_lens_buf is None
    assert leaf.forward_decode_metadata is None
    assert leaf.forward_extend_metadata is None
    assert leaf._decode_views_by_bs == {}

    leaf.init_cuda_graph_state(max_bs=8)
    new_metadata = _refresh(leaf, bs=2, actual_bs=2, replay=False)
    assert new_metadata is not old_metadata
    assert leaf.decode_workspace_buf.data_ptr() != old_owner.data_ptr()
    assert (
        new_metadata.decode_workspace.data_ptr() == leaf.decode_workspace_buf.data_ptr()
    )
    assert old_metadata.decode_workspace.data_ptr() == old_owner.data_ptr()
    assert leaf.decode_workspace_buf.tolist() == old_owner.tolist() == [1] * 8


@pytest.mark.parametrize("q_len", [1, 4])
def test_forward_passes_exact_metadata_workspace(mha_module, monkeypatch, q_len):
    leaf = _leaf(mha_module, block_decode=False, spec_width=q_len)
    leaf.init_cuda_graph_state(max_bs=8)
    metadata = _refresh(leaf, bs=2, actual_bs=2, replay=False)
    layer = SimpleNamespace(
        layer_id=0,
        qk_head_dim=128,
        v_head_dim=128,
        tp_q_head_num=6,
        tp_k_head_num=1,
        tp_v_head_num=1,
        sliding_window_size=511,
        logit_cap=0.0,
    )
    cache = torch.zeros(64, 1, 128, dtype=torch.bfloat16)
    pool = SimpleNamespace(
        get_key_buffer=lambda layer_id: cache,
        get_value_buffer=lambda layer_id: cache,
    )
    call = Mock(side_effect=lambda **kwargs: torch.zeros_like(kwargs["q"]))
    monkeypatch.setattr(mha_module, "mha_decode_with_kvcache", call)
    output = leaf.forward_decode(
        q=torch.zeros(2 * q_len, 6 * 128, dtype=torch.bfloat16),
        k=None,
        v=None,
        layer=layer,
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        token_to_kv_pool=pool,
        bs=2,
        save_kv_cache=False,
        sinks=None,
    )
    assert output.shape == (2 * q_len, 6 * 128)
    assert call.call_count == 1
    assert call.call_args.kwargs["decode_workspace"] is metadata.decode_workspace
    assert call.call_args.kwargs["max_seqlen_q"] == q_len
    assert call.call_args.kwargs["solution"] == "triton"
    assert call.call_args.kwargs["window_left"] == 511


def test_decode_metadata_requires_explicit_workspace(mha_module):
    field = inspect.signature(mha_module.MHADecodeMetadata).parameters[
        "decode_workspace"
    ]
    assert field.default is inspect.Parameter.empty
    kwargs = dict(
        page_table=torch.zeros(1, 1, dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
    )
    with pytest.raises(TypeError, match="decode_workspace"):
        mha_module.MHADecodeMetadata(**kwargs)
    assert (
        mha_module.MHADecodeMetadata(**kwargs, decode_workspace=None).decode_workspace
        is None
    )
