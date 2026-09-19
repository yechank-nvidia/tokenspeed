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

"""Guard the shared grouped-decode KV tile and its public BF16 graph path."""

from types import SimpleNamespace

import pytest
import torch
from test_mha_decode_workspace import MHA, _function
from tokenspeed_kernel.platform import ArchVersion


@pytest.mark.parametrize(
    "vendor,arch,dtypes,dk,dv,page,expected",
    [
        ("nvidia", (10, 0), ("bfloat16",) * 3, 128, 128, 64, 128),
        ("nvidia", (9, 0), ("bfloat16",) * 3, 128, 128, 64, 32),
        ("nvidia", (12, 0), ("bfloat16",) * 3, 128, 128, 64, 32),
        ("amd", (9, 5), ("bfloat16",) * 3, 128, 128, 64, 32),
        ("amd", (9, 5), ("bfloat16",) * 3, 576, 128, 64, 16),
        ("nvidia", (10, 0), ("float16",) * 3, 128, 128, 64, 32),
        ("nvidia", (10, 0), ("float8_e4m3fn",) * 3, 128, 128, 64, 32),
        ("nvidia", (10, 0), ("float8_e5m2",) * 3, 128, 128, 64, 32),
        ("nvidia", (10, 0), ("float16", "bfloat16", "bfloat16"), 128, 128, 64, 32),
        ("nvidia", (10, 0), ("bfloat16", "float16", "bfloat16"), 128, 128, 64, 32),
        ("nvidia", (10, 0), ("bfloat16", "bfloat16", "float16"), 128, 128, 64, 32),
        (
            "nvidia",
            (10, 0),
            ("bfloat16", "float8_e4m3fn", "float8_e4m3fn"),
            128,
            128,
            64,
            32,
        ),
        ("nvidia", (10, 0), ("bfloat16",) * 3, 64, 64, 64, 32),
        ("nvidia", (10, 0), ("bfloat16",) * 3, 128, 64, 64, 32),
        ("nvidia", (10, 0), ("bfloat16",) * 3, 288, 128, 64, 32),
        ("nvidia", (10, 0), ("bfloat16",) * 3, 576, 128, 64, 32),
        ("nvidia", (10, 0), ("bfloat16",) * 3, 128, 128, 16, 32),
        ("nvidia", (10, 0), ("bfloat16",) * 3, 128, 128, 128, 32),
    ],
)
def test_grouped_decode_tile_selection(
    monkeypatch, vendor, arch, dtypes, dk, dv, page, expected
):
    def reject_cuda(*args, **kwargs):
        raise AssertionError("Launcher-selection test attempted CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", reject_cuda)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def record(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return record

    # Reuse the host-test loader: execute the actual launcher body without
    # importing/registering optional GPU backends or changing its source.
    namespace = {
        "torch": torch,
        "ArchVersion": ArchVersion,
        "_MIN_BLOCK_KV": 32,
        "triton": SimpleNamespace(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
    }
    launch = _function(
        MHA / "_triton/decode.py", "_decode_grouped_att_m_fwd", namespace
    )
    platform = SimpleNamespace(
        is_nvidia=vendor == "nvidia",
        is_amd=vendor == "amd",
        arch_version=ArchVersion(*arch),
    )
    monkeypatch.setitem(namespace, "current_platform", lambda: platform)
    monkeypatch.setitem(namespace, "_fwd_grouped_kernel_stage1", Kernel())
    q = torch.empty((2, 6, dk), dtype=getattr(torch, dtypes[0]), device="cpu")
    k = torch.empty((page * 3, 1, dk), dtype=getattr(torch, dtypes[1]), device="cpu")
    v = torch.empty((page * 3, 1, dv), dtype=getattr(torch, dtypes[2]), device="cpu")
    out = torch.empty((2, 6, 4, dv), dtype=torch.float32, device="cpu")
    lse = torch.empty((2, 6, 4), dtype=torch.float32, device="cpu")
    table = torch.empty((2, 3), dtype=torch.int32, device="cpu")
    lengths = torch.empty(2, dtype=torch.int32, device="cpu")
    splits = torch.ones(2, dtype=torch.int32, device="cpu")
    launch(q, k, v, out, lse, table, lengths, splits, 4, 3, page, 1, -1, 0.125, 0.0)
    assert len(calls) == 1
    grid, args, kwargs = calls[0]
    assert grid == (2, 1, 4)
    assert all(actual is expected for actual, expected in zip(args[:3], (q, k, v)))
    assert args[4] is table and args[5] is lengths and args[8] is splits
    assert args[-2:] == (3, page)
    assert kwargs["BLOCK_N"] == expected
    assert kwargs["BLOCK_H"] == 16 and kwargs["MIN_BLOCK_KV"] == 32
    assert kwargs["num_warps"] == 4
    assert kwargs["num_stages"] == (1 if vendor == "amd" else 2)
    assert kwargs["BLOCK_DPE"] == {288: 32, 576: 64}.get(dk, 0)
    assert kwargs["BLOCK_DMODEL"] == {288: 256, 576: 512}.get(dk, dk)
    assert kwargs["BLOCK_DV"] == dv
    assert kwargs["Lk"] == dk and kwargs["Lv"] == dv
    if vendor == "amd":
        assert kwargs["waves_per_eu"] == 1 and kwargs["matrix_instr_nonkdim"] == 16
    else:
        assert "waves_per_eu" not in kwargs and "matrix_instr_nonkdim" not in kwargs


def _reference(q, k, v, table, lengths, qlen, window):
    """CPU FP32 SDPA with per-query causal/window bounds."""
    heads, dim = q.shape[1:]
    kvheads = k.shape[2]
    rows = []
    for batch, length in enumerate(lengths.tolist()):
        keys = k[table[batch].long()].reshape(-1, kvheads, dim)
        values = v[table[batch].long()].reshape(-1, kvheads, dim)
        for offset in range(qlen):
            row = batch * qlen + offset
            end = length - qlen + offset + 1
            start = max(0, end - window - 1) if window >= 0 else 0
            keys_i = keys[start:end].repeat_interleave(heads // kvheads, dim=1)
            values_i = values[start:end].repeat_interleave(heads // kvheads, dim=1)
            output = torch.nn.functional.scaled_dot_product_attention(
                q[row].float()[None, :, None, :],
                keys_i.float().permute(1, 0, 2)[None],
                values_i.float().permute(1, 0, 2)[None],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
                enable_gqa=False,
            )[0, :, 0]
            rows.append(output)
    return torch.stack(rows)


@pytest.mark.parametrize(
    "batch,qlen,heads,kvheads,length,window",
    [
        (1, 1, 6, 1, 1, -1),
        (1, 4, 6, 1, 4, -1),
        (1, 1, 6, 1, 31, -1),
        (2, 4, 8, 2, 63, 511),
        (1, 1, 6, 1, 64, 511),
        (2, 4, 12, 2, 65, -1),
        (1, 1, 16, 2, 127, -1),
        (2, 4, 16, 2, 128, 511),
        (1, 1, 24, 3, 129, 511),
        (4, 1, 32, 4, 511, -1),
        (2, 4, 32, 1, 512, 511),
        (1, 4, 6, 1, 513, -1),
        (1, 1, 6, 1, 1151, 511),
        (2, 4, 12, 2, 2049, -1),
        (1, 1, 6, 1, 4097, -1),
        (1, 4, 6, 1, 8191, 511),
    ],
)
def test_public_bf16_decode_tile_numerics_and_live_graph(
    monkeypatch, record_property, batch, qlen, heads, kvheads, length, window
):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("KV tile regression requires NVIDIA CUDA")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Wide BF16 KV tile is restricted to SM100")
    from tokenspeed_kernel.ops.attention.mha import (
        mha_decode_with_kvcache,
        prepare_mha_decode_workspace,
    )
    from tokenspeed_kernel.ops.attention.mha._triton import decode

    kernel = decode._fwd_grouped_kernel_stage1
    observed = []
    block_index = kernel.arg_names.index("BLOCK_N")

    class Observer:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def call(*args, **kwargs):
                handle = launch(*args, **kwargs)
                assert kwargs["BLOCK_N"] == 128
                assert handle.src.fn is kernel
                assert handle.src.constants[(block_index,)] == 128
                observed.append(handle)
                return handle

            return call

    monkeypatch.setattr(decode, "_fwd_grouped_kernel_stage1", Observer())
    generator = torch.Generator(device="cpu").manual_seed(42)
    pages = (length + 63) // 64
    q_cpu = torch.randn(
        (batch * qlen, heads, 128), generator=generator, dtype=torch.bfloat16
    )
    k_cpu = torch.randn(
        (batch * pages, 64, kvheads, 128), generator=generator, dtype=torch.bfloat16
    )
    v_cpu = torch.randn(k_cpu.shape, generator=generator, dtype=torch.bfloat16)
    table_cpu = torch.randperm(
        batch * pages, generator=generator, dtype=torch.int32
    ).reshape(batch, pages)
    lengths_cpu = torch.tensor(
        [max(qlen, length - i) for i in range(batch)], dtype=torch.int32
    )
    original = (q_cpu, k_cpu, v_cpu, table_cpu, lengths_cpu)
    changed = (
        -q_cpu,
        -k_cpu,
        -v_cpu,
        table_cpu.flip(1),
        (lengths_cpu - 17).clamp_min(qlen),
    )
    live = tuple(value.to(device="cuda") for value in original)
    q, k, v, table, lengths = live
    workspace = prepare_mha_decode_workspace(batch, q.device)
    pointers = tuple(value.data_ptr() for value in live)

    def call():
        return mha_decode_with_kvcache(
            q=q,
            k_cache=k,
            v_cache=v,
            page_table=table,
            cache_seqlens=lengths,
            max_seqlen_k=pages * 64,
            max_seqlen_q=qlen,
            window_left=window,
            logit_cap=0.0,
            sinks=None,
            return_lse=False,
            softmax_scale=None,
            q_scale=None,
            k_scale=None,
            v_scale=None,
            override=None,
            solution="triton",
            decode_workspace=workspace,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = call()
        output_pointer = captured.data_ptr()
        assert output_pointer not in pointers
        first_output = None
        for cpu_values in (original, changed, original):
            for value, cpu_value in zip(live, cpu_values):
                value.copy_(cpu_value)
            eager = call()
            graph.replay()
            torch.cuda.synchronize()
            assert eager.shape == captured.shape == q.shape
            assert eager.dtype == captured.dtype == torch.bfloat16
            assert torch.isfinite(eager).all() and torch.isfinite(captured).all()
            assert torch.equal(captured.view(torch.uint8), eager.view(torch.uint8))
            expected = _reference(*cpu_values, qlen, window)
            torch.testing.assert_close(
                eager.cpu().float(), expected, rtol=0.03, atol=0.03
            )
            assert captured.data_ptr() == output_pointer
            assert tuple(value.data_ptr() for value in live) == pointers
            for value, cpu_value in zip(live, cpu_values):
                assert torch.equal(
                    value.cpu().contiguous().view(torch.uint8),
                    cpu_value.contiguous().view(torch.uint8),
                )
            assert torch.equal(workspace, torch.ones_like(workspace))
            if first_output is None:
                first_output = eager.clone()
            elif cpu_values is changed:
                assert not torch.equal(eager, first_output)
            else:
                assert torch.equal(
                    eager.view(torch.uint8), first_output.view(torch.uint8)
                )
        assert observed
        record_property(
            "mha_decode_stage1",
            {
                "observed_launches": len(observed),
                "block_n": 128,
                "source_file": kernel.fn.__code__.co_filename,
                "handle_hashes": sorted({handle.hash for handle in observed}),
                "kernel_names": sorted({handle.metadata.name for handle in observed}),
            },
        )
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            graph.reset()
