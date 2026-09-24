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

"""Guard the grouped-decode KV tile switch.

Covered: the explicit ``block_n`` parameter through the Triton paged-decode
host chain, the default tile derivation ``grouped_decode_block_n``, the
``triton_mha_decode_kv32`` switch handle (registration, priority band,
name-only reachability, shared host) and the public BF16 graph path.

CPU tests execute the host functions and the registrations from source with
the ``_function`` loader of ``test_mha_decode_workspace`` against a fresh
registry, so they run without importing optional GPU backends. GPU tests are
marked ``requires_cuda`` and restrict themselves to SM100, the only platform
on which the default derivation yields the 128-token tile.
"""

import ast
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_mha_decode_workspace import MHA, _function
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import (
    KernelRegistry,
    Priority,
    describe_kernel,
    register_kernel,
)
from tokenspeed_kernel.selection import kernel_override, select_kernel
from tokenspeed_kernel.signature import format_signatures

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

OPERATOR = ("attention", "mha_decode_with_kvcache")
DEFAULT_KERNEL = "triton_mha_decode_with_kvcache"
KV32_KERNEL = "triton_mha_decode_kv32"
OVERRIDE_ENV = "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_DECODE_WITH_KVCACHE"
DECODE = MHA / "_triton/decode.py"
TRITON = MHA / "triton.py"
# Functions that carry ``block_n`` from the registered kernel to the launch.
HOST_CHAIN = (
    "_decode_grouped_att_m_fwd",
    "decode_attention_fwd_grouped",
    "decode_attention_fwd",
    "_triton_mha_decode_with_kvcache_impl",
)

# (vendor, arch, (q, k, v) dtypes, head_dim_k, head_dim_v, page_size, expected tile)
TILE_MATRIX = [
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
]


def _fake_platform(vendor, arch):
    return SimpleNamespace(
        is_nvidia=vendor == "nvidia",
        is_amd=vendor == "amd",
        arch_version=ArchVersion(*arch),
    )


def _reject_cuda(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("CPU host test attempted CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", reject)


def _paged_caches(dtypes, dk, dv, page):
    """Facade-shaped q and [pages, page_size, kv_heads, head_dim] caches."""
    q = torch.empty((2, 6, dk), dtype=getattr(torch, dtypes[0]), device="cpu")
    k = torch.empty((3, page, 1, dk), dtype=getattr(torch, dtypes[1]), device="cpu")
    v = torch.empty((3, page, 1, dv), dtype=getattr(torch, dtypes[2]), device="cpu")
    return q, k, v


class _RecordingKernel:
    """Stand-in for the stage-1 Triton kernel: records ``kernel[grid](...)``."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def record(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return record


def _launcher(vendor, arch):
    kernel = _RecordingKernel()
    namespace = {
        "torch": torch,
        "ArchVersion": ArchVersion,
        "_MIN_BLOCK_KV": 32,
        "triton": SimpleNamespace(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
        "current_platform": lambda: _fake_platform(vendor, arch),
        "_fwd_grouped_kernel_stage1": kernel,
    }
    return _function(DECODE, "_decode_grouped_att_m_fwd", namespace), kernel


def _launch_arguments(dk, dv, page):
    q = torch.empty((2, 6, dk), dtype=torch.bfloat16, device="cpu")
    k = torch.empty((page * 3, 1, dk), dtype=torch.bfloat16, device="cpu")
    v = torch.empty((page * 3, 1, dv), dtype=torch.bfloat16, device="cpu")
    out = torch.empty((2, 6, 4, dv), dtype=torch.float32, device="cpu")
    lse = torch.empty((2, 6, 4), dtype=torch.float32, device="cpu")
    table = torch.empty((2, 3), dtype=torch.int32, device="cpu")
    lengths = torch.empty(2, dtype=torch.int32, device="cpu")
    splits = torch.ones(2, dtype=torch.int32, device="cpu")
    return (q, k, v, out, lse, table, lengths, splits, 4, 3, page, 1, -1, 0.125, 0.0)


@pytest.mark.parametrize("vendor,arch,dtypes,dk,dv,page,expected", TILE_MATRIX)
def test_default_tile_derivation(
    monkeypatch, vendor, arch, dtypes, dk, dv, page, expected
):
    _reject_cuda(monkeypatch)
    namespace = {
        "torch": torch,
        "ArchVersion": ArchVersion,
        "current_platform": lambda: _fake_platform(vendor, arch),
    }
    derive = _function(DECODE, "grouped_decode_block_n", namespace)
    tile = derive(*_paged_caches(dtypes, dk, dv, page))
    assert tile == expected
    # Every default the derivation yields satisfies the launcher's contract
    # (power of two >= 16) and reaches the kernel unchanged.
    launch, kernel = _launcher(vendor, arch)
    launch(*_launch_arguments(dk, dv, page), tile)
    assert [call[2]["BLOCK_N"] for call in kernel.calls] == [tile]


@pytest.mark.parametrize("block_n", [16, 32, 128])
@pytest.mark.parametrize(
    "vendor,arch,dk",
    [
        ("nvidia", (10, 0), 128),
        ("nvidia", (9, 0), 128),
        ("nvidia", (10, 0), 288),
        ("nvidia", (10, 0), 576),
        ("amd", (9, 5), 128),
        ("amd", (9, 5), 576),
    ],
)
def test_grouped_launcher_passes_explicit_tile(monkeypatch, vendor, arch, dk, block_n):
    """The launcher forwards ``block_n`` as BLOCK_N and derives nothing itself."""
    _reject_cuda(monkeypatch)
    launch, kernel = _launcher(vendor, arch)
    page, dv = 64, 128
    arguments = _launch_arguments(dk, dv, page)
    q, k, v, _, _, table, lengths, splits = arguments[:8]
    launch(*arguments, block_n)
    assert len(kernel.calls) == 1
    grid, args, kwargs = kernel.calls[0]
    assert grid == (2, 1, 4)
    assert all(actual is expected for actual, expected in zip(args[:3], (q, k, v)))
    assert args[4] is table and args[5] is lengths and args[8] is splits
    assert args[-2:] == (3, page)
    assert kwargs["BLOCK_N"] == block_n
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


@pytest.mark.parametrize("vendor,arch", [("nvidia", (10, 0)), ("amd", (9, 5))])
@pytest.mark.parametrize("block_n", [0, -32, 1, 2, 4, 8, 48, 96])
def test_grouped_launcher_rejects_invalid_tile(monkeypatch, vendor, arch, block_n):
    """Tiles that are not a power of two, or lie below the stage-1 ``tl.dot``
    floor of 16, are rejected by the host on every vendor before any launch."""
    _reject_cuda(monkeypatch)
    launch, kernel = _launcher(vendor, arch)
    with pytest.raises(ValueError, match="power of two >= 16"):
        launch(*_launch_arguments(128, 128, 64), block_n)
    assert not kernel.calls


def test_block_n_is_explicit_through_the_host_chain():
    """No default picks a tile anywhere; the guard lives only in the derivation."""
    tree = ast.parse(DECODE.read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    launcher = functions[HOST_CHAIN[0]]
    assert launcher.args.args[-1].arg == "block_n"
    assert launcher.args.defaults == []
    for name in HOST_CHAIN[1:]:
        arguments = functions[name].args
        index = next(
            index
            for index, argument in enumerate(arguments.kwonlyargs)
            if argument.arg == "block_n"
        )
        assert arguments.kw_defaults[index] is None
        assert arguments.kwarg is None
    names_in = lambda node: {  # noqa: E731
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    }
    assert "ArchVersion" not in names_in(launcher)
    assert "ArchVersion" in names_in(functions["grouped_decode_block_n"])
    launches = [
        call
        for call in ast.walk(launcher)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Subscript)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "_fwd_grouped_kernel_stage1"
    ]
    assert len(launches) == 1
    block_keyword = next(kw for kw in launches[0].keywords if kw.arg == "BLOCK_N")
    assert isinstance(block_keyword.value, ast.Name)
    assert block_keyword.value.id == "block_n"


def _facade_inputs():
    """Every keyword the public facade passes to a decode kernel."""
    return {
        "q": torch.empty((2, 4, 8), dtype=torch.bfloat16),
        "k_cache": torch.empty((2, 64, 2, 8), dtype=torch.bfloat16),
        "v_cache": torch.empty((2, 64, 2, 8), dtype=torch.bfloat16),
        "page_table": torch.tensor([[0], [1]], dtype=torch.int32),
        "cache_seqlens": torch.tensor([32, 64], dtype=torch.int32),
        "max_seqlen_k": 64,
        "max_seqlen_q": 1,
        "window_left": -1,
        "logit_cap": 0.0,
        "sinks": None,
        "return_lse": False,
        "softmax_scale": None,
        "q_scale": None,
        "k_scale": None,
        "v_scale": None,
        "enable_pdl": False,
    }


def test_impl_requires_and_forwards_block_n(monkeypatch):
    _reject_cuda(monkeypatch)
    launched = Mock()
    namespace = {
        "torch": torch,
        "math": math,
        "prepare_mha_decode_workspace": lambda batch, device: torch.ones(
            batch, dtype=torch.int32
        ),
        "decode_attention_fwd": launched,
    }
    impl = _function(DECODE, HOST_CHAIN[-1], namespace)
    inputs = _facade_inputs()
    with pytest.raises(TypeError, match="block_n"):
        impl(**inputs, decode_workspace=None)
    launched.assert_not_called()
    for block_n in (32, 128):
        out = impl(**inputs, block_n=block_n, decode_workspace=None)
        assert out.shape == inputs["q"].shape
        assert launched.call_args.kwargs["block_n"] == block_n


def _registrations(namespace):
    """Execute triton.py's module constants and the two decode registrations."""
    tree = ast.parse(TRITON.read_text())
    wanted = {DEFAULT_KERNEL, KV32_KERNEL}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or (isinstance(node, ast.FunctionDef) and node.name in wanted)
    ]
    assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == wanted
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(ast.Module(body=future + nodes, type_ignores=[]), str(TRITON), "exec"),
        namespace,
    )
    return namespace


@pytest.fixture
def registered(fresh_registry):
    impl = Mock(name="_triton_mha_decode_with_kvcache_impl", return_value=object())
    derive = Mock(name="grouped_decode_block_n", return_value=128)
    _registrations(
        {
            "torch": torch,
            "CapabilityRequirement": CapabilityRequirement,
            "format_signatures": format_signatures,
            "register_kernel": register_kernel,
            "Priority": Priority,
            "_triton_mha_decode_with_kvcache_impl": impl,
            "grouped_decode_block_n": derive,
        }
    )
    return SimpleNamespace(registry=KernelRegistry.get(), impl=impl, derive=derive)


def test_tile_switch_registrations(registered):
    registry = registered.registry
    specs = registry.list_kernels(*OPERATOR)
    # Registry order is priority-descending: the default first, the handle last.
    assert [spec.name for spec in specs] == [DEFAULT_KERNEL, KV32_KERNEL]
    default, kv32 = specs
    assert (default.family, default.mode) == (kv32.family, kv32.mode) == OPERATOR
    assert default.solution == kv32.solution == "triton"
    assert default.priority == int(Priority.PORTABLE)
    assert kv32.priority == int(Priority.REFERENCE)
    assert kv32.priority < default.priority
    assert kv32.format_signatures == default.format_signatures
    assert kv32.traits == default.traits
    assert default.capability.vendors == frozenset({"nvidia", "amd"})
    assert kv32.capability.vendors == frozenset({"nvidia"})
    for name in (DEFAULT_KERNEL, KV32_KERNEL):
        assert registry.get_impl(name).__name__ == name
    description = describe_kernel(KV32_KERNEL)
    assert f"Kernel: {KV32_KERNEL}" in description
    assert "Operator: attention.mha_decode_with_kvcache" in description
    assert "Solution: triton" in description
    assert "Priority: 0 (REFERENCE)" in description
    assert "Priority: 4 (PORTABLE)" in describe_kernel(DEFAULT_KERNEL)


def test_tile_switch_handles_share_the_host(registered):
    """Both registrations call the same host; only ``block_n`` differs."""
    inputs = _facade_inputs()
    workspace = torch.ones(2, dtype=torch.int32)
    default = registered.registry.get_impl(DEFAULT_KERNEL)
    kv32 = registered.registry.get_impl(KV32_KERNEL)

    result = default(**inputs, decode_workspace=workspace)
    assert result is registered.impl.return_value
    registered.derive.assert_called_once_with(
        inputs["q"], inputs["k_cache"], inputs["v_cache"]
    )
    forwarded = registered.impl.call_args.kwargs
    assert forwarded["block_n"] == registered.derive.return_value
    assert forwarded["decode_workspace"] is workspace
    assert all(forwarded[key] is value for key, value in inputs.items())

    registered.impl.reset_mock()
    registered.derive.reset_mock()
    result = kv32(**inputs, decode_workspace=workspace)
    assert result is registered.impl.return_value
    registered.derive.assert_not_called()
    pinned = registered.impl.call_args.kwargs
    assert pinned["block_n"] == 32
    assert {k: v for k, v in pinned.items() if k != "block_n"} == {
        k: v for k, v in forwarded.items() if k != "block_n"
    }
    with pytest.raises(TypeError, match="decode_workspace"):
        kv32(**inputs)


def test_kv32_is_never_auto_selected_and_switches_by_name(
    registered, b200_platform, monkeypatch
):
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)
    signature = next(
        iter(format_signatures(("q", "k_cache", "v_cache"), "dense", {torch.bfloat16}))
    )
    traits = {
        "q_len": 1,
        "head_dim": 128,
        "page_size": 64,
        "logit_cap": False,
        "return_lse": False,
        "sinks": False,
        "sliding_window": False,
    }

    def select(**kwargs):
        return select_kernel(
            *OPERATOR, signature, traits=traits, platform=b200_platform, **kwargs
        ).name

    assert select() == DEFAULT_KERNEL
    assert select(solution="triton") == DEFAULT_KERNEL
    assert select(override=KV32_KERNEL) == KV32_KERNEL
    with kernel_override(*OPERATOR, KV32_KERNEL):
        assert select() == KV32_KERNEL
    assert select() == DEFAULT_KERNEL
    monkeypatch.setenv(OVERRIDE_ENV, KV32_KERNEL)
    assert select() == KV32_KERNEL


@requires_cuda
def test_real_registry_exposes_both_tile_switch_names():
    from tokenspeed_kernel.registry import load_builtin_kernels

    load_builtin_kernels()
    registry = KernelRegistry.get()
    default = registry.get_by_name(DEFAULT_KERNEL)
    kv32 = registry.get_by_name(KV32_KERNEL)
    assert default is not None and kv32 is not None
    assert default.priority == int(Priority.PORTABLE)
    assert kv32.priority == int(Priority.REFERENCE)
    assert kv32.traits == default.traits
    assert kv32.format_signatures == default.format_signatures
    names = [spec.name for spec in registry.list_kernels(*OPERATOR)]
    assert names.index(DEFAULT_KERNEL) < names.index(KV32_KERNEL)
    assert "Priority: 0 (REFERENCE)" in describe_kernel(KV32_KERNEL)


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


def _decode_inputs(batch, qlen, heads, kvheads, length, seed):
    """Seeded CPU BF16 inputs: (q, k, v, page_table, cache_seqlens)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pages = (length + 63) // 64
    q = torch.randn(
        (batch * qlen, heads, 128), generator=generator, dtype=torch.bfloat16
    )
    k = torch.randn(
        (batch * pages, 64, kvheads, 128), generator=generator, dtype=torch.bfloat16
    )
    v = torch.randn(k.shape, generator=generator, dtype=torch.bfloat16)
    table = torch.randperm(
        batch * pages, generator=generator, dtype=torch.int32
    ).reshape(batch, pages)
    lengths = torch.tensor(
        [max(qlen, length - i) for i in range(batch)], dtype=torch.int32
    )
    return q, k, v, table, lengths


def _skip_unless_sm100():
    if torch.version.hip is not None:
        pytest.skip("KV tile regression requires NVIDIA CUDA")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Wide BF16 KV tile is restricted to SM100")


@requires_cuda
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
    _skip_unless_sm100()
    from tokenspeed_kernel.ops.attention.mha import (
        mha_decode_with_kvcache,
        prepare_mha_decode_workspace,
    )
    from tokenspeed_kernel.ops.attention.mha._triton import decode

    monkeypatch.delenv(OVERRIDE_ENV, raising=False)
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
    q_cpu, k_cpu, v_cpu, table_cpu, lengths_cpu = _decode_inputs(
        batch, qlen, heads, kvheads, length, seed=42
    )
    pages = table_cpu.shape[1]
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


@requires_cuda
@pytest.mark.parametrize(
    "batch,qlen,heads,kvheads,length,window",
    [
        (1, 1, 6, 1, 31, -1),
        (2, 4, 8, 2, 63, 511),
        (1, 1, 16, 2, 129, -1),
        (4, 1, 32, 4, 511, -1),
        (2, 4, 12, 2, 2049, -1),
        (1, 1, 6, 1, 8191, 511),
    ],
)
def test_kv32_switch_numerics_against_default_tile(
    monkeypatch, record_property, batch, qlen, heads, kvheads, length, window
):
    """kv32 vs kv128 on SM100: same facade call, tile switched by name only."""
    _skip_unless_sm100()
    from tokenspeed_kernel.ops.attention.mha import (
        mha_decode_with_kvcache,
        prepare_mha_decode_workspace,
    )
    from tokenspeed_kernel.ops.attention.mha._triton import decode

    monkeypatch.delenv(OVERRIDE_ENV, raising=False)
    kernel = decode._fwd_grouped_kernel_stage1
    block_index = kernel.arg_names.index("BLOCK_N")
    launches = []

    class Observer:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def call(*args, **kwargs):
                handle = launch(*args, **kwargs)
                assert handle.src.fn is kernel
                launches.append(
                    (int(kwargs["BLOCK_N"]), int(handle.src.constants[(block_index,)]))
                )
                return handle

            return call

    monkeypatch.setattr(decode, "_fwd_grouped_kernel_stage1", Observer())
    cpu_values = _decode_inputs(batch, qlen, heads, kvheads, length, seed=7)
    q, k, v, table, lengths = (value.to(device="cuda") for value in cpu_values)
    pages = table.shape[1]
    workspace = prepare_mha_decode_workspace(batch, q.device)

    def run(override):
        del launches[:]
        output = mha_decode_with_kvcache(
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
            override=override,
            solution="triton",
            decode_workspace=workspace,
        )
        torch.cuda.synchronize()
        assert launches, "stage-1 launch was not observed"
        return output, set(launches)

    default_output, default_tiles = run(None)
    kv32_output, kv32_tiles = run(KV32_KERNEL)
    assert default_tiles == {(128, 128)}
    assert kv32_tiles == {(32, 32)}
    expected = _reference(*cpu_values, qlen, window)
    for output in (default_output, kv32_output):
        assert output.shape == q.shape and output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.cpu().float(), expected, rtol=0.03, atol=0.03)
    torch.testing.assert_close(
        kv32_output.float(), default_output.float(), rtol=0.03, atol=0.03
    )
    # The context-manager switch reaches the same handle as the explicit override.
    with kernel_override(*OPERATOR, KV32_KERNEL):
        context_output, context_tiles = run(None)
    assert context_tiles == {(32, 32)}
    assert torch.equal(context_output.view(torch.uint8), kv32_output.view(torch.uint8))
    assert torch.equal(workspace, torch.ones_like(workspace))
    difference = (kv32_output.float() - default_output.float()).abs()
    record_property(
        "mha_decode_kv_tile_switch",
        {
            "default_kernel": DEFAULT_KERNEL,
            "switch_kernel": KV32_KERNEL,
            "default_block_n": 128,
            "switch_block_n": 32,
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
        },
    )
