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

"""``gemm.decode_gemv`` through ``select_kernel``: leaf table and override switch.

``decode_gemv`` used to pick its leaf with a private, lru-cached scan of the
registry. It now asks :func:`select_kernel`, which makes
``TOKENSPEED_KERNEL_OVERRIDE_GEMM_DECODE_GEMV`` (and :func:`kernel_override`)
the one switch between the specialized leaves and ``torch_decode_gemv``, for
the FP32 expert router and, through :func:`decode_gemv_routed`, for the
measured BF16 projection route alike.

The CPU half queries the built-in registry (populated by importing
``tokenspeed_kernel.ops.gemm``) with the conftest fixture platforms, so the
leaf table and the equivalence with the old scan are checked on hosts without
the GPU they describe. The GPU half observes the same switch from
``decode_gemv`` and the route predicate on device tensors.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import tokenspeed_kernel.ops.gemm  # noqa: F401  (registration side effects)
import torch
from tokenspeed_kernel.ops.gemm import routed_gemv
from tokenspeed_kernel.ops.gemm.routed_gemv import MEASURED_ROUTE, decode_gemv_routed
from tokenspeed_kernel.ops.gemm.triton_gemv import (
    TORCH_DECODE_GEMV,
    _select,
    decode_gemv,
    torch_decode_gemv,
    triton_rowcta_gemm_fp32,
    triton_rowcta_gemv,
)
from tokenspeed_kernel.platform import Platform, PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, Priority
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectionEvent,
    add_selection_listener,
    kernel_override,
    remove_selection_listener,
    select_kernel,
    spec_matches_shape_traits,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import (
    FormatSignature,
    dense_tensor_format,
    format_signature,
)

FAMILY, MODE = "gemm", "decode_gemv"
ENV_KEY = "TOKENSPEED_KERNEL_OVERRIDE_GEMM_DECODE_GEMV"
# Example FP32 router projection: N=256 experts over a K=5120 hidden state.
ROUTER_N, ROUTER_K = 256, 5120
ROWS = (1, 2, 8, 32, 33)  # one per registry leaf plus both edges of the 2..32 band
LEAF_ROWCTA_GEMV = "triton_rowcta_gemv"
LEAF_ROWCTA_GEMV_FP32 = "triton_rowcta_gemv_fp32"
LEAF_ROWCTA_GEMM_FP32 = "triton_rowcta_gemm_fp32"
IMPL_BY_LEAF = {
    LEAF_ROWCTA_GEMV: triton_rowcta_gemv,
    LEAF_ROWCTA_GEMV_FP32: triton_rowcta_gemv,
    LEAF_ROWCTA_GEMM_FP32: triton_rowcta_gemm_fp32,
    TORCH_DECODE_GEMV: torch_decode_gemv,
}
# One measured entry per routed backend (read from MEASURED_ROUTE).
ROUTE_SAMPLES = {
    (1, 7168, 512): "flashinfer_tgv_gemv",
    (2, 768, 1536): "cute_dsl_skinny_gemv",
    (3, 1792, 7168): "cute_dsl_ll_bf16_gemv",
    (8, 1792, 7168): "flashinfer_splitk_gemv",
}


def _sig(dtype: torch.dtype) -> FormatSignature:
    return format_signature(
        x=dense_tensor_format(dtype), weight=dense_tensor_format(dtype)
    )


def _expected_fp32_leaf(rows: int) -> str:
    if rows == 1:
        return LEAF_ROWCTA_GEMV_FP32
    if 2 <= rows <= 32:
        return LEAF_ROWCTA_GEMM_FP32
    return TORCH_DECODE_GEMV


def _expected_bf16_leaf(rows: int, n: int, k: int, platform: PlatformInfo) -> str:
    routed = (
        platform.vendor == "nvidia"
        and platform.arch_version >= routed_gemv._CAPABILITY.min_arch_version
    )
    if routed and (rows, n, k) in MEASURED_ROUTE:
        suffix = f"_m{rows}_n{n}_k{k}"
        names = [
            spec.name
            for spec in KernelRegistry.get().list_kernels(FAMILY, MODE)
            if spec.name.endswith(suffix)
        ]
        assert len(names) == 1, names
        return names[0]
    if rows == 1 and n >= 128 and k >= 128:
        return LEAF_ROWCTA_GEMV
    return TORCH_DECODE_GEMV


def _legacy_scan(m: int, n: int, k: int, dtype: torch.dtype, platform: PlatformInfo):
    """The scan ``_select`` performed before it delegated to ``select_kernel``:
    first spec in registry (priority) order passing the platform, signature
    and both trait filters, else the portable leaf by fall-through."""
    traits = {"m": m, "n": n, "k": k}
    for spec in KernelRegistry.get().get_for_operator(
        FAMILY, MODE, platform=platform, format_signature=_sig(dtype)
    ):
        if spec_matches_traits(spec, traits) and spec_matches_shape_traits(
            spec, traits
        ):
            return spec.name
    return TORCH_DECODE_GEMV


@pytest.fixture
def platform_override() -> Iterator[None]:
    """Let a test pin ``current_platform()`` and restore the host afterwards."""
    saved = Platform._instance
    yield
    if saved is None:
        Platform.reset()
    else:
        Platform.override(saved)


@pytest.fixture
def selection_events() -> Iterator[list[SelectionEvent]]:
    events: list[SelectionEvent] = []
    add_selection_listener(events.append)
    yield events
    remove_selection_listener(events.append)


def test_torch_leaf_is_registered_for_bf16_and_fp32() -> None:
    spec = KernelRegistry.get().get_by_name(TORCH_DECODE_GEMV)
    assert spec is not None
    assert (spec.family, spec.mode, spec.solution) == (FAMILY, MODE, "torch")
    assert spec.supports_format_signature(_sig(torch.bfloat16))
    assert spec.supports_format_signature(_sig(torch.float32))
    assert not spec.supports_format_signature(_sig(torch.float16))
    assert spec.traits == {}
    assert spec.priority == Priority.PORTABLE
    assert KernelRegistry.get().get_impl(TORCH_DECODE_GEMV) is torch_decode_gemv


@pytest.mark.parametrize("platform_name", ["h100_platform", "b200_platform"])
@pytest.mark.parametrize("rows", ROWS)
def test_fp32_router_leaf_table_through_select_kernel(
    rows: int, platform_name: str, request: pytest.FixtureRequest
) -> None:
    platform = request.getfixturevalue(platform_name)
    selected = select_kernel(
        FAMILY,
        MODE,
        _sig(torch.float32),
        platform=platform,
        traits={"m": rows, "n": ROUTER_N, "k": ROUTER_K},
    )
    expected = _expected_fp32_leaf(rows)
    assert selected.name == expected
    assert selected.impl is IMPL_BY_LEAF[expected]


@pytest.mark.parametrize("platform_name", ["h100_platform", "b200_platform"])
@pytest.mark.parametrize("rows", ROWS)
def test_bf16_unlisted_leaf_table_through_select_kernel(
    rows: int, platform_name: str, request: pytest.FixtureRequest
) -> None:
    """The router shape is not in the measured route, so BF16 keeps the
    generic split: row-CTA at M == 1, the portable leaf above."""
    assert (rows, ROUTER_N, ROUTER_K) not in MEASURED_ROUTE
    platform = request.getfixturevalue(platform_name)
    selected = select_kernel(
        FAMILY,
        MODE,
        _sig(torch.bfloat16),
        platform=platform,
        traits={"m": rows, "n": ROUTER_N, "k": ROUTER_K},
    )
    expected = LEAF_ROWCTA_GEMV if rows == 1 else TORCH_DECODE_GEMV
    assert selected.name == expected
    assert selected.impl is IMPL_BY_LEAF[expected]


@pytest.mark.parametrize("shape,backend", sorted(ROUTE_SAMPLES.items()))
def test_bf16_measured_route_leaves_by_platform(
    shape: tuple[int, int, int],
    backend: str,
    h100_platform: PlatformInfo,
    b200_platform: PlatformInfo,
) -> None:
    """On sm100 the measured spec outranks the generic leaves; below it the
    capability gate filters the route out and the generic split remains."""
    assert shape in MEASURED_ROUTE
    m, n, k = shape
    traits = {"m": m, "n": n, "k": k}
    routed = select_kernel(
        FAMILY, MODE, _sig(torch.bfloat16), platform=b200_platform, traits=traits
    )
    assert routed.name == f"{backend}_m{m}_n{n}_k{k}"
    assert routed.name == _expected_bf16_leaf(m, n, k, b200_platform)
    generic = select_kernel(
        FAMILY, MODE, _sig(torch.bfloat16), platform=h100_platform, traits=traits
    )
    assert generic.name == (LEAF_ROWCTA_GEMV if m == 1 else TORCH_DECODE_GEMV)


def test_select_kernel_matches_the_legacy_scan_everywhere(
    h100_platform: PlatformInfo,
    b200_platform: PlatformInfo,
    b300_platform: PlatformInfo,
    mi450_platform: PlatformInfo,
) -> None:
    """Behaviour preservation: with no ``gemm`` oracle registered the ranking
    is priority order and the trait filters are the ones the scan applied, so
    every shape in the route table and around the leaf boundaries resolves to
    the same name on every platform."""
    shapes = set(MEASURED_ROUTE) | {
        (m, n, k)
        for m in (1, 2, 8, 32, 33, 64)
        for n, k in ((ROUTER_N, ROUTER_K), (999, 4096), (3216, 7168), (100, 100))
    }
    checked = 0
    for platform in (h100_platform, b200_platform, b300_platform, mi450_platform):
        for m, n, k in sorted(shapes):
            for dtype in (torch.bfloat16, torch.float32):
                selected = select_kernel(
                    FAMILY,
                    MODE,
                    _sig(dtype),
                    platform=platform,
                    traits={"m": m, "n": n, "k": k},
                )
                assert selected.name == _legacy_scan(m, n, k, dtype, platform), (
                    platform.device_name,
                    m,
                    n,
                    k,
                    dtype,
                )
                checked += 1
    assert checked >= 2 * 4 * len(MEASURED_ROUTE)


def test_select_follows_the_current_platform(
    platform_override: None, h100_platform: PlatformInfo, b200_platform: PlatformInfo
) -> None:
    m, n, k = 1, 7168, 512  # measured route entry
    Platform.override(h100_platform)
    assert _select(m, n, k, True, torch.bfloat16).name == LEAF_ROWCTA_GEMV
    assert _select(1, ROUTER_N, ROUTER_K, True, torch.float32).name == (
        LEAF_ROWCTA_GEMV_FP32
    )
    Platform.override(b200_platform)
    assert (
        _select(m, n, k, True, torch.bfloat16).name
        == "flashinfer_tgv_gemv_m1_n7168_k512"
    )
    for rows in ROWS:
        assert _select(rows, ROUTER_N, ROUTER_K, True, torch.float32).name == (
            _expected_fp32_leaf(rows)
        )


def test_non_cuda_inputs_stay_on_the_portable_leaf() -> None:
    # No platform is consulted, so this holds on a host without a GPU too.
    selected = _select(1, ROUTER_N, ROUTER_K, False, torch.float32)
    assert selected.name == TORCH_DECODE_GEMV
    assert selected.impl is torch_decode_gemv
    x = torch.ones(1, 8)
    weight = torch.ones(3, 8)
    torch.testing.assert_close(decode_gemv(x, weight), torch.full((1, 3), 8.0))


def test_unregistered_dtype_raises(
    platform_override: None, b200_platform: PlatformInfo
) -> None:
    Platform.override(b200_platform)
    with pytest.raises(NoKernelFoundError):
        _select(1, 999, 4096, True, torch.float16)


def test_select_has_no_private_cache(
    platform_override: None, b200_platform: PlatformInfo
) -> None:
    """A registration change is visible on the next call: nothing but the
    registry's own cache (invalidated on register) sits in front of it."""
    Platform.override(b200_platform)
    assert _select(1, ROUTER_N, ROUTER_K, True, torch.float32).name == (
        LEAF_ROWCTA_GEMV_FP32
    )
    registry = KernelRegistry.get()
    name = "test_plugin_router_gemv"
    registry.register(
        KernelSpec(
            name=name,
            family=FAMILY,
            mode=MODE,
            solution="test",
            format_signatures=frozenset({_sig(torch.float32)}),
            traits={"m": frozenset({1})},
            priority=Priority.PLUGIN,
        ),
        torch_decode_gemv,
    )
    try:
        assert _select(1, ROUTER_N, ROUTER_K, True, torch.float32).name == name
    finally:
        registry._unregister(name)
    assert _select(1, ROUTER_N, ROUTER_K, True, torch.float32).name == (
        LEAF_ROWCTA_GEMV_FP32
    )


def test_override_forces_the_torch_leaf_for_every_shape(
    platform_override: None,
    b200_platform: PlatformInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Platform.override(b200_platform)
    shapes = [(rows, ROUTER_N, ROUTER_K, torch.float32) for rows in ROWS]
    shapes += [(m, n, k, torch.bfloat16) for (m, n, k) in ROUTE_SAMPLES]

    with kernel_override(FAMILY, MODE, TORCH_DECODE_GEMV):
        for m, n, k, dtype in shapes:
            selected = _select(m, n, k, True, dtype)
            assert selected.name == TORCH_DECODE_GEMV, (m, n, k, dtype)
            assert selected.impl is torch_decode_gemv

    monkeypatch.setenv(ENV_KEY, TORCH_DECODE_GEMV)
    for m, n, k, dtype in shapes:
        assert _select(m, n, k, True, dtype).impl is torch_decode_gemv

    # Lifting the override is honoured on the very next call.
    monkeypatch.delenv(ENV_KEY)
    assert _select(1, ROUTER_N, ROUTER_K, True, torch.float32).name == (
        LEAF_ROWCTA_GEMV_FP32
    )
    assert _select(1, 7168, 512, True, torch.bfloat16).name == (
        "flashinfer_tgv_gemv_m1_n7168_k512"
    )


def test_override_is_witnessed_by_the_selection_listener(
    platform_override: None,
    b200_platform: PlatformInfo,
    selection_events: list[SelectionEvent],
) -> None:
    Platform.override(b200_platform)
    with kernel_override(FAMILY, MODE, TORCH_DECODE_GEMV):
        _select(1, ROUTER_N, ROUTER_K, True, torch.float32)
        _select(1, ROUTER_N, ROUTER_K, True, torch.float32)
    assert [
        (e.family, e.mode, e.source, e.kernel_name, e.override)
        for e in selection_events
    ] == [(FAMILY, MODE, "override", TORCH_DECODE_GEMV, TORCH_DECODE_GEMV)] * 2
    assert selection_events[0].format_signature == _sig(torch.float32)
    assert selection_events[0].platform_arch == b200_platform.arch

    selection_events.clear()
    _select(2, ROUTER_N, ROUTER_K, True, torch.float32)
    _select(2, ROUTER_N, ROUTER_K, True, torch.float32)
    assert [(e.source, e.kernel_name) for e in selection_events] == [
        ("ranked", LEAF_ROWCTA_GEMM_FP32),
        ("cache", LEAF_ROWCTA_GEMM_FP32),
    ]


def test_route_predicate_consults_the_registry_leaf(
    platform_override: None,
    b200_platform: PlatformInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route table admits a shape, the registry decides whether
    the measured leaf is what would run. Forcing the portable leaf answers
    False, so the linear layer keeps its ``torch_mm`` reference path."""
    Platform.override(b200_platform)
    # Behaviour preservation of the old unconditional True on sm100+: every
    # measured route spec must resolve to itself, never to the portable leaf.
    for m, n, k in MEASURED_ROUTE:
        assert routed_gemv._selects_measured_leaf(m, n, k, torch.bfloat16), (m, n, k)
    for m, n, k in ROUTE_SAMPLES:
        assert routed_gemv._selects_measured_leaf(m, n, k, torch.bfloat16)
        with kernel_override(FAMILY, MODE, TORCH_DECODE_GEMV):
            assert not routed_gemv._selects_measured_leaf(m, n, k, torch.bfloat16)
        monkeypatch.setenv(ENV_KEY, TORCH_DECODE_GEMV)
        assert not routed_gemv._selects_measured_leaf(m, n, k, torch.bfloat16)
        monkeypatch.delenv(ENV_KEY)
    # Unlisted shapes: the generic split, as before.
    assert routed_gemv._selects_measured_leaf(1, 999, 4096, torch.bfloat16)
    assert not routed_gemv._selects_measured_leaf(4, 3216, 7168, torch.bfloat16)


# --- GPU: the switch as the runtime sees it -------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _is_routed_arch() -> bool:
    from tokenspeed_kernel.platform import current_platform

    return (
        current_platform().vendor == "nvidia"
        and torch.cuda.get_device_capability() >= (10, 0)
    )


@requires_cuda
def test_decode_gemv_routed_flips_off_under_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _is_routed_arch():
        pytest.skip("route is registered for sm100 and up")
    m, n, k = 1, 7168, 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    assert decode_gemv_routed(x, weight)
    with kernel_override(FAMILY, MODE, TORCH_DECODE_GEMV):
        assert not decode_gemv_routed(x, weight)
    monkeypatch.setenv(ENV_KEY, TORCH_DECODE_GEMV)
    assert not decode_gemv_routed(x, weight)
    monkeypatch.delenv(ENV_KEY)
    assert decode_gemv_routed(x, weight)


@requires_cuda
@pytest.mark.parametrize("rows", ROWS)
def test_decode_gemv_override_runs_the_torch_leaf(
    rows: int, selection_events: list[SelectionEvent]
) -> None:
    torch.manual_seed(rows)
    x = torch.randn(rows, ROUTER_K, device="cuda", dtype=torch.float32)
    weight = torch.randn(ROUTER_N, ROUTER_K, device="cuda", dtype=torch.float32)
    reference = (x.double() @ weight.double().t()).float()

    ranked = decode_gemv(x, weight)
    assert selection_events[-1].kernel_name == _expected_fp32_leaf(rows)
    torch.testing.assert_close(ranked, reference, rtol=2e-5, atol=2e-4)

    with kernel_override(FAMILY, MODE, TORCH_DECODE_GEMV):
        forced = decode_gemv(x, weight)
    assert selection_events[-1].source == "override"
    assert selection_events[-1].kernel_name == TORCH_DECODE_GEMV
    # The forced leaf is torch's own matmul: bit-identical to it.
    torch.testing.assert_close(forced, x @ weight.t(), rtol=0, atol=0)


@requires_cuda
def test_env_override_is_honoured_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """No private cache pins the first decision: flipping the environment
    between two eager calls flips the leaf."""
    x = torch.randn(1, ROUTER_K, device="cuda", dtype=torch.float32)
    weight = torch.randn(ROUTER_N, ROUTER_K, device="cuda", dtype=torch.float32)
    assert _select(1, ROUTER_N, ROUTER_K, x.is_cuda, x.dtype).name == (
        LEAF_ROWCTA_GEMV_FP32
    )
    monkeypatch.setenv(ENV_KEY, TORCH_DECODE_GEMV)
    assert _select(1, ROUTER_N, ROUTER_K, x.is_cuda, x.dtype).name == (
        TORCH_DECODE_GEMV
    )
    torch.testing.assert_close(decode_gemv(x, weight), x @ weight.t(), rtol=0, atol=0)
    monkeypatch.delenv(ENV_KEY)
    assert _select(1, ROUTER_N, ROUTER_K, x.is_cuda, x.dtype).name == (
        LEAF_ROWCTA_GEMV_FP32
    )
