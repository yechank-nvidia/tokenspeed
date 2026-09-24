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

"""The ``--kernel-override FAMILY.MODE=KERNEL_NAME`` runtime surface.

CPU: CLI parsing of the repeatable flag, table parsing errors, validation
against the real registry (unknown name, name of another operator,
``moe.apply``, a paged MHA/MLA or Inkling rel-MHA kernel outside the solution
the configured backend pins for that operator while GDN/KDA operators and
unpinned backends stay unconstrained, conflicting pre-set variable),
all-or-nothing environment mirroring that ``select_kernel`` honours, the
sorted ready-dict representation, the cross-rank agreement check and the
launcher's DP/encode gate. A drift guard derives the set of solution-pinned
operators from the runtime and kernel sources and holds the rule table to it.
Without a GPU the platform singleton and the Triton driver are stubbed
before the kernel package is imported so its registration decorators can run;
no kernel is launched here. The TP2 ready-dict proof needs CUDA.
"""

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import os
import sys
import textwrap
import types
from collections.abc import Iterator
from pathlib import Path
from test.ci_system.ci_register import register_cuda_ci
from unittest import mock

import pytest
import torch

register_cuda_ci(est_time=180, suite="runtime-2gpu")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _install_cpu_kernel_package_stubs() -> None:
    """Let the kernel package import and register kernels on a CPU-only host.

    ``tokenspeed_kernel/__init__.py`` calls ``current_platform()`` while it
    imports and the GDN Triton helpers ask the Triton driver for the current
    target at import time; both raise without a device. The platform module is
    loaded under a placeholder package first so the override is in place
    before the real package initialises, then a Triton driver that answers the
    target query with a Blackwell target is installed.
    """
    if "tokenspeed_kernel" in sys.modules:
        return
    spec = importlib.util.find_spec("tokenspeed_kernel")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("tokenspeed_kernel is not importable", allow_module_level=True)
    placeholder = types.ModuleType("tokenspeed_kernel")
    placeholder.__path__ = list(spec.submodule_search_locations)
    sys.modules["tokenspeed_kernel"] = placeholder
    try:
        import tokenspeed_kernel.platform as platform_module
    finally:
        del sys.modules["tokenspeed_kernel"]
    platform_module.Platform.override(
        platform_module.PlatformInfo(
            vendor="nvidia",
            arch_version=platform_module.ArchVersion(10, 0),
            device_name="NVIDIA B200 (CPU test stub)",
            device_count=1,
            total_memory=180 * (1024**3),
            memory_bandwidth=8000.0,
            sm_count=148,
            max_threads_per_sm=2048,
            max_shared_memory_per_sm=232448,
            sm_features=frozenset(
                {
                    "tensor_core:f16",
                    "tensor_core:bf16",
                    "tensor_core:int8",
                    "tensor_core:f8",
                    "tensor_core:f4",
                    "memory:async_copy",
                    "memory:tma",
                    "compute:cluster",
                }
            ),
            runtime_features=frozenset({"runtime:cuda_graph"}),
            interconnect=platform_module.InterconnectInfo(topology="single_gpu"),
        )
    )
    import tokenspeed_triton
    from tokenspeed_triton.backends.compiler import GPUTarget

    class CpuHostTritonDriver:
        """Answers the import-time target query; nothing here launches."""

        def get_current_target(self):
            return GPUTarget("cuda", 100, 32)

        def get_current_device(self):
            return 0

        def get_active_torch_device(self):
            return torch.device("cpu")

    tokenspeed_triton.runtime.driver.set_active(CpuHostTritonDriver())


if not torch.cuda.is_available():
    _install_cpu_kernel_package_stubs()

from tokenspeed_kernel.platform import current_platform  # noqa: E402
from tokenspeed_kernel.registry import (  # noqa: E402
    KernelRegistry,
    load_builtin_kernels,
)
from tokenspeed_kernel.selection import select_kernel  # noqa: E402
from tokenspeed_kernel.signature import (  # noqa: E402
    dense_tensor_format,
    format_signature,
)

import tokenspeed.runtime.utils.server_args as server_args_module  # noqa: E402
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    KERNEL_OVERRIDE_SOLUTION_RULES,
    MHA_KERNEL_SOLUTION_BY_BACKEND,
    MLA_KERNEL_SOLUTION_BY_BACKEND,
    ServerArgs,
    assert_kernel_override_env,
    check_kernel_override_tables_agree,
    kernel_override_env_key,
    kernel_override_lines,
    parse_kernel_overrides,
    pinned_kernel_solution,
    prepare_server_args,
)

# Registered kernels used as test data (names, not runtime constants).
GEMV = ("gemm", "decode_gemv")
GEMV_KERNEL = "torch_decode_gemv"
GEMV_KEY = "TOKENSPEED_KERNEL_OVERRIDE_GEMM_DECODE_GEMV"
MM_KERNEL = "torch_mm"  # registered under gemm.mm
MHA_DECODE = ("attention", "mha_decode_with_kvcache")
TRITON_MHA_DECODE = "triton_mha_decode_with_kvcache"
MHA_PREFILL = ("attention", "mha_prefill")
TRITON_MHA_PREFILL = "triton_mha_prefill"
MLA_DECODE = ("attention", "mla_decode_with_kvcache")
TRITON_MLA_DECODE = "triton_mla_decode_with_kvcache"
MLA_PREFILL = ("attention", "mla_prefill")
TRITON_MLA_PREFILL = "triton_mla_prefill"
# Inkling's relative-position MHA operators, dispatched with the wrapped MHA
# leaf's pin (backends/specific/inkling.py).
REL_MHA_DECODE = ("attention", "rel_mha_decode_with_kvcache")
TRITON_REL_MHA_DECODE = "triton_rel_mha_decode_with_kvcache"
REL_MHA_PREFILL = ("attention", "rel_mha_prefill")
TRITON_REL_MHA_PREFILL = "triton_rel_mha_prefill"
REL_MHA_EXTEND = ("attention", "rel_mha_extend_with_kvcache")
FA4_REL_MHA_EXTEND = "fa4_rel_mha_extend_with_kvcache"
GDN_DECODE = ("attention", "gdn_decode_step")
TRITON_GDN_DECODE = "triton_gdn_decode_step"
KDA_PREFILL = ("attention", "kda_paged_prefill")
TRITON_KDA_PREFILL = "triton_nvidia_kda_paged_prefill"
MOE_APPLY_KERNEL = "triton_bf16_precomputed_moe_apply"


@pytest.fixture(scope="module")
def registry() -> KernelRegistry:
    load_builtin_kernels()
    return KernelRegistry.get()


OVERRIDE_ENV_PREFIX = "TOKENSPEED_KERNEL_OVERRIDE_"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """No ``TOKENSPEED_KERNEL_OVERRIDE_*`` variable before the test and none left after.

    ``validate_kernel_overrides`` writes ``os.environ`` directly, so every key the test
    added is popped here; ``monkeypatch`` restores only what it changed itself (the
    pre-existing keys deleted below come back through it).
    """
    for key in [k for k in os.environ if k.startswith(OVERRIDE_ENV_PREFIX)]:
        monkeypatch.delenv(key)
    yield monkeypatch
    for key in [k for k in os.environ if k.startswith(OVERRIDE_ENV_PREFIX)]:
        os.environ.pop(key)


def _server_args(
    kernel_override: list[str] | None, attention_backend: str | None
) -> ServerArgs:
    """A ServerArgs carrying only what the override validator reads."""
    args = object.__new__(ServerArgs)
    args.kernel_override = kernel_override
    args.attention_backend = attention_backend
    return args


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser


def _prepare(argv: list[str]) -> ServerArgs:
    """Run the full ``__post_init__`` chain without touching a device."""
    with (
        mock.patch.object(
            server_args_module, "get_nvgpu_memory_capacity", return_value=180_000
        ),
        mock.patch.object(
            server_args_module, "get_amdgpu_memory_capacity", return_value=180_000
        ),
        mock.patch.object(server_args_module, "detect_topology", return_value=None),
    ):
        return prepare_server_args(["--model", "x", *argv])


def test_test_data_is_registered(registry: KernelRegistry):
    for name, operator in (
        (GEMV_KERNEL, GEMV),
        (MM_KERNEL, ("gemm", "mm")),
        (TRITON_MHA_DECODE, MHA_DECODE),
        (TRITON_MHA_PREFILL, MHA_PREFILL),
        (TRITON_MLA_DECODE, MLA_DECODE),
        (TRITON_MLA_PREFILL, MLA_PREFILL),
        (TRITON_REL_MHA_DECODE, REL_MHA_DECODE),
        (TRITON_REL_MHA_PREFILL, REL_MHA_PREFILL),
        (FA4_REL_MHA_EXTEND, REL_MHA_EXTEND),
        (TRITON_GDN_DECODE, GDN_DECODE),
        (TRITON_KDA_PREFILL, KDA_PREFILL),
        (MOE_APPLY_KERNEL, ("moe", "apply")),
    ):
        spec = registry.get_by_name(name)
        assert spec is not None, name
        assert (spec.family, spec.mode) == operator


def test_cli_parses_repeated_kernel_override():
    parsed = _parser().parse_args(
        [
            "--model",
            "x",
            "--kernel-override",
            f"gemm.decode_gemv={GEMV_KERNEL}",
            "--kernel-override",
            f"attention.mha_decode_with_kvcache={TRITON_MHA_DECODE}",
        ]
    )
    assert parsed.kernel_override == [
        f"gemm.decode_gemv={GEMV_KERNEL}",
        f"attention.mha_decode_with_kvcache={TRITON_MHA_DECODE}",
    ]
    assert _parser().parse_args(["--model", "x"]).kernel_override is None
    assert ServerArgs.kernel_override is None


@pytest.mark.parametrize(
    "entry",
    [
        "gemm.decode_gemv",
        f"gemm={GEMV_KERNEL}",
        "gemm.decode_gemv=",
        f".decode_gemv={GEMV_KERNEL}",
        f"gemm.={GEMV_KERNEL}",
        f"={GEMV_KERNEL}",
        "",
    ],
)
def test_parse_rejects_malformed_entries(entry: str):
    with pytest.raises(ValueError, match="FAMILY.MODE=KERNEL_NAME"):
        parse_kernel_overrides([entry])


def test_parse_rejects_duplicate_operator():
    with pytest.raises(ValueError, match="gemm.decode_gemv twice"):
        parse_kernel_overrides(
            [f"gemm.decode_gemv={GEMV_KERNEL}", "gemm.decode_gemv=other"]
        )


def test_parse_table_and_none():
    assert parse_kernel_overrides(None) == {}
    assert parse_kernel_overrides([]) == {}
    table = parse_kernel_overrides(
        [f"gemm.decode_gemv={GEMV_KERNEL}", f"attention.mha_decode_with_kvcache=k"]
    )
    assert table == {GEMV: GEMV_KERNEL, MHA_DECODE: "k"}


def test_resolve_kernel_backends_fails_fast_on_malformed_table():
    args = object.__new__(ServerArgs)
    args.dense_gemm_backend = "auto"
    args.sampling_backend = "greedy"
    args.kernel_override = ["gemm.decode_gemv"]
    with pytest.raises(ValueError, match="FAMILY.MODE=KERNEL_NAME"):
        args.resolve_kernel_backends()
    args.kernel_override = [f"gemm.decode_gemv={GEMV_KERNEL}"]
    args.resolve_kernel_backends()
    assert args.kernel_override_table() == {GEMV: GEMV_KERNEL}


def test_empty_table_touches_nothing(clean_env: pytest.MonkeyPatch):
    _server_args(None, None).validate_kernel_overrides()
    assert not [k for k in os.environ if k.startswith("TOKENSPEED_KERNEL_OVERRIDE_")]


def test_unknown_name_lists_valid_names(
    registry: KernelRegistry, clean_env: pytest.MonkeyPatch
):
    with pytest.raises(ValueError, match="no kernel named 'no_such_kernel'") as info:
        _server_args(
            ["gemm.decode_gemv=no_such_kernel"], None
        ).validate_kernel_overrides()
    assert GEMV_KERNEL in str(info.value)
    assert GEMV_KEY not in os.environ


def test_unknown_operator_reports_none_registered(
    registry: KernelRegistry, clean_env: pytest.MonkeyPatch
):
    with pytest.raises(ValueError, match=r"\(none registered\)"):
        _server_args(["gemm.no_such_mode=x"], None).validate_kernel_overrides()


def test_cross_operator_name_rejected(
    registry: KernelRegistry, clean_env: pytest.MonkeyPatch
):
    with pytest.raises(ValueError, match="registered under gemm.mm") as info:
        _server_args(
            [f"gemm.decode_gemv={MM_KERNEL}"], None
        ).validate_kernel_overrides()
    assert GEMV_KERNEL in str(info.value)
    assert GEMV_KEY not in os.environ


def test_moe_apply_rejected(registry: KernelRegistry, clean_env: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="moe.apply cannot be overridden"):
        _server_args(
            [f"moe.apply={MOE_APPLY_KERNEL}"], None
        ).validate_kernel_overrides()
    assert "TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY" not in os.environ


@pytest.mark.parametrize(
    ("operator", "name"),
    [(MHA_DECODE, TRITON_MHA_DECODE), (REL_MHA_DECODE, TRITON_REL_MHA_DECODE)],
)
def test_attention_override_must_match_backend_solution(
    registry: KernelRegistry,
    clean_env: pytest.MonkeyPatch,
    operator: tuple[str, str],
    name: str,
):
    """A triton kernel is refused under fa4 and accepted where triton is pinned or nothing is."""
    entry = [_entry(operator, name)]
    key = kernel_override_env_key(*operator)
    with pytest.raises(ValueError, match="conflicts with --attention-backend fa4"):
        _server_args(entry, "fa4").validate_kernel_overrides()
    assert key not in os.environ
    for backend in ("triton", "mha", None):
        # Popped directly: ``monkeypatch.delenv`` would record the value just
        # written and restore it at teardown, leaking the override into later
        # test files.
        os.environ.pop(key, None)
        _server_args(entry, backend).validate_kernel_overrides()
        assert os.environ[key] == name


def _entry(operator: tuple[str, str], name: str) -> str:
    family, mode = operator
    return f"{family}.{mode}={name}"


@pytest.mark.parametrize(
    ("operator", "name", "backend"),
    [
        # Hybrid GDN/KDA models name an MHA/MLA backend for their full-attention
        # layers; the linear-attention operators are dispatched without a
        # pinned solution and must pass.
        (GDN_DECODE, TRITON_GDN_DECODE, "fa4"),
        (KDA_PREFILL, TRITON_KDA_PREFILL, "flashinfer"),
        (GDN_DECODE, TRITON_GDN_DECODE, "gluon"),
        # An MLA operator is not constrained by an MHA backend and vice versa;
        # Inkling's rel-MHA operators follow the MHA leaf, not an MLA backend.
        (MLA_DECODE, TRITON_MLA_DECODE, "fa4"),
        (MHA_DECODE, TRITON_MHA_DECODE, "mla"),
        (REL_MHA_PREFILL, TRITON_REL_MHA_PREFILL, "mla"),
        # A backend that leaves the choice to selection, or one outside the
        # paged MHA/MLA leaves (trtllm and flashmla call their kernels
        # directly), pins nothing.
        (MLA_DECODE, TRITON_MLA_DECODE, "mla"),
        (MLA_DECODE, TRITON_MLA_DECODE, None),
        (MHA_DECODE, TRITON_MHA_DECODE, "trtllm"),
        (MLA_PREFILL, TRITON_MLA_PREFILL, "flashmla"),
        (REL_MHA_DECODE, TRITON_REL_MHA_DECODE, "mha"),
        (REL_MHA_DECODE, TRITON_REL_MHA_DECODE, None),
        (REL_MHA_EXTEND, FA4_REL_MHA_EXTEND, "mha"),
    ],
)
def test_solution_rule_accepts_operators_the_backend_does_not_pin(
    registry: KernelRegistry,
    clean_env: pytest.MonkeyPatch,
    operator: tuple[str, str],
    name: str,
    backend: str | None,
):
    assert pinned_kernel_solution(*operator, backend) is None
    _server_args([_entry(operator, name)], backend).validate_kernel_overrides()
    assert os.environ[kernel_override_env_key(*operator)] == name


@pytest.mark.parametrize(
    ("operator", "name", "backend", "solution"),
    [
        (MHA_DECODE, TRITON_MHA_DECODE, "flashinfer", "flashinfer"),
        (MHA_PREFILL, TRITON_MHA_PREFILL, "fa4", "fa4"),
        (MLA_DECODE, TRITON_MLA_DECODE, "gluon", "gluon"),
        (MLA_PREFILL, TRITON_MLA_PREFILL, "gluon", "gluon"),
        # Inkling rel-MHA: the override would otherwise outrank the fa4 pin the
        # wrapper passes at dispatch (backends/specific/inkling.py).
        (REL_MHA_DECODE, TRITON_REL_MHA_DECODE, "fa4", "fa4"),
        (REL_MHA_PREFILL, TRITON_REL_MHA_PREFILL, "fa4", "fa4"),
        (REL_MHA_EXTEND, FA4_REL_MHA_EXTEND, "triton", "triton"),
    ],
)
def test_solution_rule_rejects_pinned_operator_outside_backend_solution(
    registry: KernelRegistry,
    clean_env: pytest.MonkeyPatch,
    operator: tuple[str, str],
    name: str,
    backend: str,
    solution: str,
):
    assert pinned_kernel_solution(*operator, backend) == solution
    with pytest.raises(
        ValueError, match=f"conflicts with --attention-backend {backend}"
    ) as info:
        _server_args([_entry(operator, name)], backend).validate_kernel_overrides()
    assert f"(solution {solution!r})" in str(info.value)
    assert kernel_override_env_key(*operator) not in os.environ


def test_solution_maps_match_the_paged_leaves():
    """The maps mirror what the paged leaves pin; drift between them is a bug."""
    from tokenspeed.runtime.layers.attention import registry as attention_registry
    from tokenspeed.runtime.layers.attention.backends.paged import mha, mla

    assert mha._KERNEL_SOLUTION_BY_BACKEND == MHA_KERNEL_SOLUTION_BY_BACKEND

    def registered_names(backend_cls) -> set[str]:
        return {
            name
            for name, (_, cls) in attention_registry._BACKEND_REGISTRY.items()
            if cls is backend_cls
        }

    assert registered_names(mha.MHAAttnBackend) == set(MHA_KERNEL_SOLUTION_BY_BACKEND)
    assert registered_names(mla.MLAAttnBackend) == set(MLA_KERNEL_SOLUTION_BY_BACKEND)


# The backend -> solution map each dispatching module's operators are ruled by:
# the map of the leaf whose ``kernel_solution`` the module reads. The paged
# leaves read their own; the Inkling wrapper reads
# ``self.inner._leaf_for(layer).kernel_solution``, and its inner backend is the
# dense MHA path (attention/registry.py: Inkling stays on the MHA path). A
# module that starts passing ``solution=<leaf>.kernel_solution`` fails the drift
# guard until it is classified here.
PINNED_LEAF_MAP_BY_DISPATCH_MODULE: dict[str, dict[str, str | None]] = {
    "tokenspeed.runtime.layers.attention.backends.paged.mha": (
        MHA_KERNEL_SOLUTION_BY_BACKEND
    ),
    "tokenspeed.runtime.layers.attention.backends.paged.mla": (
        MLA_KERNEL_SOLUTION_BY_BACKEND
    ),
    "tokenspeed.runtime.layers.attention.backends.specific.inkling": (
        MHA_KERNEL_SOLUTION_BY_BACKEND
    ),
}


def _kernel_solution_dispatch_sites() -> list[tuple[str, str, int]]:
    """``(module, wrapper, line)`` for every ``solution=<leaf>.kernel_solution`` call.

    An AST walk over every module under ``tokenspeed.runtime``: the paged
    leaves store the solution ``--attention-backend`` pins in
    ``kernel_solution`` and nothing else in the runtime does, so a call whose
    ``solution=`` argument reads that attribute is a pinned dispatch. The callee
    is the kernel-package wrapper it names (through ``functools.partial`` for
    the plan helpers the leaves bind once).
    """
    root = Path(server_args_module.__file__).resolve().parents[1]
    sites: list[tuple[str, str, int]] = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.Call) or not any(
                keyword.arg == "solution"
                and any(
                    isinstance(leaf, ast.Attribute) and leaf.attr == "kernel_solution"
                    for leaf in ast.walk(keyword.value)
                )
                for keyword in node.keywords
            ):
                continue
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id == "partial":
                callee = node.args[0]
            assert isinstance(callee, ast.Name), (
                f"{path}:{node.lineno}: pinned dispatch through "
                f"{ast.unparse(callee)!r}; the drift guard only follows a plain "
                "imported wrapper name"
            )
            parts = path.relative_to(root).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module = ".".join(("tokenspeed", "runtime", *parts))
            sites.append((module, callee.id, node.lineno))
    return sites


def _operators_selected_with_solution(wrapper) -> set[tuple[str, str]]:
    """The ``(family, mode)`` operators ``wrapper`` hands to ``select_kernel`` with ``solution=``.

    ``select_kernel`` is the one entry point that consults the
    ``TOKENSPEED_KERNEL_OVERRIDE_*`` variable (and lets it outrank
    ``solution=``); the registry planning queries the plan helpers make do not.
    The mode is a string literal or a name bound to string literals in the
    wrapper body (``dispatch_mode`` in ``mla_decode_with_kvcache``).
    """
    function = ast.parse(textwrap.dedent(inspect.getsource(wrapper))).body[0]
    bound: dict[str, set[str]] = {}
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            bound.setdefault(node.targets[0].id, set()).update(
                leaf.value
                for leaf in ast.walk(node.value)
                if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str)
            )

    def literals(argument: ast.expr) -> set[str]:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return {argument.value}
        if isinstance(argument, ast.Name) and bound.get(argument.id):
            return bound[argument.id]
        raise AssertionError(
            f"{wrapper.__module__}.{wrapper.__name__}: select_kernel operator "
            f"{ast.unparse(argument)!r} is not a string literal the drift guard "
            "can resolve"
        )

    operators: set[tuple[str, str]] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = (
            callee.id
            if isinstance(callee, ast.Name)
            else callee.attr if isinstance(callee, ast.Attribute) else None
        )
        if name != "select_kernel" or not any(
            keyword.arg == "solution" for keyword in node.keywords
        ):
            continue
        arguments = dict(enumerate(node.args))
        arguments.update(
            (("family", "mode").index(keyword.arg), keyword.value)
            for keyword in node.keywords
            if keyword.arg in ("family", "mode")
        )
        for family in literals(arguments[0]):
            for mode in literals(arguments[1]):
                operators.add((family, mode))
    return operators


def test_solution_rules_cover_every_pinned_dispatch_site():
    """The rule table is exactly the set of operators the leaves pin.

    Derived from the source on both sides: the runtime call sites that pass
    ``solution=<leaf>.kernel_solution`` name the kernel wrappers, and the
    wrappers' ``select_kernel(..., solution=)`` calls name the operators. A new
    pinned dispatch without a rule, a stale rule, or a rule keyed to the wrong
    leaf map fails here instead of letting an override outrank the pin.
    """
    sites = _kernel_solution_dispatch_sites()
    assert sites, "no solution=<leaf>.kernel_solution dispatch found; scan is broken"
    modules = {module for module, _, _ in sites}
    assert modules == set(
        PINNED_LEAF_MAP_BY_DISPATCH_MODULE
    ), "classify the leaf map of every module dispatching with kernel_solution"
    pinned: dict[tuple[str, str], set[str]] = {}
    for module, wrapper_name, _ in sites:
        wrapper = getattr(importlib.import_module(module), wrapper_name)
        for operator in _operators_selected_with_solution(wrapper):
            pinned.setdefault(operator, set()).add(module)
    assert set(pinned) == set(KERNEL_OVERRIDE_SOLUTION_RULES), sorted(
        set(pinned) ^ set(KERNEL_OVERRIDE_SOLUTION_RULES)
    )
    for operator, dispatching in pinned.items():
        for module in dispatching:
            assert (
                KERNEL_OVERRIDE_SOLUTION_RULES[operator]
                is PINNED_LEAF_MAP_BY_DISPATCH_MODULE[module]
            ), (operator, module)
    # Every inkling rel-MHA operator is reached only through the wrapper.
    inkling = "tokenspeed.runtime.layers.attention.backends.specific.inkling"
    assert {op for op, mods in pinned.items() if inkling in mods} == {
        REL_MHA_PREFILL,
        REL_MHA_EXTEND,
        REL_MHA_DECODE,
    }


def test_conflicting_preset_env_rejected(
    registry: KernelRegistry, clean_env: pytest.MonkeyPatch
):
    entry = [f"gemm.decode_gemv={GEMV_KERNEL}"]
    clean_env.setenv(GEMV_KEY, "something_else")
    with pytest.raises(ValueError, match=GEMV_KEY):
        _server_args(entry, None).validate_kernel_overrides()
    assert os.environ[GEMV_KEY] == "something_else"
    # The same value, or an empty one (which select_kernel treats as unset),
    # is not a conflict.
    clean_env.setenv(GEMV_KEY, GEMV_KERNEL)
    _server_args(entry, None).validate_kernel_overrides()
    clean_env.setenv(GEMV_KEY, "")
    _server_args(entry, None).validate_kernel_overrides()
    assert os.environ[GEMV_KEY] == GEMV_KERNEL


def test_validate_mirrors_env_that_select_kernel_honours(
    registry: KernelRegistry, clean_env: pytest.MonkeyPatch
):
    assert kernel_override_env_key(*GEMV) == GEMV_KEY
    _server_args([f"gemm.decode_gemv={GEMV_KERNEL}"], None).validate_kernel_overrides()
    assert os.environ[GEMV_KEY] == GEMV_KERNEL
    signature = format_signature(
        x=dense_tensor_format(torch.bfloat16),
        weight=dense_tensor_format(torch.bfloat16),
    )
    selected = select_kernel(*GEMV, signature, platform=current_platform())
    assert selected.name == GEMV_KERNEL


def test_rank_reassert_is_idempotent(clean_env: pytest.MonkeyPatch):
    table = {GEMV: GEMV_KERNEL, MHA_DECODE: TRITON_MHA_DECODE}
    assert_kernel_override_env(table)
    assert_kernel_override_env(table)
    assert os.environ[GEMV_KEY] == GEMV_KERNEL
    assert os.environ[kernel_override_env_key(*MHA_DECODE)] == TRITON_MHA_DECODE
    with pytest.raises(ValueError, match=GEMV_KEY):
        assert_kernel_override_env({GEMV: "other"})


def test_conflicting_table_writes_nothing(clean_env: pytest.MonkeyPatch):
    """A rejected table leaves the environment as it was and names every conflict."""
    mm_key = kernel_override_env_key("gemm", "mm")
    table = {GEMV: GEMV_KERNEL, ("gemm", "mm"): MM_KERNEL}
    clean_env.setenv(mm_key, "other")
    with pytest.raises(ValueError, match=mm_key):
        assert_kernel_override_env(table)
    assert GEMV_KEY not in os.environ
    assert os.environ[mm_key] == "other"
    clean_env.setenv(GEMV_KEY, "another")
    with pytest.raises(ValueError) as info:
        assert_kernel_override_env(table)
    assert GEMV_KEY in str(info.value) and mm_key in str(info.value)
    assert os.environ[GEMV_KEY] == "another"
    assert os.environ[mm_key] == "other"


def test_ready_dict_lines_are_sorted():
    table = {GEMV: GEMV_KERNEL, MHA_DECODE: TRITON_MHA_DECODE}
    lines = [
        f"attention.mha_decode_with_kvcache={TRITON_MHA_DECODE}",
        f"gemm.decode_gemv={GEMV_KERNEL}",
    ]
    assert kernel_override_lines(table) == lines
    assert kernel_override_lines(dict(reversed(list(table.items())))) == lines
    assert kernel_override_lines({}) == []


def test_cross_rank_tables_must_agree():
    lines = [f"gemm.decode_gemv={GEMV_KERNEL}"]
    check_kernel_override_tables_agree([lines, list(lines)])
    check_kernel_override_tables_agree([[], []])
    check_kernel_override_tables_agree([lines])
    with pytest.raises(RuntimeError, match="rank 1 reports"):
        check_kernel_override_tables_agree([lines, []])


def test_launcher_gate_compares_tp_ranks_only():
    """attn-DP and encode launches send ready dicts without the key (temporary gate)."""
    from tokenspeed.runtime.entrypoints.engine import (
        _check_ranks_agree_on_kernel_overrides,
    )

    def launch(has_dp: bool, disaggregation_mode: str):
        return types.SimpleNamespace(
            mapping=types.SimpleNamespace(attn=types.SimpleNamespace(has_dp=has_dp)),
            disaggregation_mode=disaggregation_mode,
        )

    lines = [f"gemm.decode_gemv={GEMV_KERNEL}"]
    without_key = [{"status": "ready"}]
    _check_ranks_agree_on_kernel_overrides(launch(True, "null"), without_key)
    _check_ranks_agree_on_kernel_overrides(launch(False, "encode"), without_key)
    _check_ranks_agree_on_kernel_overrides(
        launch(False, "null"),
        [{"kernel_overrides": lines}, {"kernel_overrides": list(lines)}],
    )
    with pytest.raises(RuntimeError, match="rank 1 reports"):
        _check_ranks_agree_on_kernel_overrides(
            launch(False, "prefill"),
            [{"kernel_overrides": lines}, {"kernel_overrides": []}],
        )


def test_prepare_server_args_end_to_end(
    registry: KernelRegistry, clean_env: pytest.MonkeyPatch
):
    args = _prepare(
        [
            "--kernel-override",
            f"gemm.decode_gemv={GEMV_KERNEL}",
            "--kernel-override",
            f"attention.mha_decode_with_kvcache={TRITON_MHA_DECODE}",
            "--attention-backend",
            "triton",
        ]
    )
    assert args.kernel_override == [
        f"gemm.decode_gemv={GEMV_KERNEL}",
        f"attention.mha_decode_with_kvcache={TRITON_MHA_DECODE}",
    ]
    assert args.kernel_override_table() == {
        GEMV: GEMV_KERNEL,
        MHA_DECODE: TRITON_MHA_DECODE,
    }
    assert kernel_override_lines(args.kernel_override_table()) == [
        f"attention.mha_decode_with_kvcache={TRITON_MHA_DECODE}",
        f"gemm.decode_gemv={GEMV_KERNEL}",
    ]
    assert os.environ[GEMV_KEY] == GEMV_KERNEL
    assert os.environ[kernel_override_env_key(*MHA_DECODE)] == TRITON_MHA_DECODE
    # The rank re-assert accepts the inherited table.
    assert_kernel_override_env(args.kernel_override_table())
    with pytest.raises(ValueError, match="conflicts with --attention-backend fa4"):
        _prepare(
            [
                "--kernel-override",
                f"attention.mha_decode_with_kvcache={TRITON_MHA_DECODE}",
                "--attention-backend",
                "fa4",
            ]
        )


def _write_tiny_llama_config(directory: Path) -> Path:
    """A two-layer ``LlamaForCausalLM`` config small enough for two ranks to
    load in seconds. Weights come from ``load_format="dummy"`` (random
    initialisation from the config), so no tensor file is written; the
    override proof reads ready dicts, never a token. A minimal word-level
    fast tokenizer is written next to the config because the scheduler loads
    the tokenizer unconditionally (``skip_tokenizer_init`` only affects the
    frontend)."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    directory.mkdir()
    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"tok{i}": i for i in range(3, 64)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "[UNK]",
                "model_max_length": 2048,
            }
        )
        + "\n"
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "torch_dtype": "bfloat16",
                "vocab_size": 1024,
                "hidden_size": 512,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "hidden_act": "silu",
                "max_position_embeddings": 2048,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "tie_word_embeddings": False,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        )
        + "\n"
    )
    return directory


@requires_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_tp2_ranks_report_the_same_kernel_overrides(
    tmp_path, clean_env: pytest.MonkeyPatch
):
    """Every rank re-asserts the table and the launcher proves they agree."""
    from tokenspeed.runtime.entrypoints.engine import Engine

    checkpoint = _write_tiny_llama_config(tmp_path / "tiny")
    entry = f"gemm.decode_gemv={GEMV_KERNEL}"
    engine = Engine(
        model=str(checkpoint),
        # The config directory carries no tensors: weights are randomly
        # initialised, and nothing is generated here.
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        attn_tp_size=2,
        kernel_override=[entry],
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.2,
        # Graph capture is irrelevant to the override table and its default
        # prefill capture batch sizes exceed max_num_seqs=4 (a ValueError).
        enforce_eager=True,
        disable_prefill_graph=True,
        log_level="info",
    )
    try:
        # The launcher's ready dict (the same object other engine tests read as
        # ``engine.scheduler_info``) is rank 0's re-asserted table after
        # _check_ranks_agree_on_kernel_overrides compared every rank's copy.
        assert engine.server_args.kernel_override == [entry]
        assert engine.scheduler_info["kernel_overrides"] == [entry]
        assert os.environ[GEMV_KEY] == GEMV_KERNEL
    finally:
        engine.shutdown()
