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

"""``sampling.topk_topp_renorm`` as a registry operator.

CPU contracts of the kernel-package glue with fake kernels registered in the
real registry -- selection by name (``override=``, ``kernel_override()``, the
``TOKENSPEED_KERNEL_OVERRIDE_SAMPLING_TOPK_TOPP_RENORM`` mirror), the ranked
default, the deprecated ``TS_DISABLE_FUSED_TOPK_TOPP=1`` alias and its one-time
warning, the TP broadcast rule read from the selected spec, the pre-capture
prepare hook -- plus the FlashInfer sampling backend's init and verify entry
points executed from source, and GPU checks of the two real arms (skipped on
CPU).

On a host without a GPU ``tokenspeed_kernel/__init__.py`` cannot import
(platform detection and every op family need a device) while the registry,
selection, signature and sampling operator modules can, so those are loaded
under a stub package with an overridden platform. The runtime backend module
is not importable there either; its two entry points run from source through
the AST loader ``_load`` below.
"""

from __future__ import annotations

import ast
import copy
import logging
import os
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "python/tokenspeed/runtime"
KERNEL_PACKAGE = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel"
BACKEND_SOURCE = ROOT / "python/tokenspeed/runtime/sampling/backends/flashinfer.py"
FAMILY, MODE = "sampling", "topk_topp_renorm"
OVERRIDE_ENV = "TOKENSPEED_KERNEL_OVERRIDE_SAMPLING_TOPK_TOPP_RENORM"
LEGACY_ENV = "TS_DISABLE_FUSED_TOPK_TOPP"
FUSED = "fused_topk_topp_renorm"
FLASHINFER = "flashinfer_topk_topp_renorm"
FAKE_FUSED = "fake_fused_topk_topp_renorm"
FAKE_FLASHINFER = "fake_flashinfer_topk_topp_renorm"
FAKE_UNTRAITED = "fake_untraited_topk_topp_renorm"
FAKE_OTHER_OPERATOR = "fake_other_operator_renorm"
OTHER_MODE = "other_operator_for_topk_topp_renorm_test"
TOP_K_DISABLED = 1 << 30

requires_flashinfer_kernels = pytest.mark.skipif(
    not torch.cuda.is_available() or bool(torch.version.hip),
    reason="the fused and FlashInfer renorm kernels require CUDA (not ROCm)",
)


def _load(namespace, relative, names, class_name=None, bases=()):
    """Execute selected functions (or methods of one class) of a runtime source file.

    ``relative`` is a path under ``RUNTIME``; ``names`` are the function
    names to compile. With ``class_name`` the named methods are compiled as
    a class of that name whose bases are the names in ``bases``, so a
    method body can run without importing the module (and its GPU-only
    dependencies).
    """
    path = RUNTIME / relative
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        original = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        methods = [
            copy.deepcopy(n)
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert {n.name for n in methods} == set(names)
        cls = copy.deepcopy(original)
        cls.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
        cls.keywords, cls.decorator_list, cls.body = [], [], methods
        selected = [cls]
    else:
        selected = [
            copy.deepcopy(n)
            for n in body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert {n.name for n in selected} == set(names)
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def _cpu_test_platform(platform_module):
    return platform_module.PlatformInfo(
        vendor="cpu-test",
        arch_version=platform_module.ArchVersion(0, 0),
        device_name="cpu-test",
        device_count=1,
        total_memory=0,
        memory_bandwidth=0.0,
        sm_count=0,
        max_threads_per_sm=0,
        max_shared_memory_per_sm=0,
        interconnect=platform_module.InterconnectInfo(topology="single_gpu"),
    )


@pytest.fixture(scope="module")
def kernel():
    """The kernel-package modules under test.

    Real imports on a GPU host; on a CPU host a stub ``tokenspeed_kernel``
    package (real submodules, no package ``__init__``) with a non-NVIDIA test
    platform, so the two real arms stay unregistered and only fakes exist.
    """
    before = set(sys.modules)
    installed_stub = False
    if not torch.cuda.is_available() and "tokenspeed_kernel" not in sys.modules:
        stub = types.ModuleType("tokenspeed_kernel")
        stub.__path__ = [str(KERNEL_PACKAGE)]
        sys.modules["tokenspeed_kernel"] = stub
        installed_stub = True
    import tokenspeed_kernel.platform as platform_module

    if installed_stub:
        platform_module.Platform.override(_cpu_test_platform(platform_module))
    import tokenspeed_kernel.ops.sampling as sampling
    import tokenspeed_kernel.ops.sampling.cuda as sampling_cuda
    import tokenspeed_kernel.ops.sampling.flashinfer as sampling_flashinfer
    from tokenspeed_kernel.registry import (
        KernelRegistry,
        Priority,
        error_fn,
        register_kernel,
    )
    from tokenspeed_kernel.selection import (
        NoKernelFoundError,
        kernel_override,
        select_kernel,
    )
    from tokenspeed_kernel.signature import format_signatures

    yield NS(
        sampling=sampling,
        sampling_cuda=sampling_cuda,
        sampling_flashinfer=sampling_flashinfer,
        KernelRegistry=KernelRegistry,
        Priority=Priority,
        error_fn=error_fn,
        register_kernel=register_kernel,
        NoKernelFoundError=NoKernelFoundError,
        kernel_override=kernel_override,
        select_kernel=select_kernel,
        format_signatures=format_signatures,
        platform=platform_module,
    )
    if installed_stub:
        platform_module.Platform.reset()
        for name in set(sys.modules) - before:
            if name.startswith("tokenspeed_kernel"):
                sys.modules.pop(name)


@pytest.fixture
def fakes(kernel):
    """Three fake kernels under the operator; yields their call log.

    ``FAKE_FUSED`` outranks every real registration so the ranked default is
    deterministic on GPU hosts too; ``FAKE_UNTRAITED`` omits the broadcast
    trait to exercise the declared-trait requirement.
    """
    calls: list[tuple[str, tuple]] = []

    def make(name: str, priority: int, rank_deterministic: bool | None) -> None:
        def impl(probs, top_ks, top_ps):
            calls.append((name, (probs, top_ks, top_ps)))
            return probs.clone()

        impl.__name__ = name
        traits = (
            None
            if rank_deterministic is None
            else {"rank_deterministic": frozenset({rank_deterministic})}
        )
        kernel.register_kernel(
            FAMILY,
            MODE,
            name=name,
            solution="fake",
            signatures=kernel.format_signatures("probs", "dense", {torch.float32}),
            traits=traits,
            priority=priority,
        )(impl)

    make(FAKE_FUSED, kernel.Priority.PERFORMANT + 3, True)
    make(FAKE_FLASHINFER, kernel.Priority.PORTABLE + 3, False)
    make(FAKE_UNTRAITED, kernel.Priority.REFERENCE, None)
    yield calls
    registry = kernel.KernelRegistry.get()
    for name in (FAKE_FUSED, FAKE_FLASHINFER, FAKE_UNTRAITED):
        registry._unregister(name)


@pytest.fixture
def other_operator_kernel(kernel):
    """A fake registered under another (family, mode) with the same signature.

    ``select_kernel`` resolves an override by name alone, so only the
    operator's own check keeps this kernel from being called with
    ``(probs, top_ks, top_ps)``.
    """

    def impl(probs, top_ks, top_ps):
        raise AssertionError("a kernel of another operator must never be called")

    impl.__name__ = FAKE_OTHER_OPERATOR
    kernel.register_kernel(
        FAMILY,
        OTHER_MODE,
        name=FAKE_OTHER_OPERATOR,
        solution="fake",
        signatures=kernel.format_signatures("probs", "dense", {torch.float32}),
        traits={"rank_deterministic": frozenset({True})},
        priority=kernel.Priority.REFERENCE,
    )(impl)
    yield FAKE_OTHER_OPERATOR
    kernel.KernelRegistry.get()._unregister(FAKE_OTHER_OPERATOR)


@pytest.fixture
def clean_env(monkeypatch):
    """Neither override surface set; restores whatever the resolver mirrored."""
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(LEGACY_ENV, raising=False)
    yield
    os.environ.pop(OVERRIDE_ENV, None)


@pytest.fixture
def alias_state(kernel, clean_env, monkeypatch):
    monkeypatch.setattr(kernel.sampling, "_alias_warned", False)


def _resolve(kernel, environ):
    """Resolve the alias while recording DeprecationWarnings regardless of filters."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = kernel.sampling.resolve_topk_topp_renorm_override(environ)
    return result, [w for w in caught if issubclass(w.category, DeprecationWarning)]


def _select(kernel, override):
    return kernel.sampling.select_topk_topp_renorm(
        probs_dtype=torch.float32, solution=None, override=override
    )


# --------------------------------------------------------------------------
# Registry facts
# --------------------------------------------------------------------------


def test_operator_constants_match_the_registry_and_selection_keys(kernel):
    sampling = kernel.sampling
    assert (sampling.TOPK_TOPP_RENORM_FAMILY, sampling.TOPK_TOPP_RENORM_MODE) == (
        FAMILY,
        MODE,
    )
    assert sampling.TOPK_TOPP_RENORM_OVERRIDE_ENV == OVERRIDE_ENV
    assert sampling.DISABLE_FUSED_TOPK_TOPP_ENV == LEGACY_ENV
    assert kernel.sampling_cuda.FUSED_TOPK_TOPP_RENORM == FUSED
    assert kernel.sampling_flashinfer.FLASHINFER_TOPK_TOPP_RENORM == FLASHINFER
    # The fused kernel is the only arm with pre-capture device state.
    assert sampling._TOPK_TOPP_RENORM_PREPARE == {
        FUSED: kernel.sampling_cuda.fused_topk_topp_prepare
    }


@pytest.mark.parametrize(
    "name, attribute, module_name, expected",
    [
        (FUSED, "fused_topk_topp_renorm", "sampling_cuda", True),
        (FLASHINFER, "flashinfer_topk_topp_renorm", "sampling_flashinfer", False),
    ],
)
def test_real_arms_are_registered_iff_their_wrapper_imported(
    kernel, name, attribute, module_name, expected
):
    """Registered arms carry the operator key and one broadcast-trait value;
    an arm whose import failed is unregistered, never error_fn-backed."""
    module_fn = getattr(getattr(kernel, module_name), attribute)
    spec = kernel.KernelRegistry.get().get_by_name(name)
    if module_fn is kernel.error_fn:
        assert spec is None
        with pytest.raises(kernel.NoKernelFoundError):
            _select(kernel, name)
        return
    assert (spec.family, spec.mode) == (FAMILY, MODE)
    assert spec.capability.vendors == frozenset({"nvidia"})
    assert kernel.KernelRegistry.get().get_impl(name) is module_fn
    assert kernel.sampling.topk_topp_renorm_rank_deterministic(name) is expected


# --------------------------------------------------------------------------
# Selection by name and the broadcast rule per spec
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, rank_deterministic", [(FAKE_FUSED, True), (FAKE_FLASHINFER, False)]
)
def test_select_by_name_through_every_override_surface(
    kernel, fakes, clean_env, monkeypatch, name, rank_deterministic
):
    assert _select(kernel, name).name == name
    with kernel.kernel_override(FAMILY, MODE, name):
        assert _select(kernel, None).name == name
    monkeypatch.setenv(OVERRIDE_ENV, name)  # the environment-variable override
    assert _select(kernel, None).name == name
    assert kernel.sampling.topk_topp_renorm_rank_deterministic(name) is (
        rank_deterministic
    )


def test_ranked_default_prefers_the_performant_arm(kernel, fakes, clean_env):
    assert _select(kernel, None).name == FAKE_FUSED
    solution_only = kernel.sampling.select_topk_topp_renorm(
        probs_dtype=torch.float32, solution="fake", override=None
    )
    assert solution_only.name == FAKE_FUSED


def test_unknown_override_name_raises(kernel, fakes, clean_env):
    with pytest.raises(kernel.NoKernelFoundError):
        _select(kernel, "no_such_topk_topp_renorm")


def test_cross_operator_override_is_refused_on_every_surface(
    kernel, fakes, other_operator_kernel, clean_env, monkeypatch
):
    other = other_operator_kernel
    signature = next(iter(kernel.format_signatures("probs", "dense", {torch.float32})))
    # select_kernel itself resolves the name without an operator check ...
    assert kernel.select_kernel(FAMILY, MODE, signature, override=other).name == other
    # ... so the operator's selection refuses it: explicit override= ...
    with pytest.raises(ValueError, match=f"not registered under {FAMILY}.{MODE}"):
        _select(kernel, other)
    # ... kernel_override() context, for the selection and the facade ...
    probs = torch.full((1, 4), 0.25)
    top_ks = torch.tensor([TOP_K_DISABLED], dtype=torch.int32)
    top_ps = torch.ones(1)
    with kernel.kernel_override(FAMILY, MODE, other):
        with pytest.raises(ValueError, match="not registered under"):
            _select(kernel, None)
        with pytest.raises(ValueError, match="not registered under"):
            kernel.sampling.topk_topp_renorm(
                probs, top_ks, top_ps, solution=None, override=None
            )
    # ... and the mirrored environment key.
    monkeypatch.setenv(OVERRIDE_ENV, other)
    with pytest.raises(ValueError, match="not registered under"):
        _select(kernel, None)
    monkeypatch.delenv(OVERRIDE_ENV)
    assert fakes == []  # nothing under the operator ran either
    assert _select(kernel, None).name == FAKE_FUSED  # the check is one-way


def test_rank_deterministic_requires_a_declared_trait(kernel, fakes):
    with pytest.raises(ValueError, match="not registered"):
        kernel.sampling.topk_topp_renorm_rank_deterministic("no_such_kernel")
    with pytest.raises(ValueError, match="rank_deterministic"):
        kernel.sampling.topk_topp_renorm_rank_deterministic(FAKE_UNTRAITED)


def test_facade_runs_the_selected_kernel_on_the_call_arguments(
    kernel, fakes, clean_env
):
    probs = torch.full((2, 4), 0.25)
    top_ks = torch.tensor([1, TOP_K_DISABLED], dtype=torch.int32)
    top_ps = torch.tensor([0.5, 1.0])
    out = kernel.sampling.topk_topp_renorm(
        probs, top_ks, top_ps, solution=None, override=FAKE_FLASHINFER
    )
    ((name, args),) = fakes
    assert name == FAKE_FLASHINFER
    assert args[0] is probs and args[1] is top_ks and args[2] is top_ps
    assert out is not probs and torch.equal(out, probs)


# --------------------------------------------------------------------------
# Deprecated TS_DISABLE_FUSED_TOPK_TOPP alias
# --------------------------------------------------------------------------


@pytest.mark.parametrize("legacy", [None, "0", "", "true"])
def test_alias_ignores_unset_and_non_one_values(kernel, alias_state, legacy):
    environ = {} if legacy is None else {LEGACY_ENV: legacy}
    result, warned = _resolve(kernel, environ)
    assert result is None and warned == []
    assert OVERRIDE_ENV not in environ


def test_alias_maps_to_the_flashinfer_name_and_warns_once(kernel, alias_state, caplog):
    environ = {LEGACY_ENV: "1"}
    with caplog.at_level(logging.WARNING, logger=kernel.sampling.__name__):
        result, warned = _resolve(kernel, environ)
    assert result == FLASHINFER
    assert environ[OVERRIDE_ENV] == FLASHINFER  # mirrored for every selection
    assert len(warned) == 1 and "deprecated" in str(warned[0].message)
    assert f"{FAMILY}.{MODE}={FLASHINFER}" in str(warned[0].message)
    assert any(LEGACY_ENV in record.getMessage() for record in caplog.records)
    caplog.clear()
    again, warned_again = _resolve(kernel, environ)
    assert again == FLASHINFER and warned_again == [] and caplog.records == []


def test_alias_refuses_to_outrank_a_different_explicit_override(kernel, alias_state):
    environ = {LEGACY_ENV: "1", OVERRIDE_ENV: FAKE_FUSED}
    with pytest.raises(ValueError, match=OVERRIDE_ENV):
        kernel.sampling.resolve_topk_topp_renorm_override(environ)
    assert environ[OVERRIDE_ENV] == FAKE_FUSED


def test_alias_agrees_with_an_equal_explicit_override(kernel, alias_state):
    environ = {LEGACY_ENV: "1", OVERRIDE_ENV: FLASHINFER}
    result, warned = _resolve(kernel, environ)
    assert result == FLASHINFER and len(warned) == 1
    assert environ[OVERRIDE_ENV] == FLASHINFER


# --------------------------------------------------------------------------
# Pre-capture prepare hook
# --------------------------------------------------------------------------


def test_prepare_dispatches_only_to_the_kernel_that_needs_it(
    kernel, fakes, monkeypatch
):
    seen: list[object] = []
    monkeypatch.setitem(kernel.sampling._TOPK_TOPP_RENORM_PREPARE, FUSED, seen.append)
    kernel.sampling.prepare_topk_topp_renorm(FAKE_FUSED, "cpu")
    kernel.sampling.prepare_topk_topp_renorm(FAKE_FLASHINFER, "cpu")
    assert seen == []
    kernel.sampling.prepare_topk_topp_renorm(FUSED, "cpu")
    assert seen == [torch.device("cpu")]


def test_fused_prepare_is_a_no_op_off_cuda(kernel):
    # Idempotent no-op for a CPU device whether or not the wrapper imported.
    kernel.sampling.prepare_topk_topp_renorm(FUSED, torch.device("cpu"))
    kernel.sampling.prepare_topk_topp_renorm(FUSED, torch.device("cpu"))


# --------------------------------------------------------------------------
# FlashInfer sampling backend entry points (from source)
# --------------------------------------------------------------------------


def _backend_init_from_source(namespace: dict):
    _load(
        namespace,
        "sampling/backends/flashinfer.py",
        {"_init_topk_topp_renorm"},
        "FlashInferSamplingBackend",
    )
    return namespace["FlashInferSamplingBackend"]()


def test_backend_init_binds_selection_rule_and_prepare(
    kernel, fakes, clean_env, caplog
):
    prepared: list[tuple[str, object]] = []
    namespace = dict(
        torch=torch,
        os=os,
        logger=logging.getLogger("test.topk_topp_renorm.backend"),
        resolve_topk_topp_renorm_override=(
            kernel.sampling.resolve_topk_topp_renorm_override
        ),
        select_topk_topp_renorm=kernel.sampling.select_topk_topp_renorm,
        topk_topp_renorm_rank_deterministic=(
            kernel.sampling.topk_topp_renorm_rank_deterministic
        ),
        prepare_topk_topp_renorm=lambda name, device: prepared.append((name, device)),
    )

    default = _backend_init_from_source(namespace)
    default._init_topk_topp_renorm(NS(device="cpu"))
    assert default._topk_topp_renorm.name == FAKE_FUSED
    assert default._topk_topp_rank_deterministic is True

    overridden = _backend_init_from_source(namespace)
    with caplog.at_level(logging.INFO, logger="test.topk_topp_renorm.backend"):
        with kernel.kernel_override(FAMILY, MODE, FAKE_FLASHINFER):
            overridden._init_topk_topp_renorm(NS(device="cpu"))
    assert overridden._topk_topp_renorm.name == FAKE_FLASHINFER
    assert overridden._topk_topp_rank_deterministic is False
    # Prepare is keyed on the selected name and receives the config device.
    assert prepared == [(FAKE_FUSED, "cpu"), (FAKE_FLASHINFER, "cpu")]
    # The per-rank kernel-selection log line names the kernel and the broadcast rule.
    assert any(
        f"{FAMILY}.{MODE} -> {FAKE_FLASHINFER}" in record.getMessage()
        and "rank_deterministic=False" in record.getMessage()
        for record in caplog.records
    )
    # The bound kernel is the registered callable.
    probs = torch.ones(1, 3) / 3
    overridden._topk_topp_renorm(probs, torch.ones(1, dtype=torch.int32), torch.ones(1))
    assert fakes[-1][0] == FAKE_FLASHINFER


def test_backend_init_refuses_a_cross_operator_override(
    kernel, fakes, other_operator_kernel, clean_env
):
    namespace = dict(
        torch=torch,
        os=os,
        logger=logging.getLogger("test.topk_topp_renorm.backend"),
        resolve_topk_topp_renorm_override=(
            kernel.sampling.resolve_topk_topp_renorm_override
        ),
        select_topk_topp_renorm=kernel.sampling.select_topk_topp_renorm,
        topk_topp_renorm_rank_deterministic=(
            kernel.sampling.topk_topp_renorm_rank_deterministic
        ),
        prepare_topk_topp_renorm=lambda name, device: None,
    )
    backend = _backend_init_from_source(namespace)
    with kernel.kernel_override(FAMILY, MODE, other_operator_kernel):
        with pytest.raises(ValueError, match="not registered under"):
            backend._init_topk_topp_renorm(NS(device="cpu"))


def test_backend_init_passes_the_deprecated_alias_as_override(
    kernel, alias_state, monkeypatch
):
    monkeypatch.setenv(LEGACY_ENV, "1")
    seen: dict = {}

    def select(*, probs_dtype, solution, override):
        seen.update(probs_dtype=probs_dtype, solution=solution, override=override)
        return NS(name=FLASHINFER)

    namespace = dict(
        torch=torch,
        os=os,
        logger=logging.getLogger("test.topk_topp_renorm.backend"),
        resolve_topk_topp_renorm_override=(
            kernel.sampling.resolve_topk_topp_renorm_override
        ),
        select_topk_topp_renorm=select,
        topk_topp_renorm_rank_deterministic=lambda name: False,
        prepare_topk_topp_renorm=lambda name, device: None,
    )
    backend = _backend_init_from_source(namespace)
    with pytest.warns(DeprecationWarning):
        backend._init_topk_topp_renorm(NS(device="cpu"))
    assert seen == dict(probs_dtype=torch.float32, solution=None, override=FLASHINFER)
    assert os.environ[OVERRIDE_ENV] == FLASHINFER  # mirrored on this rank
    assert backend._topk_topp_renorm.name == FLASHINFER
    assert backend._topk_topp_rank_deterministic is False


@pytest.mark.parametrize(
    "rank_deterministic, pdl, expect_broadcast",
    [
        (True, False, False),  # fused arm without PDL: bit-identical ranks
        (True, True, True),  # PDL keeps ranks aligned through rank 0
        (False, False, True),  # FlashInfer arm: always broadcast
        (False, True, True),
    ],
)
def test_verify_broadcast_rule_follows_the_bound_spec_and_pdl(
    rank_deterministic, pdl, expect_broadcast
):
    events: list[str] = []
    renorm_calls: list[tuple] = []

    def renorm(probs, top_ks, top_ps):
        renorm_calls.append((probs, top_ks, top_ps))
        return probs

    def scalars(indices, **kwargs):
        return (
            torch.ones(1),
            torch.ones(1, dtype=torch.int32),
            torch.ones(1),
            None,
            torch.zeros(1, dtype=torch.int64),
            None,
        )

    def chain(**kwargs):
        predicts = kwargs["predicts"]
        predicts.copy_(kwargs["target_probs"].argmax(-1).flatten().to(torch.int32))
        kwargs["accept_token_num"].zero_()
        kwargs["accept_index"].copy_(
            torch.arange(predicts.numel()).view_as(kwargs["accept_index"])
        )

    namespace = dict(
        torch=torch,
        nvtx_range=lambda *a, **k: lambda fn: fn,
        SPECULATIVE_ACCEPT_THRESHOLD_SINGLE=1.0,
        SPECULATIVE_ACCEPT_THRESHOLD_ACC=1.0,
        pdl_enabled=lambda: pdl,
        gather_and_expand_scalars=scalars,
        softmax=lambda logits, **kw: logits.softmax(-1),
        chain_speculative_sampling_target_only=chain,
        gather_token_logprobs=lambda logits, tokens: None,
    )
    _load(
        namespace,
        "sampling/backends/flashinfer.py",
        {"verify"},
        "FlashInferSamplingBackend",
    )
    backend = namespace["FlashInferSamplingBackend"]()
    backend.config = NS(enable_output_logprobs=False)
    backend.broadcast_verify_outputs = lambda: events.append("broadcast")
    backend._topk_topp_renorm = renorm
    backend._topk_topp_rank_deterministic = rank_deterministic
    backend._predict_buf = torch.empty(1, dtype=torch.int32)
    backend._accept_index_buf = torch.empty(1, dtype=torch.int32)
    backend._accept_length_buf = torch.empty(1, dtype=torch.int32)
    backend._coins_buf = torch.full((1, 1), 0.5)
    backend._final_coins_buf = torch.full((1,), 0.5)
    for attr in ("_temperature_pool", "_top_k_pool", "_top_p_pool"):
        setattr(backend, attr, torch.ones(1))

    logits = torch.tensor([[0.0, 3.0, 1.0]])
    output = NS(
        next_token_logits=logits, next_token_logprobs=None, logits_layout_plan=None
    )
    info = NS(
        vocab_mask=None,
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        batch_row_offset=0,
    )
    predicted, lengths = backend.verify(
        output, info, torch.zeros((1, 1), dtype=torch.int32)
    )
    assert predicted.tolist() == [1] and lengths.tolist() == [1]
    assert events == (["broadcast"] if expect_broadcast else [])
    ((probs, top_ks, top_ps),) = renorm_calls
    assert probs.shape == (1, 3) and top_ks.dtype == torch.int32
    assert top_ps.shape == (1,)


def test_backend_source_dispatches_only_through_the_bound_operator():
    """No module-level availability flag or direct arm import remains."""
    source = BACKEND_SOURCE.read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {
        "fused_topk_topp_available",
        "fused_topk_topp_renorm",
        "fused_topk_topp_prepare",
        "top_k_renorm_prob",
        "top_p_renorm_prob",
    }.isdisjoint(imported)
    assert {
        "select_topk_topp_renorm",
        "resolve_topk_topp_renorm_override",
        "topk_topp_renorm_rank_deterministic",
        "prepare_topk_topp_renorm",
    } <= imported
    assert "_FUSED_TOPK_TOPP_AVAILABLE" not in source
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FlashInferSamplingBackend"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    called = [
        n.func.attr
        for n in ast.walk(init)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert called[-1] == "_init_topk_topp_renorm"


# --------------------------------------------------------------------------
# GPU: the two real arms (not run on CPU)
# --------------------------------------------------------------------------


@requires_flashinfer_kernels
def test_gpu_both_arms_select_by_name_and_agree_numerically(kernel, clean_env):
    sampling = kernel.sampling
    for name in (FUSED, FLASHINFER):
        assert _select(kernel, name).name == name
    assert _select(kernel, None).name == FUSED  # ranked default
    assert sampling.topk_topp_renorm_rank_deterministic(FUSED) is True
    assert sampling.topk_topp_renorm_rank_deterministic(FLASHINFER) is False

    torch.manual_seed(0)
    bs, vocab = 4, 8192
    probs = torch.softmax(
        torch.randn(bs, vocab, device="cuda", dtype=torch.float32) * 3.0, dim=-1
    )
    top_ks = torch.tensor([1, 8, 50, TOP_K_DISABLED], dtype=torch.int32, device="cuda")
    top_ps = torch.tensor([1.0, 0.9, 0.5, 0.8], dtype=torch.float32, device="cuda")
    sampling.prepare_topk_topp_renorm(FUSED, probs.device)
    fused = sampling.topk_topp_renorm(
        probs, top_ks, top_ps, solution=None, override=FUSED
    )
    reference = sampling.topk_topp_renorm(
        probs, top_ks, top_ps, solution=None, override=FLASHINFER
    )
    # Same tolerance policy as tokenspeed-kernel/test/ops/test_sampling.py::
    # test_fused_topk_topp_matches_pipeline: one boundary position may differ
    # at the top-p cutoff, values within fp32 accumulation-order noise.
    assert ((fused > 0) != (reference > 0)).sum(dim=-1).max().item() <= 1
    torch.testing.assert_close(fused, reference, atol=1e-5, rtol=1e-4)
    ones = torch.ones(bs, device="cuda")
    torch.testing.assert_close(fused.sum(dim=-1), ones, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(reference.sum(dim=-1), ones, atol=1e-5, rtol=1e-5)


@requires_flashinfer_kernels
def test_gpu_backend_binds_override_prepares_side_stream_and_verifies(
    kernel, clean_env
):
    from tokenspeed_kernel.thirdparty.cuda import fused_topk_topp as native

    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    vocab, pool, bs = 256, 4, 2
    config = SamplingBackendConfig(
        max_bs=bs,
        max_draft_tokens_per_req=1,
        max_req_pool_size=pool,
        vocab_size=vocab,
        device="cuda",
        enable_output_logprobs=False,
        enable_nan_detection=False,
    )
    default = FlashInferSamplingBackend(config)
    assert default._topk_topp_renorm.name == FUSED
    assert default._topk_topp_rank_deterministic is True
    # Prepared at init under the indexed key the kernel looks up (probs.device).
    assert torch.device("cuda", torch.cuda.current_device()) in native._side_streams

    with kernel.kernel_override(FAMILY, MODE, FLASHINFER):
        reference = FlashInferSamplingBackend(config)
    assert reference._topk_topp_renorm.name == FLASHINFER
    assert reference._topk_topp_rank_deterministic is False

    for backend in (default, reference):
        sp = [
            SamplingParams(temperature=0.8, top_k=50, top_p=0.9, seed=7 + i)
            for i in range(bs)
        ]
        for p in sp:
            p.verify(vocab)
        backend.prepare_step(
            request_ids=[f"r{i}" for i in range(bs)],
            request_pool_indices=list(range(bs)),
            sampling_params_list=sp,
            num_tokens_per_req=1,
        )
        info = SamplingBatchInfo(
            req_pool_indices=torch.arange(bs, device="cuda"),
            valid_cache_lengths=torch.zeros(pool + 1, dtype=torch.int32, device="cuda"),
            device="cuda",
        )
        logits = torch.randn(bs, vocab, device="cuda")
        candidates = torch.randint(0, vocab, (bs, 1), device="cuda", dtype=torch.int32)
        predict, accept_length = backend.verify(
            LogitsProcessorOutput(next_token_logits=logits), info, candidates
        )
        assert predict.shape == (bs,) and accept_length.tolist() == [1] * bs
        assert bool(((predict >= 0) & (predict < vocab)).all())
