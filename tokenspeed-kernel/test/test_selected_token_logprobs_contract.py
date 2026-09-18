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

"""CPU-only source contracts without Torch, Triton or package imports."""

import ast
import builtins
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "python/tokenspeed_kernel"
FACADE = PACKAGE / "ops/sampling/__init__.py"
ADAPTER = PACKAGE / "ops/sampling/triton/ordered_logprobs.py"
LOW_LEVEL = PACKAGE / "thirdparty/triton/ordered_logprobs.py"


@dataclass(frozen=True)
class Device:
    type: str
    index: int


class Tensor:
    def __init__(self, shape, dtype, pointer):
        self.shape, self.dtype, self.pointer = shape, dtype, pointer
        self.device, self.layout = Device("cuda", 0), "strided"
        self.is_cuda, self.requires_grad = True, False
        self.offset = 0
        self.negative, self.conjugated = False, False
        self.strides = (151936, 1) if shape == (1, 151936) else (1,)

    def stride(self):
        return self.strides

    def storage_offset(self):
        return self.offset

    def data_ptr(self):
        return self.pointer

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugated


class NoKernelFoundError(RuntimeError):
    pass


def extract(path, names, namespace):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )


@pytest.fixture
def api():
    state = NS(
        allocations=[],
        imports=[],
        launches=[],
        selections=[],
        pdl=True,
        autocast=False,
        current_device=0,
        missing=False,
        import_missing=False,
        launch_error=False,
        platform=NS(is_nvidia=True, arch_version=NS(major=10, minor=0)),
    )

    def empty(shape, *, dtype, device):
        value = Tensor(shape, dtype, 0x1000000 + 0x100000 * len(state.allocations))
        value.device = device
        state.allocations.append(value)
        return value

    def launch(*args, **kwargs):
        state.launches.append((args, kwargs))
        if state.launch_error:
            raise RuntimeError("device launch failure")

    def importer(name, globals, locals, fromlist, level):
        if name == "__future__":
            return builtins.__import__(name, globals, locals, fromlist, level)
        state.imports.append(name)
        if state.import_missing:
            raise ImportError("optional implementation unavailable")
        if name == "tokenspeed_kernel.ops.sampling.triton.ordered_logprobs":
            return NS()
        if name == "tokenspeed_kernel.thirdparty.triton.ordered_logprobs":
            return NS(launch_ordered_logprobs=launch)
        raise AssertionError("Unexpected import: " + name)

    namespace = dict(
        torch=NS(
            Tensor=Tensor,
            strided="strided",
            float32="fp32",
            int32="i32",
            is_autocast_enabled=lambda device_type: state.autocast,
            cuda=NS(current_device=lambda: state.current_device),
            empty=empty,
        ),
        current_platform=lambda: state.platform,
        pdl_enabled=lambda: state.pdl,
        NoKernelFoundError=NoKernelFoundError,
        format_signature=lambda **kwargs: kwargs,
        dense_tensor_format=lambda dtype: dtype,
        __builtins__={**vars(builtins), "__import__": importer},
    )
    extract(
        FACADE,
        {"_supports_selected_token_logprobs", "try_gather_token_logprobs"},
        namespace,
    )
    extract(ADAPTER, {"triton_gather_token_logprobs"}, namespace)

    def select(*args, **kwargs):
        state.selections.append((args, kwargs))
        if state.missing:
            raise NoKernelFoundError("no matching implementation")
        return namespace["triton_gather_token_logprobs"]

    namespace["select_kernel"] = select
    return NS(
        state=state,
        public=namespace["try_gather_token_logprobs"],
        adapter=namespace["triton_gather_token_logprobs"],
        eligible=namespace["_supports_selected_token_logprobs"],
    )


def inputs():
    return Tensor((1, 151936), "fp32", 0x100000), Tensor((1,), "i32", 0x200000)


@pytest.mark.parametrize(
    "target, attribute, value",
    [
        ("logits", "shape", (0, 151936)),
        ("logits", "shape", (2, 151936)),
        ("logits", "shape", (1, 151935)),
        ("logits", "shape", (151936,)),
        ("tokens", "shape", (1, 1)),
        ("logits", "dtype", "bf16"),
        ("tokens", "dtype", "i64"),
        ("logits", "strides", (151940, 1)),
        ("logits", "strides", (303872, 2)),
        ("tokens", "strides", (2,)),
        ("logits", "offset", 4),
        ("tokens", "offset", 4),
        ("logits", "pointer", 0x100004),
        ("tokens", "pointer", 0x200004),
        ("logits", "pointer", 0),
        ("tokens", "pointer", 0),
        ("tokens", "pointer", 0x100010),
        ("logits", "requires_grad", True),
        ("tokens", "requires_grad", True),
        ("logits", "negative", True),
        ("logits", "conjugated", True),
        ("tokens", "negative", True),
        ("tokens", "conjugated", True),
        ("logits", "is_cuda", False),
        ("tokens", "is_cuda", False),
        ("tokens", "device", Device("cuda", 1)),
        ("logits", "layout", "sparse"),
        ("tokens", "layout", "sparse"),
    ],
)
def test_metadata_rejection_is_side_effect_free(api, target, attribute, value):
    logits, tokens = inputs()
    setattr({"logits": logits, "tokens": tokens}[target], attribute, value)
    assert api.public(logits, tokens) is None
    assert not api.state.allocations
    assert not api.state.imports
    assert not api.state.launches
    assert not api.state.selections


@pytest.mark.parametrize(
    "field,value",
    [
        ("pdl", False),
        ("autocast", True),
        ("current_device", 1),
        ("platform", NS(is_nvidia=False, arch_version=NS(major=10, minor=0))),
        ("platform", NS(is_nvidia=True, arch_version=NS(major=10, minor=3))),
        ("platform", NS(is_nvidia=True, arch_version=NS(major=9, minor=0))),
    ],
)
def test_environment_rejection_allocates_nothing(api, field, value):
    setattr(api.state, field, value)
    assert api.public(*inputs()) is None
    assert not api.state.allocations and not api.state.imports


def test_non_tensor_rejection(api):
    logits, tokens = inputs()
    assert api.public(None, tokens) is None
    assert api.public(logits, None) is None
    assert not api.state.allocations and not api.state.imports


def test_fresh_scratch_and_output_per_call(api):
    logits, tokens = inputs()
    first, second = api.public(logits, tokens), api.public(logits, tokens)
    assert first is not second
    assert first.shape == (1,) and first.dtype == "fp32"
    assert len(api.state.allocations) == 8
    assert len({value.data_ptr() for value in api.state.allocations}) == 8
    for index, (args, kwargs) in enumerate(api.state.launches):
        owned = api.state.allocations[index * 4 : index * 4 + 4]
        assert args == (logits, tokens, *owned)
        assert kwargs == {"enable_pdl": True}
        assert [value.shape for value in owned] == [(1024,), (1,), (1024,), (1,)]
        assert sum(value.shape[0] * 4 for value in owned[:3]) == 8196
    assert api.state.selections[0] == (
        ("sampling", "gather_token_logprobs", {"logits": "fp32", "tokens": "i32"}),
        {"traits": {"rows": 1, "vocab_size": 151936}},
    )


@pytest.mark.parametrize("field", ["missing", "import_missing"])
def test_missing_implementation_returns_none_before_allocating(api, field):
    setattr(api.state, field, True)
    assert api.public(*inputs()) is None
    assert not api.state.allocations and not api.state.launches


def test_direct_adapter_rejects_before_import_or_allocation(api):
    logits, tokens = inputs()
    logits.offset = 1
    with pytest.raises(ValueError, match="Unsupported"):
        api.adapter(logits, tokens)
    assert not api.state.allocations and not api.state.imports


def test_launch_errors_are_not_hidden_as_fallback(api):
    api.state.launch_error = True
    with pytest.raises(RuntimeError, match="device launch failure"):
        api.public(*inputs())
    assert len(api.state.launches) == 1


def test_new_arguments_are_explicit(api):
    for function in (api.public, api.adapter, api.eligible):
        assert all(
            parameter.default is inspect.Parameter.empty
            for parameter in inspect.signature(function).parameters.values()
        )


def test_no_runtime_dependency_or_global_tensor_workspace():
    for path in (ADAPTER, LOW_LEVEL):
        source = path.read_text()
        assert "tokenspeed.runtime" not in source
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.Assign, ast.AnnAssign)) for node in tree.body
        )
    tree = ast.parse(LOW_LEVEL.read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "_ordered_logprob_max_lanes",
        "_ordered_logprob_max_reduce",
        "_ordered_logprob_sum_lanes",
        "_ordered_logprob_sum_reduce",
        "launch_ordered_logprobs",
    }.issubset(functions)
    reducer = ast.unparse(functions["_ordered_logprob_sum_reduce"])
    assert "add.rn.f32 s01, $1, $2;" in reducer
    assert "add.rn.f32 s012, $3, s01;" in reducer
    assert "add.rn.f32 $0, $4, s012;" in reducer
    assert "add.rn.ftz" not in reducer
