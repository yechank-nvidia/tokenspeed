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

"""SM100 exactness and per-call ownership for ordered selected logprobs."""

import pytest
import torch
from tokenspeed_kernel.ops.sampling import try_gather_token_logprobs
from tokenspeed_kernel.platform import current_platform, pdl_enabled

_platform = current_platform()
pytestmark = pytest.mark.skipif(
    not (
        _platform.is_nvidia
        and (_platform.arch_version.major, _platform.arch_version.minor) == (10, 0)
    ),
    reason="The ordered selected-token implementation is admitted only on SM100.",
)
VOCAB = 151936


def _reference(logits, tokens):
    # Same public mathematical operation; no runtime dependency in kernel tests.
    raw_logprobs = torch.log_softmax(logits.float(), dim=-1)
    return raw_logprobs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


@pytest.fixture(scope="module")
def oracle():
    return torch.compile(_reference, dynamic=True, backend="inductor")


@pytest.fixture(autouse=True)
def enabled_pdl():
    previous = pdl_enabled()
    pdl_enabled(True)
    try:
        yield
    finally:
        pdl_enabled(previous)


def _fixture(recipe, token):
    row = torch.full((1, VOCAB), -4.0, dtype=torch.float32)
    if recipe == "finite":
        row.copy_(torch.linspace(-11, 9, VOCAB, dtype=torch.float32).reshape(1, -1))
    elif recipe == "constant":
        row.fill_(2)
    elif recipe == "cross_lane_ties":
        row[0, [0, 383, 384, 1023, 1024, VOCAB - 1]] = 4
    elif recipe == "signed_zero":
        row.zero_()
        row.view(torch.int32)[0, 1::2] = -2147483648
    elif recipe == "tail":
        row.fill_(-1000)
        row[0, -1] = 1
    elif recipe == "underflow":
        row.fill_(-104)
        row[0, 0] = 0
    elif recipe == "positive_inf":
        row[0, 0] = float("inf")
    elif recipe == "all_negative_inf":
        row.fill_(float("-inf"))
    elif recipe == "distinct_nan":
        row.view(torch.int32)[0, [0, 64, 384, 1024]] = torch.tensor(
            [0x7FC00001, 0x7FC00002, -0x3FFFFD, 0x7FC00400], dtype=torch.int32
        )
    else:
        raise AssertionError(recipe)
    return row.cuda(), torch.tensor([token], dtype=torch.int32, device="cuda")


def _equal(actual, expected):
    assert actual is not None
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape == (1,)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize(
    "recipe,token",
    [
        ("finite", 0),
        ("finite", VOCAB - 1),
        ("constant", 1024),
        ("cross_lane_ties", 1024),
        ("signed_zero", VOCAB - 1),
        ("tail", VOCAB - 1),
        ("underflow", 1024),
        ("positive_inf", 1024),
        ("all_negative_inf", VOCAB - 1),
        ("distinct_nan", 1024),
    ],
)
def test_exact_eager_and_fresh_retained_outputs(oracle, recipe, token):
    logits, tokens = _fixture(recipe, token)
    logits_before, tokens_before = logits.clone(), tokens.clone()
    expected = oracle(logits, tokens)
    outputs = [try_gather_token_logprobs(logits, tokens) for _ in range(4)]
    for output in outputs:
        _equal(output, expected)
    assert len({output.data_ptr() for output in outputs}) == 4
    assert not {output.data_ptr() for output in outputs} & {
        logits.data_ptr(),
        tokens.data_ptr(),
    }
    tokens.fill_((token + 1) % VOCAB)
    _equal(try_gather_token_logprobs(logits, tokens), oracle(logits, tokens))
    for output in outputs:
        _equal(output, expected)
    assert torch.equal(logits.view(torch.int32), logits_before.view(torch.int32))
    tokens.copy_(tokens_before)


def _capture(logits, tokens, count):
    try_gather_token_logprobs(logits, tokens)
    torch.cuda.synchronize()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    outputs = []
    try:
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(count):
                outputs.append(try_gather_token_logprobs(logits, tokens))
        torch.cuda.current_stream().wait_stream(stream)
        return graph, outputs
    except BaseException:
        graph.reset()
        raise


def test_graph_reads_live_logits_tokens_and_keeps_eager_outputs(oracle):
    logits, tokens = _fixture("finite", 1024)
    original, original_tokens = logits.clone(), tokens.clone()
    expected = oracle(logits, tokens)
    eager = try_gather_token_logprobs(logits, tokens)
    graph, outputs = _capture(logits, tokens, 8)
    try:
        assert all(output is not None for output in outputs)
        assert len({eager.data_ptr(), *(output.data_ptr() for output in outputs)}) == 9
        for mutation in ("unchanged", "logits", "tokens", "restored"):
            logits.copy_(original)
            tokens.copy_(original_tokens)
            if mutation == "logits":
                logits[0, 1024] = 0.75
            if mutation == "tokens":
                tokens.fill_(VOCAB - 1)
            wanted = oracle(logits, tokens)
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for output in outputs:
                _equal(output, wanted)
            _equal(eager, expected)
    finally:
        torch.cuda.synchronize()
        graph.reset()
        outputs.clear()


def test_two_active_graphs_have_independent_output_ownership(oracle):
    first = _fixture("finite", 1024)
    second = _fixture("tail", VOCAB - 1)
    expected_first, expected_second = oracle(*first), oracle(*second)
    graph1, outputs1 = _capture(*first, 2)
    graph2, outputs2 = _capture(*second, 2)
    try:
        assert len({output.data_ptr() for output in outputs1 + outputs2}) == 4
        for _ in range(3):
            graph1.replay()
            graph2.replay()
            torch.cuda.synchronize()
            for output in outputs1:
                _equal(output, expected_first)
            for output in outputs2:
                _equal(output, expected_second)
    finally:
        torch.cuda.synchronize()
        graph1.reset()
        graph2.reset()
        outputs1.clear()
        outputs2.clear()
    fresh = try_gather_token_logprobs(*first)
    _equal(fresh, expected_first)


def test_lazy_negative_view_and_offset_inputs_fall_back():
    logits, tokens = _fixture("finite", 0)
    negative = torch._neg_view(logits)
    assert negative.is_neg() and negative.data_ptr() == logits.data_ptr()
    assert try_gather_token_logprobs(negative, tokens) is None
    offset_tokens = torch.zeros((5,), dtype=torch.int32, device="cuda")[4:]
    assert offset_tokens.data_ptr() % 16 == 0 and offset_tokens.storage_offset() == 4
    assert try_gather_token_logprobs(logits, offset_tokens) is None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert try_gather_token_logprobs(logits, tokens) is None
