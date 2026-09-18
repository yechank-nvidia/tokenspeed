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

"""Ordered FP32 selected-token logprobs with explicit caller-owned buffers.

Each partial lane visits columns lane + 1024*k in order. Tagged maximum
selection preserves original operand bits, and the final sum retains explicit
non-FTZ FP32 rounding before its eight-warp reduction.
"""

from tokenspeed_kernel._triton import libdevice, tl, triton


@triton.jit
def _maximum_left_nan(a, b):
    return tl.where((a > b) | (a != a), a, b)


@triton.jit
def _ordered_logprob_max_lanes(logits, lane_max, r0_numel, ENABLE_PDL: tl.constexpr):
    lane = tl.program_id(0) * 32 + tl.arange(0, 32)[None, :]
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    accumulator = tl.full([1, 32], float("-inf"), tl.float32)
    for offset in tl.range(0, r0_numel, 1024):
        index = offset + lane
        mask = index < r0_numel
        value = tl.load(logits + index, mask, eviction_policy="evict_last", other=0.0)
        value = tl.broadcast_to(value, [1, 32])
        updated = _maximum_left_nan(accumulator, value)
        accumulator = tl.where(mask, updated, accumulator)
    tl.store(lane_max + lane, accumulator)
    if ENABLE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _maximum_ranked(a, a_rank, b, b_rank):
    a_nan = a != a
    b_nan = b != b
    nan_choose_a = a_nan & ((b == b) | (a_rank < b_rank))
    numeric_choose_a = (a > b) | ((a == b) & (a_rank > b_rank))
    choose_a = tl.where(a_nan | b_nan, nan_choose_a, numeric_choose_a)
    return tl.where(choose_a, a, b), tl.where(choose_a, a_rank, b_rank)


@triton.jit
def _ordered_logprob_max_reduce(lane_max, row_max, ENABLE_PDL: tl.constexpr):
    lane = tl.arange(0, 1024)[None, :]
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    accumulator = tl.load(lane_max + lane)
    # Rank original leaves in the oracle's lane-0 butterfly visitation order.
    thread = lane >> 2
    warp = thread >> 5
    warp_lane = thread & 31
    reversed_warp = ((warp & 1) << 2) | (warp & 2) | ((warp >> 2) & 1)
    reversed_lane = (
        ((warp_lane & 1) << 4)
        | ((warp_lane & 2) << 2)
        | (warp_lane & 4)
        | ((warp_lane >> 2) & 2)
        | ((warp_lane >> 4) & 1)
    )
    rank = ((reversed_warp * 32 + reversed_lane) * 4) + (lane & 3)
    maximum, selected_rank = tl.reduce((accumulator, rank), 1, _maximum_ranked)
    maximum = maximum[:, None]
    tl.store(row_max + tl.full([1, 1], 0, tl.int32), maximum)
    if ENABLE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _ordered_logprob_sum_lanes(
    logits, row_max, lane_sum, r0_numel, ENABLE_PDL: tl.constexpr
):
    lane = tl.program_id(0) * 32 + tl.arange(0, 32)[None, :]
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    maximum = tl.load(row_max + tl.full([1, 1], 0, tl.int32))
    accumulator = tl.full([1, 32], 0, tl.float32)
    for offset in tl.range(0, r0_numel, 1024):
        index = offset + lane
        mask = index < r0_numel
        value = tl.load(logits + index, mask, eviction_policy="evict_last", other=0.0)
        shifted = value - maximum
        exponential = libdevice.exp(shifted)
        exponential = tl.broadcast_to(exponential, [1, 32])
        updated = accumulator + exponential
        accumulator = tl.where(mask, updated, accumulator)
    tl.store(lane_sum + lane, accumulator)
    if ENABLE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _ordered_logprob_sum_reduce(
    logits, tokens, row_max, lane_sum, output, r0_numel, ENABLE_PDL: tl.constexpr
):
    lane = tl.arange(0, 256)[None, :]
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    # Match the oracle's four-adjacent-lane serial FP32 tree before shuffles.
    a0 = tl.load(lane_sum + 4 * lane)
    a1 = tl.load(lane_sum + 4 * lane + 1)
    a2 = tl.load(lane_sum + 4 * lane + 2)
    a3 = tl.load(lane_sum + 4 * lane + 3)
    partial = tl.inline_asm_elementwise(
        """
        {
            .reg .f32 s01, s012;
            add.rn.f32 s01, $1, $2;
            add.rn.f32 s012, $3, s01;
            add.rn.f32 $0, $4, s012;
        }
        """,
        constraints="=f,f,f,f,f",
        args=[a0, a1, a2, a3],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    total = tl.sum(partial, 1)[:, None]
    token = tl.load(tokens + 0)
    token = tl.broadcast_to(token, [1, 1])
    tl.device_assert(
        (0 <= token) & (token < r0_numel), "selected token index out of bounds"
    )
    selected = tl.load(logits + token, None, eviction_policy="evict_last")
    maximum = tl.load(row_max + tl.full([1, 1], 0, tl.int32))
    shifted = selected - maximum
    logarithm = tl.math.log(total)
    result = shifted - logarithm
    tl.store(output + tl.full([1, 1], 0, tl.int32), result, None)
    if ENABLE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


def launch_ordered_logprobs(
    logits, tokens, lane_max, row_max, lane_sum, output, *, enable_pdl
):
    """Launch four stages into explicit buffers; return four compiled handles.

    FP32 logits[1,151936], INT32 tokens[1], FP32 lane_max[1024], row_max[1],
    lane_sum[1024], output[1] must all be CUDA/current-device, contiguous with
    their exact row-major strides, 16-byte aligned, and pairwise disjoint.
    tokens[0] must be in [0,151936); device assertion checks it without a host
    read. All float bit categories are admitted without normalization.
    Every scratch/output element is overwritten; no allocation, host sync,
    finite check, token read, or warming occurs here. Caller retains all owners
    and compiled handles through graph replay. Explicit enable_pdl controls
    wait-before-load and late-after-store completion in every stage.
    """
    import torch

    if type(enable_pdl) is not bool:
        raise ValueError("enable_pdl must be bool")
    specs = (
        (logits, (1, 151936), (151936, 1), torch.float32),
        (tokens, (1,), (1,), torch.int32),
        (lane_max, (1024,), (1,), torch.float32),
        (row_max, (1,), (1,), torch.float32),
        (lane_sum, (1024,), (1,), torch.float32),
        (output, (1,), (1,), torch.float32),
    )
    ranges = []
    for tensor, shape, stride, dtype in specs:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise ValueError("all arguments must be CUDA tensors")
        if (
            tuple(tensor.shape) != shape
            or tuple(tensor.stride()) != stride
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError("unsupported shape, stride or dtype")
        if tensor.device != logits.device or tensor.requires_grad:
            raise ValueError("one device and no autograd required")
        pointer = tensor.data_ptr()
        if pointer <= 0 or pointer % 16:
            raise ValueError("all tensor pointers must be nonzero and 16-byte aligned")
        ranges.append((pointer, pointer + tensor.numel() * tensor.element_size()))
    if logits.device.index != torch.cuda.current_device():
        raise ValueError("current device must match tensors")
    ranges.sort()
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("all argument byte ranges must be disjoint")
    options = {
        "num_stages": 1,
        "enable_fp_fusion": True,
        "enable_reflect_ftz": True,
        "launch_pdl": enable_pdl,
        "debug": True,
        "sanitize_overflow": False,
    }
    first = _ordered_logprob_max_lanes[(32,)](
        logits, lane_max, 151936, ENABLE_PDL=enable_pdl, num_warps=1, **options
    )
    second = _ordered_logprob_max_reduce[(1,)](
        lane_max, row_max, ENABLE_PDL=enable_pdl, num_warps=8, **options
    )
    third = _ordered_logprob_sum_lanes[(32,)](
        logits, row_max, lane_sum, 151936, ENABLE_PDL=enable_pdl, num_warps=1, **options
    )
    fourth = _ordered_logprob_sum_reduce[(1,)](
        logits,
        tokens,
        row_max,
        lane_sum,
        output,
        151936,
        ENABLE_PDL=enable_pdl,
        num_warps=8,
        **options
    )
    return first, second, third, fourth
