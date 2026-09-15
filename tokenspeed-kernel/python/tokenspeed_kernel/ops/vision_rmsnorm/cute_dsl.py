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

"""Original one-launch vision RMSNorm; pinned ATen mean order, no FMA or FTZ.

ATen provenance is bound by the disconnected probe packet. No third-party
implementation is copied here. Optional CUTLASS is confined to this backend.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_compile_cache: dict[tuple, object] = {}


@dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b)],
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(value)


@dsl_user_op
def _mul(a, b, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b)],
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(value)


@dsl_user_op
def _rsqrt(a, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
        "rsqrt.approx.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(value)


@cute.jit
def _row_mean(x: cute.Tensor, row: cutlass.Int32, lane: cutlass.Int32):
    s0 = cutlass.Float32(0.0)
    s1 = cutlass.Float32(0.0)
    s2 = cutlass.Float32(0.0)
    s3 = cutlass.Float32(0.0)
    for chunk in cutlass.range_constexpr(8):
        col = 4 * lane + 128 * chunk
        x0 = cutlass.Float32(x[row, col])
        x1 = cutlass.Float32(x[row, col + 1])
        x2 = cutlass.Float32(x[row, col + 2])
        x3 = cutlass.Float32(x[row, col + 3])
        s0 = _add(s0, _mul(x0, x0))
        s1 = _add(s1, _mul(x1, x1))
        s2 = _add(s2, _mul(x2, x2))
        s3 = _add(s3, _mul(x3, x3))
    total = _add(_add(_add(s0, s1), s2), s3)
    for level in cutlass.range_constexpr(5):
        total = _add(total, cute.arch.shuffle_sync_down(total, 16 >> level))
    # Each active warp owns one full row; the tail predicate is warp-uniform.
    total = cute.arch.shuffle_sync(total, 0)
    return _mul(total, cutlass.Float32(1.0 / 1024.0))


@cute.kernel
def _norm_kernel(x: cute.Tensor, weight: cute.Tensor, out: cute.Tensor):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    lane = thread % 32
    row = block * 8 + thread // 32
    if row < x.shape[0]:
        scale = _rsqrt(_add(_row_mean(x, row, lane), cutlass.Float32(1e-6)))
        for chunk in cutlass.range_constexpr(8):
            for component in cutlass.range_constexpr(4):
                col = 4 * lane + 128 * chunk + component
                normalized = _mul(cutlass.Float32(x[row, col]), scale)
                out[row, col] = cutlass.BFloat16(
                    _mul(cutlass.Float32(weight[col]), normalized)
                )


@cute.kernel
def _mean_kernel(x: cute.Tensor, out: cute.Tensor):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    lane = thread % 32
    row = block * 8 + thread // 32
    if row < x.shape[0]:
        mean = _row_mean(x, row, lane)
        if lane == 0:
            out[row, 0] = mean


@cute.jit
def _launch(
    x: cute.Tensor, weight: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream
):
    _norm_kernel(x, weight, out).launch(
        grid=(cute.ceil_div(x.shape[0], 8), 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_mean(x: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
    _mean_kernel(x, out).launch(
        grid=(cute.ceil_div(x.shape[0], 8), 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


def _compiled(kind, tensors):
    key = (kind, tensors[0].device.index)
    if key not in _compile_cache:
        views = [
            from_dlpack(
                t.detach(), assumed_align=t.element_size(), enable_tvm_ffi=True
            ).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in tensors
        ]
        _compile_cache[key] = cute.compile(
            _launch if kind == "norm" else _launch_mean,
            *views,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _compile_cache[key]


@register_kernel(
    "vision_rmsnorm",
    "apply_vision_rmsnorm",
    name="vision_aten_order_cute_dsl",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    priority=Priority.SPECIALIZED,
    tags={"vision", "rounding-preserving-prototype"},
)
def apply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run validated CUDA BF16 [N,1024] activations; return a fresh BF16 tensor."""
    from tokenspeed_kernel.ops.vision_rmsnorm import TORCH_VERSION, supports_layout

    if (
        not x.is_cuda
        or torch.__version__ != TORCH_VERSION
        or torch.cuda.get_device_capability(x.device) != (10, 0)
        or not supports_layout(x, weight, 1e-6)
    ):
        raise ValueError("vision norm requires pinned torch / sm_100 / BF16 [N,1024]")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _compiled("norm", (x, weight, out))(x, weight, out)
    return out


def diagnostic_mean(x: torch.Tensor) -> torch.Tensor:
    """Return FP32 [N,1] means using the identical helper; never used for timing."""
    out = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    _compiled("mean", (x, out))(x, out)
    return out
