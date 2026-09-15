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

"""Release graph-captured NCCL nodes before destroying their groups."""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-2gpu")

from tokenspeed.runtime.execution import (  # noqa: E402
    forward_step as forward_step_module,
)
from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    BreakableCapture,
)
from tokenspeed.runtime.execution.device import DeviceHandle  # noqa: E402
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner  # noqa: E402
from tokenspeed.runtime.execution.forward_thread import ForwardThread  # noqa: E402
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph  # noqa: E402


def _collective_graph_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    host_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=20))
    x = torch.full((16,), float(rank + 1), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            warmup = x * 1
            dist.all_reduce(warmup)
            warmup.add_(1)
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    decode_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(decode_graph, pool=pool, stream=stream):
        decode_out = x * 1
        dist.all_reduce(decode_out)

    capture = BreakableCapture(pool=pool, stream=stream)
    with capture:
        y = x * 1
        dist.all_reduce(y)
        capture.add_eager(lambda: y.add_(1))
        prefill_out = y * 2

    decode = ForwardStepRunner.__new__(ForwardStepRunner)
    decode.device_module = torch.cuda
    decode.graphs = {("default", 1): decode_graph}
    decode.output_buffers = {("default", 1): (decode_out,)}
    decode._metadata_snapshots = {}
    forward_step_module.global_graph_memory_pool = pool
    prefill = PrefillGraph.__new__(PrefillGraph)
    prefill._captures = {16: capture}
    prefill._outputs = {16: prefill_out}
    prefill._ctx = None
    prefill._input_embeds_buf = x
    prefill._pool = pool
    thread = ForwardThread(torch.device("cuda", rank))
    executor = SimpleNamespace(
        forward_thread=thread,
        device_module=torch.cuda,
        prefill_graph=prefill,
        forward_step=decode,
    )
    handle = DeviceHandle(executor, l2_cache_executor=None, kv_transfer=None)

    def replay():
        decode_graph.replay()
        capture.replay(valid_rows=None)
        torch.cuda.synchronize()

    try:
        thread.run(replay)
        torch.testing.assert_close(decode_out, torch.full_like(decode_out, 3))
        torch.testing.assert_close(prefill_out, torch.full_like(prefill_out, 8))
        handle.close()
        handle.close()
        assert not thread._thread.is_alive()
        assert not decode.graphs and not decode.output_buffers
        assert not prefill._captures and not capture._graphs
        assert forward_step_module.global_graph_memory_pool is None
        # An early rank must keep shared host state live until its peer has
        # also released graph executables and joined its device thread.
        dist.monitored_barrier(
            group=host_group, timeout=timedelta(seconds=20), wait_all_ranks=True
        )
        dist.destroy_process_group()
    finally:
        thread.shutdown()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or torch.version.hip is not None,
    reason="two NVIDIA GPUs required",
)
def test_collective_graph_teardown_completes_before_process_group_destroy(tmp_path):
    context = multiprocessing.get_context("spawn")
    rendezvous = (tmp_path / "collective-rendezvous").as_uri()
    processes = [
        context.Process(target=_collective_graph_worker, args=(rank, rendezvous))
        for rank in range(2)
    ]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        deadline = time.monotonic() + 75
        for process in started:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        assert [process.exitcode for process in started] == [0, 0]
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
