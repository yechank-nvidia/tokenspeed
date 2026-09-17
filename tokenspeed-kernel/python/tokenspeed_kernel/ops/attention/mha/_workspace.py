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

import torch


def prepare_mha_decode_workspace(
    max_batch_size: int, device: torch.device | str
) -> torch.Tensor:
    """Prepare caller-owned, read-only metadata for MHA decode.

    Args:
        max_batch_size: Maximum number of decode metadata entries, not query
            tokens. Pass a nonnegative integer.
        device: Device on which the decode queries execute.

    Returns:
        Contiguous int32 metadata with shape [max_batch_size]. Treat its
        contents as immutable and implementation-owned; pass a [:batch] view
        as decode_workspace. Prepare before capture, order initialization
        before use on other streams, and retain the storage until all captured
        graphs and in-flight readers are released. Backends that do not use
        this metadata may ignore it.
    """
    if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
        raise TypeError("max_batch_size must be an integer")
    if max_batch_size < 0:
        raise ValueError("max_batch_size must be nonnegative")
    return torch.ones((max_batch_size,), dtype=torch.int32, device=device)
