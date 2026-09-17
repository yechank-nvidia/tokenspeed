# MHA decode execution metadata

`mha_decode_with_kvcache` requires the keyword-only argument
`decode_workspace`. Pass `None` to keep the existing per-call preparation
path, or a caller-owned view returned by
`prepare_mha_decode_workspace(max_batch_size, device)` to reuse metadata.
Kernel selection, split policy, attention math and output semantics do not
change. Backends that do not need this workspace consume and ignore it;
the keyword is never passed into an external vendor implementation.

## Ownership and shape

Prepare the workspace once before graph capture. Its capacity is the number
of decode metadata entries, not the number of query tokens: compact q4 decode
still has one entry per request. Supply a contiguous `workspace[:batch]` view
whose length equals `cache_seqlens.shape[0]`, including padded entries.
The preparation helper returns contiguous int32 storage on the requested
device. The portable implementation checks dtype, device, one-dimensional
layout, contiguity and exact batch length without reading device values or
synchronizing during forward.

The contents are implementation-owned and must be treated as immutable.
Use the preparation helper rather than constructing or changing values to
select a split policy. Reusing this workspace does not reuse attention
outputs or Q/K/V scratch, and it does not carry request or prefix state.

Initialize on the execution stream or establish a dependency before any
other stream reads it. Keep a strong reference to the storage and retain
pointer-stable per-batch views through all in-flight readers and captured
graphs. Release graphs and readers before dropping or replacing the
workspace; then prepare new storage before recapture. Do not initialize it
inside capture or place it in scratch storage another operation can overwrite.

The same prepared view is valid for eager calls and graph replay. There is
no per-forward fill, value check, graph-only workspace branch or new global
cache. Passing `None` remains an explicit choice when callers cannot provide
this lifetime guarantee.
