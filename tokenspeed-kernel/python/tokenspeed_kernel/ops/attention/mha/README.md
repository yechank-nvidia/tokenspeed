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

## Grouped decode KV tiles

The stage-1 KV tile of grouped (GQA/MQA/MLA) paged decode is an explicit
`block_n` parameter of the Triton host chain in `_triton/decode.py`
(`_decode_grouped_att_m_fwd`, `decode_attention_fwd_grouped`,
`decode_attention_fwd`, `_triton_mha_decode_with_kvcache_impl`). No function in
that chain has a default for it and the launcher derives nothing from the
platform: it validates that the tile is a power of two of at least 16 (the K
dimension of the stage-1 PV `tl.dot`) and launches `BLOCK_N=block_n`. The
registered kernel chooses the tile:

- `triton_mha_decode_with_kvcache` (solution `triton`, `Priority.PORTABLE`,
  the ranked default) passes `grouped_decode_block_n(q, k_cache, v_cache)`,
  the one derivation of the default tile: 128 on NVIDIA SM100 when q/k/v are
  BF16 with 128-wide key and value heads on 64-token pages, 16 on AMD for key
  heads of 576 or more (MI3xx shared memory), 32 otherwise.
- `triton_mha_decode_kv32` (solution `triton`, `Priority.REFERENCE`, NVIDIA
  only) is the same host with `block_n=32`. It is not a second implementation
  and is never auto-selected while the default is registered; it exists so the
  32-token tile is a registry entity that can be forced by name for an A/B
  against the SM100 128-tile default:
  `mha_decode_with_kvcache(..., override="triton_mha_decode_kv32")`,
  `kernel_override("attention", "mha_decode_with_kvcache", "triton_mha_decode_kv32")`,
  or `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_DECODE_WITH_KVCACHE=triton_mha_decode_kv32`.
  Both registrations share traits and signatures, so switching by name never
  changes the facade's trait filtering.

The tile is an execution-geometry choice in the same kernel, independent of
batch size, query width and compute window; eager execution and graph capture
use the same value, and a name override must be fixed before capture. Larger
tiles reduce online-softmax loop iterations. Split counts, scratch ownership,
causal/window masks and stage2 do not change. The different reduction grouping
can change output bits; BF16 results are tested against the existing SDPA
tolerance, not bitwise equality between tiles. No model-specific path or
persistent state is introduced. The single-KV-head launcher
(`kv_group_num == 1`) keeps its own fixed tile and does not consume `block_n`.
