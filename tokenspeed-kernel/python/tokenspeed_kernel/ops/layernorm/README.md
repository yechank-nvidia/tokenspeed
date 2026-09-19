# Layer normalization operations

## Reused Q/K RMSNorm and position-indexed scaling

Import `staged_qk_rmsnorm_ssmax` from `tokenspeed_kernel.ops.layernorm`:

```python
q_out, k_out = staged_qk_rmsnorm_ssmax(
    q, k, q_weight, k_weight, positions, scale_table, eps, out
)
```

This wrapper now reuses the same internal fused Q/K RMSNorm kernel as ordinary
`qk_rmsnorm`, instead of maintaining a second normalization implementation.
Ordinary calls explicitly disable staged rounding and position scaling.

Staged mode performs FP32 mean-square reduction, BF16 normalized-value and
affine-result rounding, then FP32 position-indexed query scaling followed by
BF16 output rounding. Weights are rounded to BF16. K is not scaled.
The caller supplies the FP32 table; no scaling formula is imposed.

Q/K are BF16 rank-two tensors with contiguous last axes and head dimension
1–1024. Weights are contiguous BF16/FP32 vectors. Positions are contiguous
INT32/INT64; the scale table is nonempty contiguous FP32. Invalid positions
produce NaNs only in Q without an out-of-bounds table read.
All inputs share one GPU. Epsilon is finite and positive.
`out=None` allocates outputs; explicit outputs must be contiguous matching
tensors without storage aliasing. Empty batches launch nothing.

The shared kernel handles partial dimension/head tiles and groups four heads
per program in staged mode. Tests cover striding, preallocated outputs, invalid
positions, controlled rounding, and graph replay with changing inputs.

```bash
python -m pytest tokenspeed-kernel/test/ops/test_layernorm.py -k 'qk or ssmax' -q
```

This branch contains a source refactor, not a new measured speedup. GPU
correctness, graph replay and performance must be checked on the built candidate.
