# Layer normalization operations

## Staged Q/K RMSNorm and position-indexed scaling

Import `staged_qk_rmsnorm_ssmax` from `tokenspeed_kernel.ops.layernorm`:

```python
q_out, k_out = staged_qk_rmsnorm_ssmax(
    q, k, q_weight, k_weight, positions, scale_table, eps, out
)
```

All arguments are explicit. This operation combines per-head RMSNorm with
SSMax-style position-indexed query scaling. The caller supplies the FP32
scale table; the kernel does not choose a scale formula. It performs these
steps in one GPU launch:

1. Compute each head's mean square and inverse RMS in FP32.
2. Multiply by the inverse RMS and round the normalized values to BF16.
3. Convert weights to BF16, multiply, and round the affine result to BF16.
4. Multiply Q by `scale_table[positions[token]]` in FP32 and round to BF16.
   K keeps the BF16 affine result from step 3.

These intermediate roundings are part of the API contract. Existing
`qk_rmsnorm` multiplies by FP32 weights before its final rounding, so appending
a scale multiplication to that operation does not preserve this contract.
The new entry point is registered as `triton_staged_qk_rmsnorm_ssmax` in
`layernorm.staged_qk_rmsnorm_ssmax` with portable priority.

### Inputs and output storage

- Q and K are BF16 `[T, QH * D]` and `[T, KH * D]`, with positive head
  counts and `1 <= D <= 1024`. Non-power-of-two head dimensions are supported.
  Their last axis is contiguous; token strides, broadcast token rows and
  storage offsets are supported.
- Both weights are contiguous BF16 or FP32 vectors of length `D`.
- Positions are a contiguous INT32 or INT64 vector of length `T`.
- The scale table is a nonempty contiguous FP32 vector. Valid positions are
  in `[0, len(scale_table))`. Invalid positions produce NaNs in the affected
  Q row without an out-of-bounds read. K is independent of positions and
  scales. This check runs on the device, without a host synchronization.
- All tensors share one CUDA or ROCm GPU. Epsilon is finite and positive.
- `out=None` allocates contiguous BF16 outputs. A supplied pair must match
  input shapes, dtypes and device. Nonempty destinations cannot share storage
  with inputs or each other, including disjoint views of an allocation.
- Empty token batches return empty outputs without a kernel launch.

The Triton kernel processes four heads per program and masks partial head
and dimension tiles. Warm up the same shapes before CUDA Graph capture;
allocated and supplied outputs both support replay with updated inputs,
positions and table values. FP32 reduction order can affect final BF16
values, so tests compare against the staged Torch composition with tolerances
and use a controlled-input test to check the rounding stages exactly.

```bash
python -m pytest tokenspeed-kernel/test/ops/test_staged_qk_rmsnorm_ssmax.py -q
```
