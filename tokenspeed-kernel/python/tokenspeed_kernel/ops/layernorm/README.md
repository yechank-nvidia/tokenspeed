# Layer normalization operations

## Staged Q/K RMSNorm and RoPE

Import `staged_qk_rmsnorm_rope` from `tokenspeed_kernel.ops.layernorm`:

```python
q_out, k_out = staged_qk_rmsnorm_rope(q, k, q_weight, k_weight, cos, sin, eps, out)
```

All arguments are explicit. The kernel performs per-head RMSNorm and full
NeoX RoPE, pairing the first and second halves of each 128-element head.
The operation has a staged BF16 arithmetic contract:

1. Convert weights and rotary factors to BF16.
2. Compute the mean square and inverse RMS in FP32; round normalized values
   to BF16.
3. Multiply by the BF16 weights and round to BF16.
4. Compute each cosine/sine product and round each product to BF16.
5. Add or subtract the rotary products and round to BF16.

Fusion must preserve these intermediate roundings. The existing
`fused_qk_rmsnorm_rope` combines normalization and affine multiplication before
rounding, then combines FP32 rotary products before its final store. It has a
different numerical contract and uses positions plus a packed rotary cache.
The new API takes the already selected full-width cosine and sine rows.

### Inputs and output storage

- Q and K are BF16 `[T, QH * 128]` and `[T, KH * 128]`, with positive head
  counts. Their final axis is contiguous; token strides and storage offsets
  are supported.
- Both weights are contiguous BF16 or FP32 vectors of length 128.
- Cosine and sine factors are BF16 or FP32 `[T, 128]`, with a contiguous
  final axis and optional token strides.
- All tensors share one CUDA or ROCm GPU. Epsilon is finite and positive.
- `out=None` allocates contiguous BF16 outputs. A supplied pair must match
  the input shapes, dtypes and device. Nonempty destinations cannot share
  storage with an input or each other, including disjoint allocation views.
- An empty token batch returns empty outputs without a kernel launch.

The portable Triton backend is registered as `triton_staged_qk_rmsnorm_rope`
in `layernorm.staged_qk_rmsnorm_rope`, with a head-dimension trait of 128.
Warm up the same shapes before CUDA Graph capture. Allocated and supplied
output pairs support replay with updated inputs. FP32 reduction differences
between implementations can affect final BF16 values; numerical tests use
tolerances, with an exact controlled-input test for the rounding stages.

```bash
python -m pytest tokenspeed-kernel/test/ops/test_staged_qk_rmsnorm_rope.py -q
```
