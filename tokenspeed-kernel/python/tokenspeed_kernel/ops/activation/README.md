# Activation operations

## Biased attention output gate

Import `attention_gate_mul` from `tokenspeed_kernel.ops.activation` and call
`attention_gate_mul(output, gate, head_bias, floor, temperature)` with all
arguments explicit. It updates and returns `output`:

```text
output[t, h, d] *= floor + (1 - floor) * sigmoid((gate[t, h, d] + head_bias[h]) / temperature)
```

The biased gate reuses the existing `sigmoid_mul` kernel rather than a
separate device implementation. Its FP32 operations round separately, with
an FP32 reciprocal for scalar division and contraction disabled; the final
store rounds to the output dtype. The ordinary unbiased `sigmoid_mul` call
retains its existing sigmoid arithmetic.

- `output`: contiguous BF16, FP16 or FP32 `[T, H * D]`.
- `gate`: matching dtype, either `[T, H * D]` or `[T, H, D]`. Token and head
  strides may differ from a contiguous tensor; the final stride must be one.
- `head_bias`: contiguous BF16 or FP32 `[H]`, with `H > 0`.
- All tensors share a CUDA or ROCm device.
- `floor` must be finite; `temperature` must be finite and positive.
- Nonempty output storage must be separate from both inputs, including
  disjoint views of the same allocation. Contiguous storage offsets are allowed.
- Empty output tensors return without a kernel launch.

The portable Triton adapter remains registered as `triton_attention_gate_mul`
in `activation.attention_gate_mul`; both gate APIs dispatch the shared
`_sigmoid_mul_kernel`. Warm up the input shape before CUDA Graph
capture. Replay reads the current output and gate values, so callers should
refresh the output before each replay when repeated gating is not intended.

Run the public API, numerical and graph tests with:

```bash
python -m pytest tokenspeed-kernel/test/ops/test_activation.py -q
```
