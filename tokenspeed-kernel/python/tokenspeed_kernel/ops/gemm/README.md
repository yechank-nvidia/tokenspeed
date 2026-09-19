# GEMM and GEMV operations

## Dtype-aware decode GEMV

Use the existing `decode_gemv(x, weight, out)` dispatcher for `x @ weight.T`:

```python
from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

result = decode_gemv(x, weight, None)
```

Contiguous single-row FP32 GPU inputs reuse the BF16 row-CTA implementation.
The input is `[1, K]`, the weight is `[N, K]`, and both have matching dtype and
device. Pass `None` to allocate `[1, N]`, or supply a destination with that
shape, matching dtype/device and contiguous storage. Keep destination storage
separate from inputs; the general dispatcher does not diagnose storage aliasing.
Contiguous inputs and destinations may have nonzero storage offsets.

Selection is cached by shape, device kind and dtype, and registry lookup
filters both input signatures. FP32 products and accumulation do not inherit
Torch's reduced-precision matmul setting. FP32 uses a full-row reduction up to
65536 elements and tiled accumulation above that width; BF16 retains its
existing tile configuration. `N = 0` returns without launching a kernel and
`K = 0` produces zeros. Row addresses use 64-bit arithmetic.

Other row counts, unsupported signatures and noncontiguous inputs retain the
existing Torch fallback and its matmul semantics. The standalone
`fp32_decode_gemv` API/registration is removed in favor of this shared entry.
Compare against a floating-point reference with tolerance: a fixed reduction
order need not match a BLAS reduction bitwise.

Warm up shapes before CUDA Graph capture. Replay reads live inputs, and callers
own any supplied output buffer. Run shared dispatch, numerical and graph tests:

```bash
python -m pytest tokenspeed-kernel/test/ops/gemm/test_routed_gemv.py -q
```
