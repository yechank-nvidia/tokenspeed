# GEMM and GEMV operations

## Single-row FP32 GEMV

`fp32_decode_gemv(x, weight, out)` computes `x @ weight.T` with a portable
Triton kernel. Import it from `tokenspeed_kernel.ops.gemm` and pass `out`
explicitly: `None` allocates an output, while a tensor reuses its storage.

- `x`: contiguous FP32 `[1, K]`, with `1 <= K <= 65536`.
- `weight`: contiguous FP32 `[N, K]` on the same CUDA or ROCm device.
- `out`: contiguous FP32 `[1, N]` on that device, or `None`.
- The returned tensor has shape `[1, N]`. A supplied destination is returned
  directly. `N = 0` returns an empty tensor without launching a kernel.
- Nonempty destinations must not share storage with either input, including
  disjoint views of the same allocation. Inputs may have contiguous storage
  offsets.

One thread block reduces each weight row in FP32. Masked loads handle widths
that are not powers of two. The reduction order is fixed for a given launch,
but can differ from a BLAS reduction, so comparisons use floating-point
tolerances rather than require identical output bits. Weight row addresses
use 64-bit arithmetic.

The implementation is registered as `triton_fp32_decode_gemv` under
`gemm.fp32_decode_gemv`. It is an explicit single-row API; callers of the
existing matrix-multiplication APIs keep their current dispatch.

Warm up the same shapes before CUDA Graph capture. Both destination modes
support capture and replay, including updates to the input tensors.

Run correctness, validation, registration and graph tests with:

```bash
python -m pytest tokenspeed-kernel/test/ops/gemm/test_fp32_decode_gemv.py -q
```
