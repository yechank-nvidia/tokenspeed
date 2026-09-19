# Layer normalization operations

## Reused Q/K RMSNorm and RoPE

Use `fused_qk_rmsnorm_rope` from
`tokenspeed_kernel.ops.layernorm.triton`. The existing fused implementation
accepts an explicit `staged_bf16` argument; there is no separate staged kernel
or `staged_qk_rmsnorm_rope` export.

```python
q_out, k_out = fused_qk_rmsnorm_rope(
    q, k, q_weight, k_weight, cos_sin_cache, positions, eps,
    num_q_heads, num_kv_heads, head_dim, staged_bf16=True,
)
```

The packed cache contains cosine and sine halves indexed by `positions`.
Staged mode requires BF16 Q/K. It rounds weights, factors, normalized values,
affine results, and each rotary product to BF16 before the next operation.
The ordinary mode keeps its FP32 intermediates. Pass the flag explicitly;
existing DFlash callers use `False`. Outputs are newly allocated contiguous
Q/K tensors. Token strides and even head dimensions are supported; empty
token batches return without a launch.

Both modes share the fused kernel, using eight heads per program in staged
mode and one otherwise. Tests cover both modes, non-contiguous input,
non-power-of-two dimensions, and controlled staged rounding under graph replay.

```bash
python -m pytest tokenspeed-kernel/test/ops/test_layernorm.py -k fused_qk -q
```

This branch contains a source refactor, not a new measured speedup. GPU
correctness, graph replay and performance must be checked on the built candidate.
