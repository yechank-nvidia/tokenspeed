# MoE routing operators

## `moe.group_mask` operator

The kept-group expert mask of grouped (group-limited) top-k routing: the operator
`("moe", "group_mask")` behind the facade
`tokenspeed_kernel.ops.moe.group_mask.moe_group_mask(choices, group_ids,
group_scores, grouped, *, solution, override)`. Both path arguments are
keyword-only without defaults; a caller passes `solution=None, override=None`
for the ranked default.

Contract: `choices` `[rows, experts]` FP32 biased scores, `group_ids` `[rows,
topk_groups]` int64 kept-group indices (values in `[0, num_groups)` are a producer
precondition), `group_scores` `[rows, num_groups]` FP32 and `grouped` the
`[rows, num_groups, experts // num_groups]` view of `choices`. The facade returns
a fresh FP32 `[rows, experts]` tensor equal, byte for byte, to
`choices.masked_fill(~keep, -inf)` where `keep` marks the experts of the kept
groups; the inputs are not modified. Two kernels are registered:

- `torch_group_mask` (solution `torch`, `Priority.PORTABLE`, any device): the
  three torch mask statements plus the `masked_fill` they fed, admitting every
  shape. The ranked default and the production path.
- `triton_group_mask_bits` (solution `triton`, `Priority.REFERENCE`, NVIDIA
  only; `ops/moe/triton/group_mask.py`): one program per row copies the 256 FP32
  words as uint32 and replaces the words of excluded groups by `0xFF800000` (the
  bit pattern of FP32 `-inf`); no floating-point arithmetic, so kept words (NaN
  payloads, `-0.0`) are copied bit for bit. Never auto-selected while
  `torch_group_mask` is registered; reached only by name:
  `moe_group_mask(..., override="triton_group_mask_bits")`,
  `kernel_override("moe", "group_mask", "triton_group_mask_bits")`,
  `TOKENSPEED_KERNEL_OVERRIDE_MOE_GROUP_MASK=triton_group_mask_bits`, or the
  TokenSpeed server flag `--kernel-override moe.group_mask=triton_group_mask_bits`.
  Both registrations share the FP32/int64 signature, so switching by name never
  changes the facade's filtering.

Admission of the Triton kernel, declared as spec traits and re-stated by the host
guard `group_mask_bits_rejection`: `experts == 256`, `num_groups == 8`,
`topk_groups == 4`, all four tensors contiguous FP32/int64 CUDA tensors on one
device, and any positive row count (`rows` is only the grid size; there is no
`rows` trait). The host guard alone refuses `rows == 0`, so an empty grid is never
launched (`torch_group_mask` serves that call). Any other call under the by-name
override raises `ValueError` with the reason; there is no in-function fallback.
The kernel is a correctness reference (REFERENCE band, never ranked) for opt-in
A/B runs and shape-restricted probes, not a serving default.

Its select statement is the shared grouped top-k select
(`ops/moe/triton/minimax_topk.py`) with the `-inf` operand as the uint32 word.
Correctness is established by the GPU byte-equality test
(`test/nvidia/ops/moe/test_group_mask_bits.py`, rows 1..4096 including 3, 5, 7
and 33), not by a binary hash.
