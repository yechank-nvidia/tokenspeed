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

## `moe.route_epilogue` operator

The selected-weight epilogue of top-k expert routing (the tail of a routing
callback): the operator `("moe", "route_epilogue")` behind the facade
`tokenspeed_kernel.ops.moe.route_epilogue.moe_route_epilogue(scores, ids, *,
renormalize, solution, override)`. All three keyword arguments are keyword-only
without defaults (`renormalize` selects the arithmetic branch and must be a
`bool`); a caller passes `renormalize=renormalize, solution=None, override=None`
for the ranked default.

Contract: `scores` `[rows, experts]` FP32 sigmoid scores and `ids` `[rows, topk]`
int64 selected expert ids (values in `[0, experts)` are a producer precondition).
The facade returns `(weights, ids)`: a fresh FP32 `[rows, topk]` tensor equal,
byte for byte, to `scores.gather(-1, ids)` divided per row by `(sum + 1e-20)` when
`renormalize`, and the ids as a fresh int32 tensor; the inputs are not modified.
Two kernels are registered:

- `torch_route_epilogue` (solution `torch`, `Priority.PORTABLE`, any device): the
  three torch tail statements verbatim (`weights = scores.gather(-1, ids)`;
  `weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)` under
  `renormalize`; `ids.to(torch.int32)`), admitting every shape and both
  `renormalize` values. The ranked default and the production path.
- `triton_route_epilogue` (solution `triton`, `Priority.REFERENCE`, NVIDIA only;
  `ops/moe/triton/route_epilogue.py`): one program per row gathers the eight
  selected scores and, under `NORMALIZE`, divides them by their sum plus `1e-20`
  with the sum spelled in the order Torch's CUDA reduction uses --
  `((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))`, then `+ 1e-20`, then a correctly
  rounded division, each step one `add.rn.f32` (inline PTX) or `div.rn.f32`
  (`tl.div_rn`); without `NORMALIZE` no floating-point arithmetic runs. Never
  auto-selected while `torch_route_epilogue` is registered; reached only by name:
  `moe_route_epilogue(..., override="triton_route_epilogue")`,
  `kernel_override("moe", "route_epilogue", "triton_route_epilogue")`,
  `TOKENSPEED_KERNEL_OVERRIDE_MOE_ROUTE_EPILOGUE=triton_route_epilogue`, or the
  TokenSpeed server flag
  `--kernel-override moe.route_epilogue=triton_route_epilogue`. Both
  registrations share the FP32/int64 signature, so switching by name never
  changes the facade's filtering.

Admission of the Triton kernel, declared as spec traits and re-stated by the host
guard `route_epilogue_rejection`: `experts == 256`, `topk == 8`, `renormalize` a
`bool` (either value; both are compiled variants), `scores` and `ids` contiguous
FP32/int64 CUDA tensors on one device, and any positive row count (`rows` is only
the grid size; there is no `rows` trait). The host guard alone refuses
`rows == 0`, so an empty grid is never launched (`torch_route_epilogue` serves
that call). Any other call under the by-name override raises `ValueError` with
the reason; there is no in-function fallback. The kernel is a correctness
reference (REFERENCE band, never ranked) for opt-in A/B runs and shape-restricted
probes, not a serving default. Every positive row count is admitted from the
start; the GPU byte-equality test (`test/nvidia/ops/moe/test_route_epilogue_cuda.py`)
covers fifteen row counts including 3, 5, 7 and 33, both `renormalize` values,
with ties, negatives, `-inf` and denormals.

The bitwise contract rests on Torch: on torch `2.13.0+cu130`
(`ATen/native/cuda/Reduce.cuh`) the CUDA `sum(dim=-1)` of a contiguous FP32
`[rows, 8]` takes no input vectorization, uses eight lanes per row with one
element each and combines them by shuffle-down offsets 4, 2, 1 -- the tree the
kernel reproduces; `+ 1e-20` is a separate elementwise add and `/` is
`DivFunctor`'s `a / b` (non-FTZ `div.rn.f32`). A Torch change to that order, the
vectorization threshold, the block sizing or an FTZ/fast-division build flag
voids the contract without touching the kernel, so the GPU test runs the pin
`reduction_order_mismatch(device)` first (the fixture rows include the rounding
row `(1, 2^-24, 2^-25, 2^-26, .5, 2^-23, 2^-24, 2^-25)` on which different orders
give different bits) and fails every byte-equality test on a mismatch -- never a
skip. A mismatch means the Triton kernel cannot be used on that build until it is
re-pinned. Nothing probes the order at run time (a routing callback may run under
CUDA-graph capture, where a host sync is illegal). The CPU test pins the tree
literally in the kernel source by AST.

The kernel body is the straightforward per-row gather and pinned-order sum;
`enable_reflect_ftz=False` is pinned explicitly (the kernel links no libdevice, so
the compiled PTX carries no `.ftz` and the pin leaves the PTX unchanged).
Correctness is established by the byte-equality tests, not by a binary hash.
