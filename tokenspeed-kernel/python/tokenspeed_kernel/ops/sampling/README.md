# Selected-token logprobs

`try_gather_token_logprobs(logits, tokens)` returns a fresh FP32 `[1]` tensor
for an admitted ordered implementation, otherwise `None`. Callers preserve
their existing fallback when it returns `None`; this API neither samples nor
applies temperature, top-k, top-p, grammar, or RNG changes. It computes raw
distribution `log_softmax(logits)[tokens]` without a full-vocabulary output.
The existing `argmax` API is unchanged.

Current metadata admission is deliberately narrow: NVIDIA SM100, PDL enabled,
FP32 logits `[1,151936]` with stride `(151936,1)`, INT32 tokens `[1]` with
stride `(1,)`, the same current CUDA device, zero storage offsets, 16-byte
aligned pointers, disjoint inputs, no gradients, no lazy negation/conjugation,
and no CUDA autocast. A valid token index is a caller invariant checked by a
device assertion, not a host read. Unsupported metadata returns `None` before
allocation, synchronization, copying, or optional implementation import.

The registered Triton adapter uses the ordinary kernel-package compiler seam.
It allocates fresh per-call `lane_max[1024]`, `row_max[1]`, `lane_sum[1024]`
(8196 logical FP32 scratch bytes) and a separate fresh output. There is no
global/backend tensor cache, prepared object, or cross-call output reuse.
CUDA graphs own capture-time allocations, read live logits/token contents on
replay, and must be released before captured owners. Retained eager outputs
remain unchanged by subsequent calls or graph replays.

Each partial lane visits `lane + 1024*k` in order. Maximum reduction carries
the original value bits and its original reduction-order rank: first NaN wins;
otherwise the rightmost equal maximum wins. This makes duplicate shared
publications bit-identical without canonicalizing NaNs or signed zeros. The
final sum keeps four adjacent lanes' three explicit non-FTZ FP32 additions
before warp/inter-warp reduction. Exponential/FMA/logarithm and PDL dependencies
remain part of the implementation contract. No exceptional input is sanitized.

The stdlib-only contract suite tests fail-closed metadata/selection and fresh
ownership. NVIDIA tests compare raw FP32 words against the compiled public
mathematical operation, including exceptional values and live graph mutation.
Compiler/device upgrades still require artifact and numerical qualification;
component behavior alone does not establish sampler or end-to-end speed.
