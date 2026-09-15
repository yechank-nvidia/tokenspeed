# Experimental vision RMSNorm

This candidate is isolated from the onboarding branch and is not an accepted
serving optimization. It fuses the vision-only nine-operation RMSNorm chain
into one CuteDSL launch while preserving the pinned ATen FP32 reduction and
rounding order. Language-backbone normalization is unchanged.

The fast path requires NVIDIA sm_100, PyTorch 2.13.0+cu130, BF16 [N, 1024],
16 <= N < 2**21, epsilon 1e-6, unit column stride, non-overlapping rows,
contiguous weights, and inference-only execution. Other cases use eager math.

## Recorded measurements

GB200, profiler-free, three alternating baseline/candidate pairs in one
process; three warmups, 100 norm calls or 10 full ViT calls per timing group.
Times include allocation, dispatch, and host launch gaps. Percentages are
medians of paired reductions, not ratios of the displayed role medians.

| Image grid | Norm host median, us | Paired reduction | Full ViT host median, ms | Paired reduction |
| --- | ---: | ---: | ---: | ---: |
| 15 x 52 | 95.314 -> 28.938 | 69.64% | 35.212 -> 31.311 | 11.15% |
| 24 x 36 | 91.051 -> 27.693 | 69.59% | 35.014 -> 31.322 | 10.54% |
| 88 x 32 | 90.911 -> 27.063 | 70.10% | 52.054 -> 48.411 | 6.98% |

NCU confirms nine launches become one. Summed serialized replay durations
are 50.848 -> 8.384 us and 70.240 -> 10.432 us for the small and large grids;
these are profiler replay numbers, not live serving latency.

The measured prototype matched all 9 captured norms, 195 fresh norm leaves,
and 3 complete ViT outputs bitwise on the tested inputs. The packaged adapter
also passed 9 norm and 3 complete ViT comparisons with 204 observed calls.
Those results are bounded evidence, not universal floating-point equivalence.

Kernel code here is the measured prototype with repository formatting only.
The commit records the original source and result digests. Existing captures,
raw timing groups, and profiler reports remain external; no measurements
were repeated solely to organize the commit.

## Remaining gates

E2E latency/throughput, fixed-prefix logits, full accuracy, and serving
reproducibility are not validated for this candidate. Do not infer a server
speedup from the module reductions or enable it in the onboarding branch.

Run the portable metadata/fallback and bounded CUDA regression tests with:

```bash
PYTHONPATH=python:tokenspeed-kernel/python python -m pytest \\
  tokenspeed-kernel/test/ops/test_vision_rmsnorm.py
```
