# FA-256 full-attention prefill research

## Why this is the first target

The Qwen3.5-0.8B model used for the current benchmark has:

- 8 query heads
- 2 KV heads
- `head_dim=256`
- 6 full-attention layers at L3, L7, L11, L15, L19 and L23

This exactly matches the shape family targeted by oMLX's Qwen3.5/3.6 FA-256
prefill path.

## Findings from public implementation research

oMLX's public patch is narrowly gated to causal prefill/chunked-prefill, query
length at least 16, head dimension 256, valid GQA, non-quantized KV cache, and
no sinks or array masks. Decode and MTP verification remain on the stock path.
The implementation uses a Steel attention kernel so the score matrix is not
materialized and QK/PV use simdgroup matrix multiplication.

The public implementation also has two important safety mechanisms:

1. It avoids routing tiny query lengths because scanning the full KV cache is
   slower than the normal path.
2. It limits work per dispatch and calibrates the limit. This matters on M1-M4:
   oMLX has documented a pathological slowdown when a single custom dispatch
   scans too much KV data and trips macOS GPU command-buffer preemption.

The bounded O(L) SDPA path is separate. It is primarily a memory-safety route
for long contexts when the normal O(L^2) score transient exceeds the live
memory guard; it should not replace the faster stock route whenever the
transient fits.

## Current local baseline

Run:

```sh
./.venv/bin/python tools/experiment_fa256_attention.py \
  --tokens 4096 --q-heads 8 --kv-heads 2 --head-dim 256
```

This measures the current MLX 0.32.2 stock SDPA with the exact Qwen3.5
full-attention shape. A custom kernel will only be considered successful if it
passes numerical comparison against this output and improves the full-layer
and end-to-end prefill curve.

Initial M4 measurements:

| Context | Stock SDPA |
|---:|---:|
| 4K | 60.982 ms |
| 8K | 235.213 ms |
| 16K | 1154.546 ms |

MLX's public fused switch was then measured on the same shapes on the same
machine:

| Context | Default SDPA | `force_fused=True` | Ratio |
|---:|---:|---:|---:|
| 1K | 3.768 ms | 2.621 ms | 1.44x |
| 2K | 17.439 ms | 9.751 ms | 1.79x |
| 4K | 71.792 ms | 38.229 ms | 1.88x |

The comparison was numerically exact for these BF16 probes (`max_abs_error=0`)
and the route was verified inside the real model: Qwen3.5 full-attention calls
arrived as `(B=1, Q=8, L=1024, D=256)` with `(KV=2)` and had zero fused
fallbacks. However, enabling the route in the full 4K benchmark with the
current 1024-token prefill chunks regressed the end-to-end result from the
roughly 150 tok/s baseline to 126 tok/s. An isolated faster SDPA is therefore
not sufficient when six layers and chunk boundaries are scheduled repeatedly.
The route remains opt-in and is not enabled in the normal benchmark.

The model-level A/B was then repeated with the new runtime `--prefill-step-size`
override, separating chunk scheduling from the attention route:

| route / chunk | TTFT (ms) | TPOT (ms) | TG TPS |
|---|---:|---:|---:|
| stock / 512 | 2211.6 | 7.67 | 130.36 |
| stock / 1024 (baseline) | ~1801 | ~5.87 | ~170.3 |
| stock / 2048 | 2133.7 | 7.11 | 140.70 |
| fused / 1024 | 2060.3 | 7.28 | 137.41 |
| fused / 2048 | 2041.2 | 7.22 | 138.55 |
| fused / 4096 | 2070.7 | 7.13 | 140.24 |

The fused route hit all six full-attention layers with zero fallback at every
tested supported chunk size, but did not beat the stock 1024-chunk model path.
The 4096 case is valid evidence here: it ran successfully and used six fused
calls per measured pass. Therefore the isolated fused-kernel advantage does not
survive the current full-model scheduling and graph interaction. FA-256 remains
opt-in research only; it is not promoted to the default path.

The scaling confirms that this is a meaningful long-context target. The next
experiment is a supported long-prefill model A/B with chunk sizes accepted by
the current generation path. Only a model-level improvement will promote the
native fused route from opt-in.

## First independent Metal kernel

`tools/experiment_fa256_metal.py` now contains a correctness-first causal GQA
kernel. It assigns one threadgroup to each query and uses an online-softmax
recurrence, so it does not materialize the score matrix.

Results:

- 1 token: max absolute error `0`
- 128 tokens: max absolute error `1.5625e-2` in BF16 output
- 128 tokens stock SDPA: `0.579 ms`
- 128 tokens reference kernel: `2.956 ms`

The BF16 error is acceptable for this first numerical reference, but the
reference kernel is about 5x slower because every key uses a scalar
threadgroup reduction. It is deliberately not installed in production. The
next implementation must use tiled Q/K/V loading and simdgroup matrix
multiplication before any model-level A/B test.

### Long-context scaling check

The reference kernel was also measured at the real Qwen shape with the same
kernel family:

| Context | Stock SDPA | Custom tiled online softmax | Relative |
|---:|---:|---:|---:|
| 4K | 67.856 ms | 728.384 ms | 10.7x slower |
| 8K | 269.861 ms | 3109.878 ms | 11.5x slower |

An 8-query query-tile prototype was then tested to remove the one-query-per-
threadgroup structure. It was numerically equivalent at 4K, but measured
`1391.593 ms`, still slower than the existing KV-tiled prototype (`727.832
ms`). The reason is that it repeats scalar dot-product work for every query
inside the threadgroup without using matrix instructions; changing the grid
shape alone is not a sufficient optimization.

The custom path grows with the expected quadratic work but has a much larger
constant. It provides no long-context advantage, so this implementation is
retired as a production candidate. The fact that it avoids materializing the
score matrix is not enough: it still reloads data inefficiently and performs
scalar reductions. The next prototype must parallelize over query tiles and
KV tiles, keep Q/K/V tiles on-chip where possible, and use simdgroup matrix
operations. It must first beat stock SDPA in an isolated 4K/8K test before
being connected to the model.

### SIMD-group MMA prototype result

An independent `simdgroup_matrix<float, 8, 8>` prototype was tested against
the same causal GQA reference. Its first failure was a custom-kernel launch
mapping mistake: MLX's grid is specified in threads, so the x dimension must
be `query_blocks * 256`, matching the working tiled kernels. Fixing that
restored complete output coverage, but the real MMA result still failed the
reference comparison. The remaining error is in the lane-to-matrix-element
mapping/transpose convention, not in dispatch coverage.

The prototype is rejected for production. Its low timings are not valid
evidence until numerical correctness passes; the existing stock SDPA route
remains unchanged. A future MMA attempt must first use a standalone
matrix-layout probe that verifies every loaded matrix element before adding
causal softmax and value accumulation back in.

This direction follows the general IO-aware principle of FlashAttention, but
the implementation remains independent: [FlashAttention](https://arxiv.org/abs/2205.14135)
and [FlashAttention-2](https://arxiv.org/abs/2307.08691) motivate the tiling
and work-partitioning questions; MLX's own SDPA API remains the baseline and
fallback ([MLX scaled dot-product attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)).

## Implementation boundary

We will implement an independent Metal kernel based on the FlashAttention
online-softmax recurrence, with:

- causal masking
- GQA (`q_heads / kv_heads`)
- BF16/FP16 inputs
- FP32 online-softmax accumulators
- Q/K tile and KV dispatch limits
- stock SDPA fallback for unsupported shapes or failures

No oMLX source code will be copied. The public oMLX patch is used only to
identify shape gates, failure modes and benchmark questions.
