# Long-context prefill baseline (2026-09-24)

This is the accepted MiniMLX long-context baseline after aligning the greedy
prefill path with `mlx_lm.generate_step` stream semantics. Do not change the
prefill stream placement, attention dispatch, GDN kernel, or benchmark settings
without a source-backed hypothesis and a measured improvement against this
record.

## Reproduction

Run from the project root, without extra arguments or environment variables:

```sh
./.venv/bin/python tools/benchmark_context_matrix.py
```

Configuration at the time of measurement: warmup 32, repeats 5, median,
prefill step 2048, model
`/path/to/Qwen3.5-4B-MLX-4bit`.
The benchmark printed `q4_native=True`, `fa256_native=True`,
`q4_dense_disabled=False`, `q4_mlp_fusion_disabled=True`. On this GPU, native
Q4 NAX reports unavailable and correctly falls back.

## Accepted measurement

| Context | TTFT | TPOT | TG_TPS | E2E |
|---:|---:|---:|---:|---:|
| 1,024 | 2,787.6 ms | 33.16 ms | 30.16 | 7,031.7 ms |
| 4,096 | 10,489.2 ms | 33.87 ms | 29.52 | 14,824.8 ms |
| 8,192 | 23,751.4 ms | 35.67 ms | 28.03 | 28,317.4 ms |
| 16,384 | 49,069.4 ms | 39.10 ms | 25.57 | 54,074.5 ms |

Prior same-machine oMLX reference records: 8K about 23,763 ms and 16K about
48,617 ms. These are historical single-run references, not a matched median
series: MiniMLX is effectively tied at 8K and about 0.9% slower at 16K, within
the uncertainty of comparing these measurement sets. At 4K MiniMLX is faster
than the prior oMLX record (~11,440 ms).

## Change that established this baseline

In `python/minimlx_runtime/adapters/qwen35.py`, the greedy fast path now keeps
the complete prefill loop, cache-state evaluation, chunk synchronization, and
first-token computation inside `mlx_lm.generate.generation_stream`. The prior
implementation submitted prefill work on the ambient stream but synchronized
the generation stream at chunk boundaries. This differed from
`mlx_lm.generate_step`, whose prefill loop runs inside that generation stream.
After the fix, the 16K median fell from 59,138.5 ms to 49,069.4 ms and the 8K
median from 26,876.9 ms to 23,751.4 ms, with kernel switches unchanged.

## Lock fingerprint

SHA-256 at acceptance:

```text
5fb8d5f872fa83da70b65b5f7e03d616f433b6af2bb9d7060b78b877a9629b32  python/minimlx_runtime/adapters/qwen35.py
6f4cd4bfeb66763177cbfd581320bdefdafea8b9798ee08bff674a06fe4abfaa  tools/benchmark_context_matrix.py
```

This project directory has no Git metadata, so this document and these hashes
are the local baseline marker rather than a Git tag. Preserve a copy of this
version before future performance experiments.

## Optimization decision

No further production optimization is accepted at this point. The measured
long-context result is at parity with the available oMLX reference, and source
inspection found no concrete remaining FA256/GDN algorithm mismatch with a
credible material speedup. Any future attempt should first name the exact
source-level difference, predict its impact, and use a narrow correctness or
microbenchmark before changing the production path.
