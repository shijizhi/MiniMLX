# Q4 prefill research

## Scope

This item targets inference-only prefill projection dispatch. Quantized Q4 weights remain the canonical representation for decode and for unsupported shapes. A BF16 dequantized copy is an optional, strictly gated prefill representation; this is not a Qwen-only correctness assumption.

## Public evidence

- MLX exposes `quantized_matmul` as the native QMM primitive: <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul>
- MLX issue #3251 documents that QMM performance depends strongly on group size and shape; dense matmul after dequantization can win for prefill while QMM remains appropriate for decode: <https://github.com/ml-explore/mlx/issues/3251>
- MLX issue #4265 discusses the small-M regime where QMM cannot amortize weight reads: <https://github.com/ml-explore/mlx/issues/4265>
- MLX issue #3584 records shape-dependent QMM/split-K regressions: <https://github.com/ml-explore/mlx/issues/3584>

These references motivate measurement and dispatch; no implementation is copied from oMLX.

## Implementation

`python/minimlx_runtime/kernels/mlp.py` now caches BF16 dequantized gate/up/down weights once at model load. Gate and up rows are concatenated into one dense matrix, so prefill uses one larger GEMM, splits the result, applies SiLU/gating, and performs the down GEMM. The path is enabled only when:

- quantization mode is affine and all required metadata exists;
- the model exposes the expected MLP projection interface;
- the module is in inference mode;
- the sequence dimension is at least the configured threshold (currently 1024);
- `MINIMLX_DISABLE_Q4_PREFILL_DENSE=1` is not set.

All other calls fall back to the original MLX implementation. The Q4 fusion path remains available as a fallback for models/counts where the dense representation is not installed.

## Isolated measurements on M4

Model: Qwen3.5-0.8B-MLX-4bit, L0, BF16 input, Q4 weights, group size 64.

| sequence | stock QMM MLP | BF16 dense MLP | dense gain |
|---:|---:|---:|---:|
| 1024 | 10.815 ms | 10.250 ms | 0.565 ms |
| 2048 | 20.410 ms | 18.684 ms | 1.726 ms |
| 4096 | 42.037 ms | 36.905 ms | 5.132 ms |

At longer isolated lengths the same direction remains:

- 16K: 170.951 ms QMM vs 153.980 ms BF16 dense;
- 32K: 345.469 ms QMM vs 308.091 ms BF16 dense.

The concatenated gate/up dense projection measured 6.610 ms at the actual 1024-token chunk size and 24.085 ms at 4096 tokens. The full dense MLP result matched the stock QMM result with `max_abs_error=0`; the tool also checks the concatenated gate/up output directly.

## Independent native-Q4 tile probe

The first independent Metal Q4 probe was measured on the same real L0 shape at
4096 tokens. It is useful as a rejection baseline before investing in more
complicated tiling:

| operation | median |
|---|---:|
| stock gate QMM | 14.172 ms |
| stock up QMM | 14.143 ms |
| stock concatenated gate/up QMM | 26.329 ms |
| independent scalar-decode Q4 gate | 72.451 ms |
| independent scalar-decode Q4 up | 72.890 ms |
| independent 8x8 simdgroup probe | 911.743 ms |

The independent kernels were numerically close at BF16 output precision
(`1.56e-2` gate max error, `7.81e-3` up max error), but are substantially
slower than MLX's native QMM. The simdgroup probe is especially unsuitable:
its current matrix layout does not map efficiently to the packed affine-Q4
format. It is not installed or used for dispatch. The practical Q4 prefill
optimization remains cached BF16 dense weights plus the fused gate/up GEMM;
the native-Q4 tile remains a research item rather than a production candidate.

## End-to-end evidence

With the normal 4096-token benchmark and 1024-token prefill chunks, one observed paired comparison was:

- dense enabled: TTFT 2375.5 ms, TPOT 7.94 ms;
- dense disabled: TTFT 2463.4 ms, TPOT 7.93 ms.

That is an 87.9 ms / 3.6% TTFT improvement in that pair, with decode unchanged. A second disabled run was 2492.4 ms, confirming the direction but also showing normal system variance. The long-context measurements above are isolated L0 measurements; full 16K/32K end-to-end runs are still required before marking the item complete.

## Remaining validation

1. Repeat enabled/disabled A/B under the same process and warm-up policy.
2. Test 1K, 4K, 16K and 32K+ contexts, including tail chunks.
3. Verify decode and short-prefill fallback do not regress.
4. Test a model with a different supported MLP shape before broadening the dispatch gate.

## Status

`[~]` Dense Q4 prefill is implemented and shows an isolated and end-to-end gain.
Keep it enabled behind the strict gate. The first independent native-Q4 tile
probe is rejected on speed; long-context repeated A/B validation of the dense
route is still required before marking the broader Q4 item complete.
