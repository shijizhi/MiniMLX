# MiniMLX model support matrix

The broader support and performance process is defined in
[`Model_Support_and_Performance_Plan.md`](Model_Support_and_Performance_Plan.md).
Historical and future MiniMLX/oMLX numbers are maintained in
[`Model_Benchmark_Comparison.md`](Model_Benchmark_Comparison.md).
The current hardware authority for this matrix is a MacBook Air M4 with 24 GB
unified memory; 32K results are conditional on memory safety.

This matrix tracks model-family compatibility separately from performance
optimization. A model is not considered supported from `config.json` alone:
it must load through its adapter, execute a short prefill and decode, and pass
the family-specific shape checks.

## Initial Qwen3.5 coverage

| Model | Format | Architecture | Adapter | Short generation | Status |
|---|---|---|---|---|---|
| Qwen3.5-0.8B-MLX-4bit | affine Q4, group 64 | `qwen3_5` | Qwen35Adapter | previously validated | Supported |
| Qwen3.5-0.8B-MLX-bf16 | BF16 | `qwen3_5` | Qwen35Adapter | PASS: 4 tokens generated | Supported |
| Qwen3.5-4B-MLX-4bit | affine Q4, group 64 | `qwen3_5` | Qwen35Adapter | PASS: 4 tokens generated | Supported |
| Qwen3.5-4B-MLX-bf16 | BF16 | `qwen3_5` | Qwen35Adapter | PASS: 4 tokens generated | Supported (1K/4K/16K; 7.899 GiB MLX peak) |
| Qwen3.5-35B-A3B-OptiQ-4bit-REAP-19B | mixed Q4/Q8, group 64 | `qwen3_5_moe` | Qwen35Adapter | PASS: 4 tokens at 1K/4K/16K; 11.586 GiB smoke peak | Context-pass through 16K; per-context peak memory unreported |

The BF16 model intentionally skips Q4-only MLP prefill paths. It should still
use the common Qwen3.5 GDN, full-attention, cache, tokenizer, and generation
logic. No BF16-specific performance optimization is being added in this phase.

## BF16 verification result

- 24 layers: 18 linear-attention/GDN and 6 full-attention
- Q4 MLP layers detected: 0
- Prompt: token `[9419]`
- Generated tokens: `[11, 271, 40, 1044]`
- Text: `",\\n\\nI am"`
- Prefill: 107.695 ms
- Decode: 49.560 ms
- Result: `compatibility=PASS`

## BF16 context matrix

Configuration: `warmup=32`, `repeats=5`, median, generated tokens `128`,
prefill step `1024`, corpus `code_python.txt`.

| Context | TTFT | TPOT | TG TPS | E2E |
|---:|---:|---:|---:|---:|
| 1K | 448.8 ms | 16.68 ms | 59.97 | 2583.3 ms |
| 2K | 871.4 ms | 16.75 ms | 59.70 | 3015.6 ms |
| 4K | 1752.8 ms | 17.10 ms | 58.50 | 3941.0 ms |
| 8K | 3829.2 ms | 17.54 ms | 57.02 | 6074.0 ms |
| 16K | 8939.7 ms | 18.73 ms | 53.40 | 11336.9 ms |

Result: all tested context lengths passed. The higher BF16 TPOT relative to
the 4-bit model is expected from the larger unquantized weight bandwidth and
does not indicate a compatibility failure.

## Qwen3.5-4B BF16 smoke result

- Architecture: `Qwen3_5ForConditionalGeneration`
- Layers: 32 total, 24 linear-attention/GDN and 8 full-attention
- Q4 MLP layers: 0 (expected for BF16)
- Prompt tokens: `[9419]`
- Generated tokens: `[11, 353, 2688, 4313]`
- Text: `", I'm trying"`
- Prefill: 867.459 ms
- Decode: 279.811 ms
- Result: `compatibility=PASS`
- Context checks: 1K, 4K, and 16K; each completed with 4 generated tokens.
- Context-check TTFT: 2572.2 ms / 10071.8 ms / 40317.6 ms.
- Context-check E2E: 2833.7 ms / 10336.6 ms / 40609.2 ms.
- Smoke peak MLX memory: 7.899 GiB on the 24 GB MacBook Air M4.
- These single-run short-generation
  timings are compatibility evidence, not a performance comparison.

## BF16 MiniMLX / oMLX comparison

Assuming the screenshot uses the same model, prompt corpus, 128 generated
tokens, warm-up, repeats, and single-request settings:

| Context | MiniMLX TTFT | oMLX TTFT | MiniMLX TPOT | oMLX TPOT | MiniMLX TG TPS | oMLX TG TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 448.8 ms | 501.4 ms | 16.68 ms | 18.7 ms | 59.97 | 53.8 |
| 4K | 1752.8 ms | 2506.1 ms | 17.10 ms | 19.0 ms | 58.50 | 53.1 |
| 8K | 3829.2 ms | 4798.6 ms | 17.54 ms | 19.7 ms | 57.02 | 51.2 |
| 16K | 8939.7 ms | 10992.6 ms | 18.73 ms | 21.2 ms | 53.40 | 47.5 |

MiniMLX is faster in both Prefill and Decode at all four lengths. Relative to
oMLX, TTFT is lower by approximately 10.5%, 30.0%, 20.2%, and 18.7%; TPOT is
lower by approximately 10.8%, 10.0%, 11.0%, and 11.7%. This is a performance
comparison, not part of the compatibility acceptance gate.

## Verification command

```bash
cd /Users/jizhishi/mycode/ai/minimlx
./.venv/bin/python tools/verify_model_support.py \
  /Users/jizhishi/.omlx/models/mlx-community/Qwen3.5-0.8B-MLX-bf16
```

## Acceptance criteria

1. Descriptor resolves to the Qwen3.5 family adapter.
2. Model loads without quantization-only assumptions.
3. Layer counts and linear/full-attention split are reported.
4. Short prefill and greedy decode complete.
5. Generated output is non-empty and the process exits cleanly.
