# MiniMLX Context Matrix

- Generated (UTC): 2026-10-16T00:53:46Z
- Model: `/Users/jizhishi/.minimlx/models/mlx-community--Qwen2.5-7B-Instruct-4bit`
- Model type: `qwen2`
- Architecture: `Qwen2ForCausalLM`
- Quantization metadata: `not declared in config.json`
- Model disk size: 4.0 GB
- Hardware: Apple M5 Pro
- Unified memory: 64.0 GB
- macOS: 27.0.1
- Python packages: mlx=0.32.2, mlx-lm=0.31.3
- Source revision: dbad46b (modified working tree)
- Benchmark corpus: `/Users/jizhishi/mycode/ai/minimlx/Resources/BenchmarkCorpus/code_python.txt`
- Corpus SHA-256: `59f6ec70e0b687ec49463dda575322348d7efe0d0770e69ff37b0e3ec6907aa1`
- MTP draft: `disabled`
- Warm-up: 32 tokens at 2K context
- Measurement: 1 run(s) per context; median and raw samples reported
- Generation limit: 128 tokens
- Prefill step: 1024 tokens
- Contexts: [1024, 4096, 8192, 16384]
- Prompt protocol: oMLX-aligned raw completion
- Ignore EOS: true
- Runtime: MiniMLX only

## Raw benchmark output

```text
prompt protocol (oMLX-aligned Python raw completion, max_context_tokens=16384)
MiniMLX context matrix (warmup=32 tokens at context=2048, measured_runs=1, step=1024)
runtime kernels (q4_native=True, fa256_native=True, q4_dense_disabled=True, q4_mlp_fusion_disabled=True)
decode backend (autoregressive)
runtime route (text-only mlx-lm)
prefill/decode options (skip_intermediate_logits_requested=True, kv_bits=off, kv_start=n/a)
generation stop policy (ignore_eos=True, eos_ids=[151645])
context=1024 TTFT=545.6 ms TPOT=15.43 ms PP_TPS=1876.7 TG_TPS=65.32 generated=128/128 first_token=2132 first_token_is_eos=False E2E=2571.3 ms statistic=median
  samples=545.6/15.43/1876.7/65.32/2571.3 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=3.993 GiB peak=4.754 GiB
context=4096 TTFT=2073.4 ms TPOT=16.12 ms PP_TPS=1975.5 TG_TPS=62.54 generated=128/128 first_token=2132 first_token_is_eos=False E2E=4184.4 ms statistic=median
  samples=2073.4/16.12/1975.5/62.54/4184.4 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=3.993 GiB peak=4.965 GiB
context=8192 TTFT=4446.4 ms TPOT=17.58 ms PP_TPS=1842.4 TG_TPS=57.34 generated=128/128 first_token=2132 first_token_is_eos=False E2E=6740.9 ms statistic=median
  samples=4446.4/17.58/1842.4/57.34/6740.9 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=3.993 GiB peak=5.231 GiB
context=16384 TTFT=9636.7 ms TPOT=19.27 ms PP_TPS=1700.2 TG_TPS=52.30 generated=128/128 first_token=2132 first_token_is_eos=False E2E=12147.8 ms statistic=median
  samples=9636.7/19.27/1700.2/52.30/12147.8 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=3.993 GiB peak=5.483 GiB
suite_peak_memory_gib=5.483

```
