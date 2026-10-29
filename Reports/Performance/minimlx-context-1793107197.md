# MiniMLX Context Matrix

- Generated (UTC): 2026-10-27T13:19:57Z
- Model: `/Users/jizhishi/.minimlx/models/mlx-community--Mistral-Nemo-Instruct-2407-4bit`
- Model type: `mistral`
- Architecture: `MistralForCausalLM`
- Quantization metadata: `not declared in config.json`
- Model disk size: 6.4 GB
- Hardware: Apple M5 Pro
- Unified memory: 64.0 GB
- macOS: 27.0.1
- Python packages: mlx=0.32.2, mlx-lm=0.31.3
- Source revision: 188ed46 (modified working tree)
- Benchmark corpus: `/Users/jizhishi/mycode/ai/minimlx/Resources/BenchmarkCorpus/code_python.txt`
- Corpus SHA-256: `59f6ec70e0b687ec49463dda575322348d7efe0d0770e69ff37b0e3ec6907aa1`
- MTP draft: `disabled`
- DFlash draft: `disabled`
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
generation stop policy (ignore_eos=True, eos_ids=[2])
context=1024 TTFT=922.0 ms TPOT=26.66 ms PP_TPS=1110.6 TG_TPS=37.81 generated=128/128 first_token=11745 first_token_is_eos=False E2E=4370.0 ms statistic=median
  samples=922.0/26.66/1110.6/37.81/4370.0 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=6.417 GiB peak=7.290 GiB
context=4096 TTFT=3720.9 ms TPOT=28.56 ms PP_TPS=1100.8 TG_TPS=35.29 generated=128/128 first_token=11745 first_token_is_eos=False E2E=7409.8 ms statistic=median
  samples=3720.9/28.56/1100.8/35.29/7409.8 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=6.417 GiB peak=7.592 GiB
context=8192 TTFT=8562.0 ms TPOT=31.59 ms PP_TPS=956.8 TG_TPS=31.90 generated=127/128 first_token=2757 first_token_is_eos=False E2E=12604.5 ms statistic=median
  samples=8562.0/31.59/956.8/31.90/12604.5 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=6.417 GiB peak=8.264 GiB
context=16384 TTFT=20599.2 ms TPOT=37.56 ms PP_TPS=795.4 TG_TPS=27.73 generated=25/128 first_token=1293 first_token_is_eos=False E2E=21571.7 ms statistic=median
  samples=20599.2/37.56/795.4/27.73/21571.7 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=6.417 GiB peak=9.469 GiB
suite_peak_memory_gib=9.469

```
