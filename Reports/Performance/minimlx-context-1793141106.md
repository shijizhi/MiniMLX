# MiniMLX Context Matrix

- Generated (UTC): 2026-10-27T22:45:06Z
- Model: `/Users/jizhishi/.minimlx/models/mlx-community--Phi-4-mini-instruct-4bit`
- Model type: `phi3`
- Architecture: `Phi3ForCausalLM`
- Quantization metadata: `{"group_size":64,"bits":4}`
- Model disk size: 2.0 GB
- Hardware: Apple M5 Pro
- Unified memory: 64.0 GB
- macOS: 27.0.1
- Python packages: mlx=0.32.2, mlx-lm=0.31.3
- Source revision: 188ed46 (modified working tree)
- Benchmark corpus profile: `code_python`
- Benchmark corpus file: `/Users/jizhishi/mycode/ai/minimlx/Resources/BenchmarkCorpus/code_python.txt`
- Corpus SHA-256: `59f6ec70e0b687ec49463dda575322348d7efe0d0770e69ff37b0e3ec6907aa1`
- MTP draft: `disabled`
- DFlash draft: `disabled`
- Warm-up: 32 tokens at 2K context
- Measurement: 1 run(s) per context; median and raw samples reported
- Generation limit: 128 tokens
- Prefill step: 2048 tokens
- Contexts: [1024, 4096, 8192, 16384]
- Prompt protocol: oMLX-aligned corpus completion
- Ignore EOS: false
- Runtime: MiniMLX only

## Raw benchmark output

```text
prompt protocol (oMLX-aligned corpus=code_python, max_context_tokens=16384)
MiniMLX context matrix (warmup_prompt=32 tokens, warmup_output=8, measured_runs=1, step=2048)
metric definitions (TTFT=wall time to first generated token; PP_TPS=engine prompt_tps; TPOT/TG_TPS=engine generation_tps; E2E=benchmark call wall time)
runtime kernels (q4_native=True, fa256_native=True, q4_dense_disabled=True, q4_mlp_fusion_disabled=True)
decode backend (autoregressive)
runtime route (text-only mlx-lm)
prefill/decode options (skip_intermediate_logits_requested=True, kv_bits=off, kv_start=n/a)
generation stop policy (ignore_eos=False, eos_ids=[200020])
context=1024 TTFT=384.5 ms TPOT=11.08 ms PP_TPS=3254.3 TG_TPS=90.94 generated=128/128 first_token=480 first_token_is_eos=False E2E=1804.6 ms statistic=median
  samples=384.5/11.08/3254.3/90.94/1804.6 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=2.976 GiB
context=4096 TTFT=1350.3 ms TPOT=12.84 ms PP_TPS=3203.3 TG_TPS=78.52 generated=128/128 first_token=16 first_token_is_eos=False E2E=2993.0 ms statistic=median
  samples=1350.3/12.84/3203.3/78.52/2993.0 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.643 GiB
context=8192 TTFT=2877.5 ms TPOT=15.32 ms PP_TPS=2918.5 TG_TPS=65.77 generated=128/128 first_token=51 first_token_is_eos=False E2E=4835.5 ms statistic=median
  samples=2877.5/15.32/2918.5/65.77/4835.5 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.791 GiB
context=16384 TTFT=6770.4 ms TPOT=20.32 ms PP_TPS=2445.8 TG_TPS=49.60 generated=128/128 first_token=309 first_token_is_eos=False E2E=9362.0 ms statistic=median
  samples=6770.4/20.32/2445.8/49.60/9362.0 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=4.662 GiB
suite_peak_memory_gib=4.662

```
