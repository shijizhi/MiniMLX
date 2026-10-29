# MiniMLX Context Matrix

- Generated (UTC): 2026-10-27T22:41:59Z
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
context=1024 TTFT=386.5 ms TPOT=11.04 ms PP_TPS=3244.3 TG_TPS=91.29 generated=128/128 first_token=39574 first_token_is_eos=False E2E=1801.1 ms statistic=median
  samples=386.5/11.04/3244.3/91.29/1801.1 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=2.976 GiB
context=4096 TTFT=1347.2 ms TPOT=12.74 ms PP_TPS=3206.0 TG_TPS=79.11 generated=128/128 first_token=7376 first_token_is_eos=False E2E=2977.7 ms statistic=median
  samples=1347.2/12.74/3206.0/79.11/2977.7 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.643 GiB
context=8192 TTFT=2879.5 ms TPOT=15.29 ms PP_TPS=2915.2 TG_TPS=65.92 generated=128/128 first_token=4421 first_token_is_eos=False E2E=4832.6 ms statistic=median
  samples=2879.5/15.29/2915.2/65.92/4832.6 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.791 GiB
context=16384 TTFT=6772.4 ms TPOT=20.09 ms PP_TPS=2444.2 TG_TPS=50.17 generated=128/128 first_token=2697 first_token_is_eos=False E2E=9334.4 ms statistic=median
  samples=6772.4/20.09/2444.2/50.17/9334.4 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=4.662 GiB
suite_peak_memory_gib=4.662

```
