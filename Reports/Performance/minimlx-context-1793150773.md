# MiniMLX Context Matrix

- Generated (UTC): 2026-10-28T01:26:13Z
- Model: `/Users/jizhishi/.minimlx/models/mlx-community--Phi-4-mini-instruct-4bit`
- Model type: `phi3`
- Architecture: `Phi3ForCausalLM`
- Quantization metadata: `{"group_size":64,"bits":4}`
- Model disk size: 2.0 GB
- Hardware: Apple M5 Pro
- Unified memory: 64.0 GB
- macOS: 27.0.1
- Python packages: mlx=0.32.2, mlx-lm=0.31.4.dev132+g94cdcae13
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
metric definitions (oMLX single-request fallback) (TTFT=request start to first generated token; PP_TPS=prompt_tokens/TTFT when prompt_tps is unavailable; decode interval=first to last generated-token timestamp; TPOT=decode_interval/(completion_tokens-1); TG_TPS=completion_tokens/decode_interval; E2E=benchmark call wall time)
runtime kernels (q4_native=True, fa256_native=True, q4_dense_disabled=True, q4_mlp_fusion_disabled=True)
decode backend (autoregressive)
runtime route (text-only mlx-lm)
prefill/decode options (skip_intermediate_logits_requested=True, kv_bits=off, kv_start=n/a)
generation stop policy (ignore_eos=False, eos_ids=[199999, 200020])
context=1024 TTFT=289.7 ms TPOT=10.45 ms PP_TPS=3535.0 TG_TPS=96.42 generated=128/128 first_token=65538 first_token_is_eos=False E2E=1625.7 ms statistic=median
  samples=289.7/10.45/3535.0/96.42/1625.7 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=2.929 GiB
context=4096 TTFT=1181.1 ms TPOT=11.99 ms PP_TPS=3467.8 TG_TPS=84.04 generated=128/128 first_token=2958 first_token_is_eos=False E2E=2712.3 ms statistic=median
  samples=1181.1/11.99/3467.8/84.04/2712.3 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.596 GiB
context=8192 TTFT=2599.7 ms TPOT=14.32 ms PP_TPS=3151.1 TG_TPS=70.37 generated=128/128 first_token=4421 first_token_is_eos=False E2E=4425.9 ms statistic=median
  samples=2599.7/14.32/3151.1/70.37/4425.9 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.853 GiB
context=16384 TTFT=6213.8 ms TPOT=18.86 ms PP_TPS=2636.7 TG_TPS=53.45 generated=128/128 first_token=2697 first_token_is_eos=False E2E=8615.1 ms statistic=median
  samples=6213.8/18.86/2636.7/53.45/8615.1 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=4.639 GiB
suite_peak_memory_gib=4.639

```
