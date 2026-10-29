# MiniMLX Context Matrix

- Generated (UTC): 2026-10-27T23:04:32Z
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
metric definitions (oMLX single-request fallback) (TTFT=request start to first generated token; PP_TPS=prompt_tokens/TTFT when prompt_tps is unavailable; decode interval=first to last generated-token timestamp; TPOT=decode_interval/(completion_tokens-1); TG_TPS=completion_tokens/decode_interval; E2E=benchmark call wall time)
runtime kernels (q4_native=True, fa256_native=True, q4_dense_disabled=True, q4_mlp_fusion_disabled=True)
decode backend (autoregressive)
runtime route (text-only mlx-lm)
prefill/decode options (skip_intermediate_logits_requested=True, kv_bits=off, kv_start=n/a)
generation stop policy (ignore_eos=False, eos_ids=[200020])
context=1024 TTFT=359.2 ms TPOT=10.08 ms PP_TPS=2850.4 TG_TPS=99.96 generated=128/128 first_token=13535 first_token_is_eos=False E2E=1651.1 ms statistic=median
  samples=359.2/10.08/2850.4/99.96/1651.1 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=2.976 GiB
context=4096 TTFT=1234.1 ms TPOT=11.76 ms PP_TPS=3319.1 TG_TPS=85.72 generated=128/128 first_token=2958 first_token_is_eos=False E2E=2739.2 ms statistic=median
  samples=1234.1/11.76/3319.1/85.72/2739.2 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.643 GiB
context=8192 TTFT=2668.1 ms TPOT=14.12 ms PP_TPS=3070.3 TG_TPS=71.40 generated=128/128 first_token=51 first_token_is_eos=False E2E=4471.6 ms statistic=median
  samples=2668.1/14.12/3070.3/71.40/4471.6 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.791 GiB
context=16384 TTFT=6267.4 ms TPOT=18.59 ms PP_TPS=2614.1 TG_TPS=54.23 generated=128/128 first_token=314 first_token_is_eos=False E2E=8638.5 ms statistic=median
  samples=6267.4/18.59/2614.1/54.23/8638.5 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=4.662 GiB
suite_peak_memory_gib=4.662

```
