# MiniMLX Context Matrix

- Generated (UTC): 2026-10-27T14:27:08Z
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
generation stop policy (ignore_eos=True, eos_ids=[200020])
context=1024 TTFT=290.9 ms TPOT=10.26 ms PP_TPS=3519.6 TG_TPS=98.24 generated=128/128 first_token=1385 first_token_is_eos=False E2E=1685.7 ms statistic=median
  samples=290.9/10.26/3519.6/98.24/1685.7 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=2.976 GiB
context=4096 TTFT=1209.0 ms TPOT=11.86 ms PP_TPS=3388.0 TG_TPS=85.13 generated=101/128 first_token=1385 first_token_is_eos=False E2E=2487.9 ms statistic=median
  samples=1209.0/11.86/3388.0/85.13/2487.9 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.215 GiB
context=8192 TTFT=2665.3 ms TPOT=14.22 ms PP_TPS=3073.6 TG_TPS=70.99 generated=108/128 first_token=1385 first_token_is_eos=False E2E=4279.4 ms statistic=median
  samples=2665.3/14.22/3073.6/70.99/4279.4 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=3.767 GiB
context=16384 TTFT=6360.3 ms TPOT=18.76 ms PP_TPS=2576.0 TG_TPS=53.74 generated=128/128 first_token=1385 first_token_is_eos=False E2E=8834.7 ms statistic=median
  samples=6360.3/18.76/2576.0/53.74/8834.7 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=2.010 GiB peak=4.638 GiB
suite_peak_memory_gib=4.638

```
