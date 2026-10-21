# MiniMLX Context Matrix

- Generated (UTC): 2026-10-19T01:50:34Z
- Model: `/Users/jizhishi/.minimlx/models/mlx-community--Qwen3.8-27B-4bit`
- Model type: `qwen3_5`
- Architecture: `Qwen3_5ForConditionalGeneration`
- Quantization metadata: `{"group_size":64,"bits":4,"mode":"affine"}`
- Model disk size: 15.0 GB
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
generation stop policy (ignore_eos=True, eos_ids=[248044, 248046])
[mlp-native] route x=(1, 1024, 5120) gate_up=(34816) down=5120
context=1024 TTFT=2188.4 ms TPOT=58.44 ms PP_TPS=467.9 TG_TPS=17.24 generated=128/128 first_token=2958 first_token_is_eos=False E2E=9621.0 ms statistic=median
  samples=2188.4/58.44/467.9/17.24/9621.0 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=14.094 GiB peak=15.853 GiB
context=4096 TTFT=8422.4 ms TPOT=57.63 ms PP_TPS=486.3 TG_TPS=17.49 generated=128/128 first_token=1817 first_token_is_eos=False E2E=15751.4 ms statistic=median
  samples=8422.4/57.63/486.3/17.49/15751.4 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=14.094 GiB peak=16.127 GiB
context=8192 TTFT=16751.5 ms TPOT=58.48 ms PP_TPS=489.0 TG_TPS=17.24 generated=128/128 first_token=14116 first_token_is_eos=False E2E=24188.4 ms statistic=median
  samples=16751.5/58.48/489.0/17.24/24188.4 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=14.094 GiB peak=16.564 GiB
context=16384 TTFT=35818.4 ms TPOT=63.09 ms PP_TPS=457.4 TG_TPS=15.98 generated=128/128 first_token=1590 first_token_is_eos=False E2E=43841.3 ms statistic=median
  samples=35818.4/63.09/457.4/15.98/43841.3 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=14.094 GiB peak=17.439 GiB
suite_peak_memory_gib=17.439

```
