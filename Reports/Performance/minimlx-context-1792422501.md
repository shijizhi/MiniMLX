# MiniMLX Context Matrix

- Generated (UTC): 2026-10-19T15:08:21Z
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
- DFlash draft: `z-lab--Qwen3.8-27B-DFlash2`
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
decode backend (DFlash, draft=z-lab--Qwen3.8-27B-DFlash2, block_size=5)
runtime route (text-only mlx-lm)
prefill/decode options (skip_intermediate_logits_requested=True, kv_bits=off, kv_start=n/a)
generation stop policy (ignore_eos=True, eos_ids=[248044, 248046])
[mlp-native] route x=(1, 1024, 5120) gate_up=(34816) down=5120
context=1024 TTFT=2136.6 ms TPOT=21.11 ms PP_TPS=479.3 TG_TPS=47.75 generated=128/128 first_token=2958 first_token_is_eos=False E2E=4934.4 ms statistic=median
  samples=2136.6/21.11/479.3/47.75/4934.4 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=17.679 GiB peak=19.526 GiB
context=4096 TTFT=8566.6 ms TPOT=21.06 ms PP_TPS=478.1 TG_TPS=47.87 generated=128/128 first_token=1817 first_token_is_eos=False E2E=11356.8 ms statistic=median
  samples=8566.6/21.06/478.1/47.87/11356.8 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=17.679 GiB peak=19.930 GiB
context=8192 TTFT=17507.6 ms TPOT=22.07 ms PP_TPS=467.9 TG_TPS=45.67 generated=128/128 first_token=14116 first_token_is_eos=False E2E=20425.9 ms statistic=median
  samples=17507.6/22.07/467.9/45.67/20425.9 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=17.679 GiB peak=20.335 GiB
context=16384 TTFT=37702.3 ms TPOT=23.42 ms PP_TPS=434.6 TG_TPS=43.04 generated=128/128 first_token=1590 first_token_is_eos=False E2E=40794.5 ms statistic=median
  samples=37702.3/23.42/434.6/43.04/40794.5 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=17.679 GiB peak=21.210 GiB
suite_peak_memory_gib=21.210

```
