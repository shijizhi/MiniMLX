# MiniMLX Context Matrix

- Generated (UTC): 2026-10-19T01:53:45Z
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
- MTP draft: `mlx-community--Qwen3.8-27B-MTP-4bit`
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
[mtp] MiniMLX native Q4/FA256 kernels disabled; using coherent MLX-VLM target path
prompt protocol (oMLX-aligned Python raw completion, max_context_tokens=16384)
MiniMLX context matrix (warmup=32 tokens at context=2048, measured_runs=1, step=1024)
runtime kernels (q4_native=True, fa256_native=True, q4_dense_disabled=True, q4_mlp_fusion_disabled=True)
decode backend (Qwen3.5 MTP, draft=mlx-community--Qwen3.8-27B-MTP-4bit, block_size=3)
runtime route (text-only mlx-lm)
prefill/decode options (skip_intermediate_logits_requested=True, kv_bits=off, kv_start=n/a)
generation stop policy (ignore_eos=True, eos_ids=[248046])
[mtp] verifier_argmax_hook=active mode=logits
context=1024 TTFT=2057.5 ms TPOT=25.54 ms PP_TPS=497.7 TG_TPS=39.47 generated=128/128 first_token=2958 first_token_is_eos=False E2E=5310.7 ms statistic=median
  samples=2057.5/25.54/497.7/39.47/5310.7 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=15.185 GiB peak=16.952 GiB
  mtp rounds=43 drafted=85 accepted=85 acceptance=100.0% attention_routes=0 attention_fallbacks=0 gdn_routes=0 gdn_prefill_routes=0 gdn_prefill_fallbacks=0
context=4096 TTFT=8418.7 ms TPOT=26.39 ms PP_TPS=486.5 TG_TPS=38.19 generated=128/128 first_token=1817 first_token_is_eos=False E2E=11780.6 ms statistic=median
  samples=8418.7/26.39/486.5/38.19/11780.6 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=15.196 GiB peak=17.133 GiB
  mtp rounds=43 drafted=85 accepted=85 acceptance=100.0% attention_routes=0 attention_fallbacks=0 gdn_routes=0 gdn_prefill_routes=0 gdn_prefill_fallbacks=0
context=8192 TTFT=17942.5 ms TPOT=28.10 ms PP_TPS=456.6 TG_TPS=35.87 generated=128/128 first_token=14116 first_token_is_eos=False E2E=21521.2 ms statistic=median
  samples=17942.5/28.10/456.6/35.87/21521.2 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=15.215 GiB peak=17.410 GiB
  mtp rounds=43 drafted=85 accepted=85 acceptance=100.0% attention_routes=0 attention_fallbacks=0 gdn_routes=0 gdn_prefill_routes=0 gdn_prefill_fallbacks=0
context=16384 TTFT=42125.0 ms TPOT=32.35 ms PP_TPS=388.9 TG_TPS=31.15 generated=128/128 first_token=1590 first_token_is_eos=False E2E=46243.9 ms statistic=median
  samples=42125.0/32.35/388.9/31.15/46243.9 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
  memory active=15.246 GiB peak=17.934 GiB
  mtp rounds=43 drafted=85 accepted=85 acceptance=100.0% attention_routes=0 attention_fallbacks=0 gdn_routes=0 gdn_prefill_routes=0 gdn_prefill_fallbacks=0
suite_peak_memory_gib=17.934

```
