# MiniMLX Context Matrix

- Generated (UTC): 2026-10-07T04:45:51Z
- Model: `/Users/jizhishi/.omlx/models/mlx-community/Qwen3.5-0.8B-MLX-4bit`
- Model type: `qwen3_5`
- Architecture: `Qwen3_5ForConditionalGeneration`
- Quantization metadata: `{"group_size":64,"bits":4,"mode":"affine"}`
- Model disk size: 622 MB
- Hardware: Apple M5 Pro
- Unified memory: 64.0 GB
- macOS: 27.0.1
- Python packages: mlx=0.32.2, mlx-lm=0.31.3
- Source revision: dbad46b (modified working tree)
- Benchmark corpus: `/Users/jizhishi/mycode/ai/minimlx/Resources/BenchmarkCorpus/code_python.txt`
- Corpus SHA-256: `59f6ec70e0b687ec49463dda575322348d7efe0d0770e69ff37b0e3ec6907aa1`
- Warm-up: 32 tokens at 2K context
- Measurement: 3 runs per context; median and raw samples reported
- Generation limit: 128 tokens
- Contexts: 1K, 4K, 8K, 16K
- Runtime: MiniMLX only

## Raw benchmark output

```text
MiniMLX context matrix (warmup=32 tokens at context=2048, measured_runs=3, step=2048)
runtime kernels (q4_native=True, fa256_native=True, q4_dense_disabled=False, q4_mlp_fusion_disabled=True)
decode backend (autoregressive)
context=1024 TTFT=96.8 ms TPOT=3.01 ms PP_TPS=10579.9 TG_TPS=334.42 generated=128/128 E2E=756.5 ms statistic=median
  samples=96.8/3.01/10579.9/334.42/770.1; 97.6/3.02/10492.3/334.26/756.5; 94.1/3.01/10885.9/334.84/750.9 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
context=4096 TTFT=313.8 ms TPOT=3.20 ms PP_TPS=13051.6 TG_TPS=315.24 generated=128/128 E2E=998.9 ms statistic=median
  samples=313.8/3.16/13051.6/319.23/990.1; 313.5/3.20/13065.9/315.24/998.9; 313.9/3.20/13048.6/315.06/999.8 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
context=8192 TTFT=650.7 ms TPOT=3.40 ms PP_TPS=12589.8 TG_TPS=296.53 generated=128/128 E2E=1361.8 ms statistic=median
  samples=647.5/3.35/12652.3/300.86/1350.4; 650.7/3.42/12589.8/294.37/1361.8; 662.7/3.40/12361.5/296.53/1372.9 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
context=16384 TTFT=1429.8 ms TPOT=3.79 ms PP_TPS=11458.8 TG_TPS=266.16 generated=128/128 E2E=2189.9 ms statistic=median
  samples=1446.8/3.79/11324.0/266.16/2208.6; 1429.8/3.62/11458.8/278.55/2173.4; 1421.8/3.84/11523.4/262.46/2189.9 (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)
suite_peak_memory_gib=2.160

```
