# MiniMLX 性能测试架构

性能测试分成四层，任何优化必须先在对应层次证明，再运行全局测试：

```text
contract  →  micro-op  →  stage  →  end-to-end
正确性       单个算子       模型阶段      完整 decode
```

## 层次定义

### contract

验证输出 token、dtype、shape、cache/state 和误差容忍度。性能结果无效时，
首先检查 contract 是否通过。

### micro-op

只测一个不可再拆的操作，例如 quantized matmul、RMSNorm、RoPE 或自定义
Metal kernel。输入 shape、dtype、stride 和同步边界必须写入结果。

### stage

测模型结构中的完整阶段，例如 Qwen3.5 的：

```text
linear_attention
full_attention
MLP
final_norm + lm_head
```

stage 内部再记录每个子阶段，但不能把不同同步边界的数字直接相加。

### end-to-end

固定 prompt、生成 token 数、warmup、repeats 和 statistic，报告 prefill、
decode 和 tok/s。它只用于确认阶段优化最终是否反映到用户可见性能。

## 运行原则

- 默认 `warmup=3`、`repeats=5`、`median`；
- 一次只改变一个变量；
- benchmark 结果保存为 JSON，包含模型、runtime、MLX 版本和测试策略；
- 每个优化都必须有 baseline、candidate 和 correctness result；
- micro/stage 测试不自动触发 end-to-end，避免反复浪费完整模型运行时间；
- `decode_ms` 使用 cached decode 边界，不能混入 prefill；
- 所有 GPU 时间点必须明确 `mx.eval`/同步边界。

## 当前入口

```bash
PYTHONPATH=python .venv/bin/python benchmarks/run.py \
  --model /Users/jizhishi/.omlx/models/mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --suite e2e --prompt Hello --max-tokens 16 \
  --warmup 3 --repeats 5 --output results/latest.json
```

PyMiniMLX 的对照脚本仍然独立运行；它的数字可以作为 reference 输入，
但不会被 MiniMLX runtime 导入。

## oMLX 单请求对齐配置

RustRover 的默认 Run 配置对应：

```text
model      = Qwen3.5-0.8B-MLX-4bit
prompt     = Resources/BenchmarkCorpus/code_python.txt
context    = exactly 4096 tokenizer tokens
generation = 128 tokens
warmup     = Quick 32 tokens（一次 warm-up 请求，不是 32 次完整 benchmark）
```

截图中的 batch=2 和 batch=4 不属于单请求路径。只有 Python scheduler
实现真正的并发 batch 后，才会加入同一份报告。
