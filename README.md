# MiniMLX

面向 Apple Silicon 的自有 MLX 推理 runtime。当前路线是 Rust/GPUI 负责
桌面界面、API 和进程管理，Python 负责 MLX 推理核心，Swift 只保留为可选
的 macOS 菜单栏辅助组件。推理热路径不再依赖 MLX Swift 或 `mlx-swift-lm`。

第一阶段采用最小完整框架 + Qwen3.5-0.8B 垂直验证。PyMiniMLX 只作为
独立参考工程，不会被安装、导入或打包进 MiniMLX：

```text
ModelRegistry → Qwen35Adapter → Runtime → JSON-lines Worker
```

Rust workspace 位于 [`crates/`](crates/)：

```text
crates/
├── minimlx-protocol/    # Rust/Python JSON-lines 协议
├── minimlx-supervisor/  # Python worker 生命周期
├── minimlx-api/         # Rust API 边界
└── minimlx-gpui/        # GPUI 桌面入口
```

RustRover 运行说明见 [`RustRover.md`](RustRover.md)。

Qwen35Adapter 当前使用 MiniMLX 自己 `.venv` 中的 `mlx-lm` 作为基线；后续
会在同一 adapter 内加入我们的专用 Qwen3.5 执行路径、ExecutionPlan 和 Kernel Registry，
而不是通过全局 monkey-patch 修改其他模型。

## 运行

```bash
PYTHONPATH=python python -m minimlx_runtime --help
PYTHONPATH=python python -m pytest
```

固定 benchmark 输入位于 [`Benchmarks/cases.json`](Benchmarks/cases.json)，正式基线按设计文档执行 3 次预热、5 次正式运行并取 median。

## Qwen3.5 性能测试

需要 Apple Silicon 和 MiniMLX 自己的 Python 环境：

```bash
./tools/bootstrap_venv.sh

PYTHONPATH=python python -m minimlx_runtime generate \
  /path/to/Qwen3.5-0.8B-MLX-4bit "Hello" --max-tokens 16

PYTHONPATH=python python -m minimlx_runtime benchmark \
  /path/to/Qwen3.5-0.8B-MLX-4bit --prompt Hello \
  --max-tokens 16 --warmup 3 --repeats 5
```

模型目录应包含标准 Hugging Face `tokenizer.json`、`config.json` 和
`safetensors` 权重。性能对照脚本仍然可以单独使用 PyMiniMLX 的环境，
但那只属于 benchmark/reference，不属于 MiniMLX runtime。

完整架构见 [`design/MiniMLX_Runtime_Architecture.md`](design/MiniMLX_Runtime_Architecture.md)。
