# MiniMLX

面向 Apple Silicon 的自有 MLX 推理 runtime。当前路线是 Rust/GPUI 负责
桌面界面、API 和进程管理，Python 负责 MLX 推理核心，Swift 只保留为可选
的 macOS 菜单栏辅助组件。推理热路径不再依赖 MLX Swift 或 `mlx-swift-lm`。

当前近期里程碑是全类型支持框架 + Qwen3.5 垂直验证。PyMiniMLX 只作为
独立参考工程，不会被安装、导入或打包进 MiniMLX：

```text
Discovery → CapabilityResolver → ExecutionPlan
                         ↓
TaskAdapter × FamilyAdapter × BackendAdapter → Runtime → JSON-lines Worker
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

在 macOS 上从仓库根目录运行 `cargo run -p minimlx-gpui` 可打开 GPUI 桌面应用。
目前可扫描本地模型、加载/卸载、从 Hugging Face/ModelScope 下载、查看实时系统
内存快照、运行 MiniMLX context-matrix 并保存报告，以及管理本机 OpenAI-compatible
API 服务。GPU 利用率暂不展示（无通用 macOS telemetry API）；性能测试只运行
MiniMLX，不启动或调用 oMLX。下载源的可选依赖与 API 路由见 [`RustRover.md`](RustRover.md)。

Qwen35Adapter 使用 MiniMLX 自己 `.venv` 中的 `mlx-lm` 作为常规基线，MTP
通过显式可选的 `mlx-vlm` drafter/verifier 路径接入。模型发现、能力解析、
执行计划和生命周期由通用框架管理；家族/后端实现留在 adapter 边界内，
而不是通过全局 monkey-patch 修改其他模型。

视觉语言请求使用独立的 `qwen35-vlm` provider：当检查点具备 processor 和
vision encoder，且环境安装 `mlx-vlm` 时，统一请求协议中的 `image` part 会
进入视觉处理和文本生成路径；支持本地 URI、data URI、base64 和多图请求，
并使用 Qwen3.5 原生视觉特征缓存。依赖或组件不完整时会在 resolver 阶段返回
结构化阻塞原因。

图片生成支持 `stream=true`：Python worker 和 Unix socket worker 返回 NDJSON
格式的 `delta` / `done` 事件；Rust supervisor 可通过 `request_stream` 收集完整
事件序列。

## 运行

```bash
PYTHONPATH=python ./.venv/bin/python -m minimlx_runtime --help
PYTHONPATH=python ./.venv/bin/python -m unittest discover -s Tests -p 'test*.py'
cargo test --workspace
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

全类型模型支持的权威架构和实施路线见
[`design/Universal_Model_Support_Architecture.md`](design/Universal_Model_Support_Architecture.md)。
新 task/family/backend/component provider 的开发合同见
[`design/Adapter_Development_Template.md`](design/Adapter_Development_Template.md)。
跨项目调研与架构取舍见
[`design/Architecture_Decisions.md`](design/Architecture_Decisions.md)。
早期 runtime 设计记录见 [`design/MiniMLX_Runtime_Architecture.md`](design/MiniMLX_Runtime_Architecture.md)。
