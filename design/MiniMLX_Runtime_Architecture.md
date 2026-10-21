# MiniMLX Runtime Architecture

> 早期 runtime 架构记录。当前权威的全任务/全模态分层和实施路线见
> [`Universal_Model_Support_Architecture.md`](Universal_Model_Support_Architecture.md)。

## 目标

MiniMLX 的目标不是复刻 oMLX 的通用服务，而是构建一个面向 Apple Silicon 的高性能本地推理 runtime：

- GPUI/Rust 负责桌面界面、API 和进程管理；
- Python 负责 MLX 推理核心；
- Swift 只作为 macOS 菜单栏 `NSStatusItem` 的轻量辅助组件；
- PyMiniMLX 作为正确性参考和性能基线；
- 不在推理热路径中使用 Swift MLX，也不引入容器。

## 对 oMLX 的研究结论

oMLX 当前采用的是：

```text
Swift macOS app
    -> FastAPI / OpenAI API
    -> EnginePool
    -> BatchedEngine / Scheduler
    -> mlx-lm 或 mlx-vlm
    -> paged KV cache / SSD cache / custom kernels
```

这是一个合理的通用服务架构。它的模型支持主要由四部分组成：

1. 使用 `mlx-lm`、`mlx-vlm` 等上游 loader 和模型实现；
2. 根据 `model_type`、checkpoint 配置和模型能力选择加载路径；
3. 在加载前对上游模块进行模型族补丁、注册或兼容性修复；
4. 在运行时通过模型设置启用 custom kernel、MTP、prefill 优化或缓存策略。

这种方式覆盖面很强，但代价是补丁与上游版本紧密耦合。oMLX 的公开代码中可以看到，它会注册缺失的 `mlx_lm` 模型模块、替换已有模块中的类、包装模型的 `__call__` 和 `sanitize`，并为不同模型族维护大量独立 patch。Qwen3.5 的 MTP patch 还曾出现替换 `__call__` 后破坏 pipeline-parallel 路径的问题。这说明 monkey-patch 适合快速兼容，但不应成为 MiniMLX 的核心扩展机制。

## MiniMLX 的分层设计

```text
GPUI Desktop / Rust API Server
            |
            | Unix Domain Socket
            v
Python Runtime Service
            |
            +-- Request Manager
            +-- Scheduler
            +-- Execution Plan
            +-- Model Registry
            +-- Cache Manager
            +-- Kernel Registry
            v
Model Adapter
            |
            +-- Generic MLX-LM Adapter
            +-- Qwen3.5 Adapter
            +-- Future model-family adapters
            v
MLX Python / custom Metal kernels
```

### 1. Model Registry

只负责回答“这个 checkpoint 应该由谁加载”，不负责执行优化：

```python
ModelDescriptor(
    model_type="qwen3_5",
    architecture="hybrid_linear_full_attention",
    tokenizer_family="qwen",
    supported_modes={"prefill", "decode", "batch"},
)
```

识别顺序：

1. 读取 `config.json`；
2. 读取 quantization、layer types、MTP 和 multimodal 元数据；
3. 选择显式注册的 ModelAdapter；
4. 没有专用 adapter 时回退到 `mlx-lm`；
5. 对未知或不安全的 remote code 默认拒绝，除非用户明确允许。

### 2. Model Adapter

每个模型族由显式 adapter 提供，而不是修改 `mlx_lm` 的全局类：

```python
class ModelAdapter(Protocol):
    def can_load(self, descriptor) -> bool: ...
    def load(self, path, options) -> ModelInstance: ...
    def create_prefill_plan(self, request) -> ExecutionPlan: ...
    def create_decode_plan(self, request) -> ExecutionPlan: ...
    def capabilities(self) -> Capabilities: ...
```

每个 adapter 内部可以有三种路径：

```text
generic      -> mlx-lm 原生实现
optimized    -> 本项目显式实现的模型执行路径
fallback     -> 兼容性较低或不支持优化时的安全路径
```

这样 `Qwen3.5` 的特殊逻辑只存在于 `Qwen35Adapter`，不会污染其他模型。

### 3. Execution Plan

模型加载和执行策略分离。Plan 根据阶段、batch、context、dtype 和硬件能力选择路径：

```python
ExecutionPlan(
    phase="decode",
    batch_size=1,
    context_length=2048,
    use_custom_kernels=True,
    use_continuous_batching=False,
    use_speculative_decoding=False,
)
```

第一版优先支持：

- single-request decode；
- single-request prefill；
- recurrent state 和 KV cache；
- Qwen3.5 hybrid layer；
- custom kernel 的正确性和性能 gating。

之后再加入：

- continuous batching；
- prefix cache；
- speculative decoding；
- 多模型 resident/eviction。

### 4. Kernel Registry

custom kernel 不通过隐式全局 patch 启用，而是显式注册和选择：

```python
KernelCandidate(
    op="quantized_linear",
    phase="decode",
    shape_predicate=...,
    dtype_predicate=...,
    implementation="qmv_q4_decode_v2",
    correctness_tolerance=...,
)
```

选择逻辑必须同时满足：

1. shape 和 dtype 匹配；
2. 当前阶段匹配；
3. correctness probe 通过；
4. benchmark 证明比 generic 路径更快；
5. 失败时自动回退，不影响模型正确性。

这使 `Qwen3.5` 的 qkv/z/beta/a/output projection 优化可以独立演进，而不改变通用模型执行器。

## Qwen3.5 的第一专用路径

Qwen3.5 是当前最适合验证架构的模型，因为它同时包含：

- GatedDeltaNet linear-attention layer；
- full-attention layer；
- recurrent state；
- KV cache；
- 量化 projection；
- 后续可接入 MTP/speculative decoding。

它的 adapter 应明确暴露两个 layer executor：

```python
LinearAttentionExecutor
FullAttentionExecutor
```

而不是在一个通用 `__call__` 里隐式判断所有情况。

## oMLX 能力的取舍

### 直接借鉴

- 模型族注册和能力声明；
- generic loader + specialized path 的组合；
- custom kernel 的可选启用；
- 模型级配置和 feature gating；
- continuous batching 的调度思想；
- 后续的 prefix/KV cache 分层设计。

### 不直接复制

- 不通过全局 monkey-patch 改写 `mlx-lm` 类；
- 不让模型补丁同时改变 sanitize、forward、pipeline 和 MTP 状态；
- 不把服务调度、模型兼容、kernel 选择放进一个 loader 文件；
- 不默认让 single-user decode 经过通用 batching 调度；
- 不把所有模型的特殊逻辑都放在统一 giant engine 中。

## 推荐的项目结构

```text
minimlx/
├── crates/
│   ├── minimlx-gpui/          # GPUI 主界面
│   ├── minimlx-api/           # Rust API server
│   ├── minimlx-protocol/      # Rust/Python IPC 协议
│   ├── minimlx-supervisor/    # Python worker 生命周期
│   └── minimlx-menubar/       # 可选 Swift NSStatusItem helper
├── python/
│   └── minimlx_runtime/
│       ├── api/               # worker 内部协议
│       ├── runtime/           # request、scheduler、decode
│       ├── registry/           # model descriptors 和 adapters
│       ├── adapters/           # qwen35、generic mlx-lm 等
│       ├── plans/              # prefill/decode/batch plans
│       ├── kernels/            # kernel registry 和 implementations
│       └── cache/              # KV/recurrent/prefix cache
├── tools/
├── benchmarks/
└── design/
```

## 实施顺序

1. 在 MiniMLX 自己的 Python package 中实现 `Qwen35Adapter`，不导入 PyMiniMLX。
2. 先以 MiniMLX `.venv` 中的 `mlx-lm` 作为 baseline adapter。
3. 固定 prefill/decode correctness contract。
4. 建立 kernel registry，只迁移已经验证的 custom QMV。
5. 建立 single-request decode plan。
6. 用 PyMiniMLX、`mlx-lm`、oMLX 做三方基准。
7. 再加入 Rust API Server 和 GPUI。
8. 最后加入 batching、cache tiering 和 speculative decoding。

核心原则是：

> `mlx-lm` 负责覆盖面，Qwen35Adapter 负责性能，Kernel Registry 负责可验证的局部替换，Runtime Plan 负责在不同请求条件下选择路径。
