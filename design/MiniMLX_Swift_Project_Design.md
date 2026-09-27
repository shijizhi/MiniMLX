# MiniMLX Swift 项目设计文档

## 1. 文档目的

本文档定义一个独立的 Swift 项目 `minimlx`。

它和当前的 Python 项目 `pyminimlx` 具有不同职责：

```text
pyminimlx
    学习、解释、验证、参考实现

minimlx
    Swift 原生实现、性能优化、对标 oMLX
```

两个项目使用相同的模型、tokenizer、输入样例和测试协议，但不共享实现细节。这样既能保留 Python 项目的教学价值，又能验证 Swift/MLX 在 Apple Silicon 上是否具备对标 oMLX 的性能和工程基础。

## 2. 项目定位

MiniMLX 不是 Python 项目的 Swift 翻译版，也不是立即重写所有底层算子的项目。

它的定位是：

> 一个面向 Apple Silicon 的、可验证、可测量、逐步原生化的 LLM inference runtime。

第一目标不是功能最多，而是建立一个可以持续对标 oMLX 的实现：

```text
同一模型
同一权重
同一 tokenizer
同一 prompt
同一生成长度
同一采样策略
        ↓
Python PyMiniMLX / Swift MiniMLX / oMLX
        ↓
正确性 + 延迟 + 吞吐 + 内存 + 能耗
```

## 3. 项目边界

### 3.1 第一阶段必须完成

- 加载 `Qwen3.5-0.8B-MLX-4bit` 模型目录。
- 使用标准 tokenizer，而不是 Python 项目早期的教学 tokenizer。
- 实现文本编码和解码。
- 实现 prefill 和逐 token decode。
- 实现 Qwen3.5 所需的混合层：Gated DeltaNet、Full Attention、RoPE、Gated MLP、RMSNorm、LM Head。
- 支持 recurrent state、KV Cache 和 position 管理。
- 使用 greedy decoding，保证结果可重复。
- 输出和 Python 版、oMLX 的对标结果。
- 建立固定 benchmark 基线。

### 3.2 第一阶段不做

- 不先做复杂的聊天 UI。
- 不先做多用户服务端。
- 不先接 ANE 专用后端。
- 不为了跨平台而抽象出 CUDA、Vulkan 等后端。
- 不在没有 benchmark 证据时重写所有算子。

## 4. 技术选择

### 4.1 Swift + MLX Swift

MiniMLX 使用官方 MLX Swift 作为底层 tensor/Metal substrate，但不依赖
`mlx-swift-lm` 的模型和生成 runtime：

- Swift 作为模型运行时和调度语言。
- MLX Swift 负责 tensor、自动求值、Metal GPU 执行和统一内存。
- MiniMLX 自己负责模型加载、请求状态、prefill、decode 和采样。
- 必要时通过 Metal kernel 做定制优化。
- 后续再评估 Core ML、MPSGraph 和 ANE。

这样可以优先验证“原生 Swift 调度是否比 Python 调度更适合我们的推理路径”，而不是同时引入一套完全不同的张量系统。

### 4.2 为什么不是直接重写 Metal

直接 Metal 会同时改变模型表达、内存管理、量化格式和 kernel 调度，难以判断性能变化来自哪里。

MiniMLX 采用分层策略：

```text
Swift + MLX Swift
        ↓
确认模型正确性和 Swift 调度基线
        ↓
替换单个热点为 Metal kernel
        ↓
比较替换前后收益
        ↓
必要时再下沉更多算子
```

## 5. 总体架构

```text
┌──────────────────────────────────────────────────────────────┐
│                       MiniMLX Swift                           │
├──────────────────────────────────────────────────────────────┤
│  App / CLI                                                     │
│  main.swift · Commands · Output                                │
├──────────────────────────────────────────────────────────────┤
│  Runtime                                                       │
│  NativeQwen35Runtime · NativeQwen35Request · Sampler           │
├──────────────────────────────────────────────────────────────┤
│  Model                                                         │
│  MiniMLXQwen35Model · DecoderLayer · LinearAttention · FullAttention │
│  GatedMLP · RMSNorm · RoPE · LMHead                            │
├──────────────────────────────────────────────────────────────┤
│  Weights / Tokenizer                                           │
│  ModelConfig · SafeTensorsLoader · QuantizedWeight             │
│  Tokenizer · ChatTemplate                                      │
├──────────────────────────────────────────────────────────────┤
│  Backend                                                       │
│  MLX Swift substrate · MLX Fast Ops · Custom Metal Kernels      │
├──────────────────────────────────────────────────────────────┤
│  Platform                                                      │
│  Apple Silicon · Unified Memory · Metal GPU · Instruments       │
└──────────────────────────────────────────────────────────────┘
```

## 6. 推荐文件结构

```text
minimlx/
├── Package.swift
├── README.md
├── Sources/
│   ├── MiniMLX/
│   │   ├── MiniMLX.swift
│   │   ├── CLI/
│   │   │   ├── Command.swift
│   │   │   ├── RunCommand.swift
│   │   │   └── BenchmarkCommand.swift
│   │   ├── Runtime/
│   │   │   ├── NativeQwen35Runtime.swift
│   │   │   ├── NativeQwen35Request.swift
│   │   │   ├── MLXExecutionState.swift
│   │   │   ├── GenerationConfig.swift
│   │   │   └── Sampler.swift
│   │   ├── Models/
│   │   │   ├── Model.swift
│   │   │   ├── Qwen35NativeModel.swift
│   │   │   ├── Qwen35Config.swift
│   │   │   ├── Qwen35DecoderLayer.swift
│   │   │   └── ModelFactory.swift
│   │   ├── NN/
│   │   │   ├── Linear.swift
│   │   │   ├── QuantizedLinear.swift
│   │   │   ├── RMSNorm.swift
│   │   │   ├── RoPE.swift
│   │   │   ├── GatedDeltaNet.swift
│   │   │   ├── GatedAttention.swift
│   │   │   ├── GatedMLP.swift
│   │   │   └── CausalAttention.swift
│   │   ├── Weights/
│   │   │   ├── SafeTensorsLoader.swift
│   │   │   ├── WeightMap.swift
│   │   │   └── Quantization.swift
│   │   ├── Tokenizer/
│   │   │   ├── Tokenizer.swift
│   │   │   ├── HFTokenizer.swift
│   │   │   ├── BPE.swift
│   │   │   └── ChatTemplate.swift
│   │   ├── Backend/
│   │   │   ├── MLXBackend.swift
│   │   │   ├── MetalKernelRegistry.swift
│   │   │   └── MetalKernels/
│   │   │       ├── GatedDeltaKernel.metal
│   │   │       ├── QuantizedLinearKernel.metal
│   │   │       └── LMHeadKernel.metal
│   │   └── Observability/
│   │       ├── Timer.swift
│   │       ├── Profiler.swift
│   │       ├── BenchmarkResult.swift
│   │       └── Trace.swift
│   └── minimlx/
│       └── main.swift
├── Tests/
│   ├── MiniMLXTests/
│   ├── TokenizerTests/
│   ├── ModelShapeTests/
│   ├── NumericalTests/
│   └── BenchmarkTests/
├── Benchmarks/
│   ├── cases.json
│   └── expected-results.json
└── Resources/
    └── README.md
```

`main.swift` 只负责启动命令，不承载模型逻辑。模型应通过 Swift API 创建：

```swift
let runtime = try await ModelFactory.loadRuntime(from: modelDirectory)
let result = try await runtime.generate(prompt: "Hello", maxTokens: 32)

// Streaming/interactive callers can retain an independent request:
let request = runtime.makeRequest()
try request.prefill(tokenIDs: [9419])
let token = try request.decodeNext(eosTokenID: runtime.model.config.eosTokenID)
```

## 7. 核心模块职责

### ModelFactory

根据模型目录中的配置创建模型实例。未来新增模型时，增加配置和 architecture adapter，不把所有模型逻辑塞进入口文件。

### SafeTensorsLoader

负责：

- 读取 safetensors header。
- 建立 tensor name 到文件位置的映射。
- 延迟或按需加载权重。
- 识别 BF16、FP16 和 packed 4-bit 权重。

### MiniMLXQwen35Model

负责完整的 Qwen3.5 前向结构：

```text
embedding
  → 24 decoder layers
  → final RMSNorm
  → LM Head
  → logits
```

### NativeQwen35Request

统一保存一次生成请求中的状态，并提供明确的 prefill/decode 边界：

- 当前 position。
- 每层 Gated DeltaNet recurrent state。
- 每个 Full Attention 层的 KV Cache。
- `prefill(tokenIDs:)` 保存首个 logits。
- `decodeNext(eosTokenID:)` 执行一个采样和 decode step。
- `decode(maxTokens:eosTokenID:)` 执行有限生成。

### NativeQwen35Runtime

持有模型、tokenizer 和 MLX backend，并创建相互隔离的
`NativeQwen35Request`。它负责文本 API、计时和 benchmark；请求对象负责
实际缓存和逐 token 状态推进。

生成控制流程：

```text
encode prompt
  → prefill
  → logits
  → sample next token
  → decode one token
  → update state
  → repeat
```

它不直接知道 Qwen3.5 的具体层结构。

### Observability

提供统一的性能数据：

- tokenizer 时间。
- model loading 时间。
- prefill 延迟。
- decode 总时间。
- tokens/second。
- 每层和每类算子耗时。
- 内存峰值。
- 可选 Metal trace 文件路径。

## 8. 推理时序

```text
User
 │
 │ prompt
 ▼
Tokenizer.encode()
 │ token_ids
 ▼
NativeQwen35Request.prefill()
 │
 ├─ embedding lookup
 ├─ layer 0 ... layer 23
 │   ├─ RMSNorm
 │   ├─ GatedDeltaNet 或 FullAttention
 │   ├─ residual
 │   ├─ RMSNorm
 │   ├─ GatedMLP
 │   └─ residual
 ├─ final RMSNorm
 └─ LM Head → logits
 │
 ▼
Sampler.select(logits)
 │ next_token
 ▼
NativeQwen35Request
 │
 ├─ Tokenizer.decode()
 ├─ request state 更新
 └─ decodeNext()
       │
       └─ 每轮只输入一个 token，重复上述层计算
```

## 9. 正确性策略

性能优化必须建立在逐层数值验证之上。

### 9.1 分层验证顺序

1. tokenizer 编码结果。
2. tokenizer 解码结果。
3. embedding 输出。
4. 单个 RMSNorm。
5. 单个 Linear。
6. 单个 GatedMLP。
7. 单个 Gated DeltaNet。
8. 单个 Full Attention。
9. 单个 DecoderLayer。
10. prefill logits。
11. 第一个 decode logits。
12. 完整 greedy token 序列。

### 9.2 参考实现与对标对象

项目中固定保留三类参照：

```text
Python PyMiniMLX  → 教学、结构和数值参考
oMLX              → 目标产品、性能和工程能力的对标对象
MiniMLX Swift     → 当前实现
```

oMLX 是外部对标运行时，不是 MiniMLX 的内部依赖。对标时记录完整请求参数、模型目录、tokenizer 版本、生成配置、吞吐、延迟、内存和稳定性。

## 10. Benchmark 设计

第一版固定以下测试集：

```json
[
  {"name": "short", "prompt": "Hello", "max_tokens": 16},
  {"name": "english", "prompt": "Write a short introduction about machine learning.", "max_tokens": 64},
  {"name": "mixed", "prompt": "请用英文解释什么是 GPU。", "max_tokens": 64}
]
```

每个 case：

- 预热 3 次。
- 正式运行 5 次。
- 使用 median，不使用单次最好成绩。
- 固定 greedy decoding。
- 分开统计 prefill 和 decode。
- oMLX 另外记录 HTTP end-to-end 延迟。

核心输出格式：

```text
MiniMLX Swift benchmark
  short:   prefill=... ms, decode=... tok/s
  english: prefill=... ms, decode=... tok/s
  mixed:   prefill=... ms, decode=... tok/s

Reference
  PyMiniMLX: ...
  oMLX:      ...
```

## 11. 性能路线

### Phase 1：Swift 正确性基线

- 使用 MLX Swift 原生算子。
- 完成模型加载和生成。
- 不写自定义 Metal kernel。
- 与 Python、oMLX 对齐输出。

### Phase 2：Swift 调度基线

- 复用权重和 pipeline 对象。
- 减少循环中的临时对象。
- 合并不必要的 `eval` 边界。
- 统一 state 和 cache 的生命周期。
- 对 prefill 和 decode 分别优化。

### Phase 3：Metal 热点优化

优先顺序：

1. Gated DeltaNet state update。
2. 4-bit Quantized Linear。
3. LM Head。
4. RMSNorm + projection 等小算子融合。
5. KV Cache 写入和布局转换。

每次只替换一个热点，并保留 fallback：

```text
Custom Metal Kernel
        ↓ 失败/不支持
MLX Swift reference path
```

### Phase 4：平台能力评估

- Xcode Metal capture。
- Instruments 时间线和内存分析。
- MPSGraph 实验。
- Core ML / Core AI 实验。
- ANE 可用性和端到端收益评估。

这一步不改变主线基线，所有平台实验都作为独立 backend 或实验分支存在。

## 12. 版本规划

| 版本 | 目标 |
|---|---|
| v0.1 | Swift Package、模型目录、MLX 初始化 |
| v0.2 | SafeTensors、配置和 tokenizer |
| v0.3 | Qwen3.5 单层数值验证 |
| v0.4 | 完整 prefill |
| v0.5 | decode、state、KV Cache |
| v0.6 | greedy 文本生成 |
| v0.7 | 与 Python/oMLX 自动对标 |
| v0.8 | Swift 性能基线 |
| v0.9 | Metal profiling 和第一个 custom kernel |
| v1.0 | 稳定的 MiniMLX 对标 runtime |

## 13. 成功标准

MiniMLX v1.0 达到以下条件才算完成第一阶段：

- 可以独立加载目标 Qwen3.5 4-bit 模型。
- 不依赖 Python 进程即可生成文本。
- tokenizer、首个 token 和固定 greedy 序列通过一致性检查。
- benchmark 可重复运行并产生 median 结果。
- 能区分 prefill、decode、kernel 和端到端时间。
- 任何自定义优化都有 MLX fallback。
- Python 版仍然可以单独运行和用于教学。
- oMLX 只作为外部对标对象，不被硬编码进 MiniMLX 的模型执行逻辑。

## 14. 与 Python 项目的协作规则

两个项目共享“实验协议”，不共享“实现代码”：

```text
共享：模型、tokenizer、prompt、测试数据、结果格式
不共享：模型类、运行时、缓存实现、Metal kernel、CLI 实现
```

Python 项目继续回答：

> 模型每一步为什么这样计算？

Swift 项目继续回答：

> 在 Apple Silicon 上怎样把这一步执行得更快？

最终形成：

```text
PyMiniMLX  = 可读的 Reference Runtime
MiniMLX    = 原生 Swift Performance Runtime
oMLX       = 外部生产级对标 Runtime
```
