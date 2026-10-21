# MiniMLX 全类型模型支持架构与实施规划

## 目标

MiniMLX 的目标是覆盖 oMLX 所提供的模型任务与输入/输出模态，并保持模型家族、检查点格式、推理后端可独立扩展。支持范围包含文本生成、视觉语言/OCR、embedding、多模态 embedding、reranking、语音识别、语音生成、语音到语音，以及后续出现的新任务；视频作为输入模态进入支持契约。

### 当前近期目标与完成终点

近期目标是完成**可承载上述所有类型的通用框架**，并将 Qwen3.5 迁入该框架作为首个完整验证样本。终点为：全类型契约、注册、路由、协议和生命周期框架通过架构测试，Qwen3.5 通过新旧接口与正确性回归。此目标不包含把所有 VLM、embedding、reranker、audio 家族的实际 backend 都实现完；这些能力必须能在框架中被描述、注册、解析并给出准确支持状态，后续再逐项实现。

“全类型”是架构目标，不代表未经实现和验证的模型可以宣称已支持。每种任务/模态/模型组合必须有能力声明、适配器、验证用例和状态记录。

## 设计原则

1. **任务、模态、结构、加速、后端正交建模**：MoE 是结构，不是任务；图像是模态，不是执行引擎；MTP/dFlash 是生成加速能力，不是独立模型任务。
2. **发现与执行解耦**：发现器只报告检查点事实、文件组件和证据；解析器依据请求、检查点和运行环境生成执行计划；适配器执行计划。
3. **显式且可组合的适配器**：任务适配器定义输入输出语义；家族适配器处理模型结构差异；后端适配器封装 mlx-lm、mlx-vlm、mlx-audio 或 MiniMLX 原生实现；组件适配器处理视觉塔、投影器、声码器、draft 等组件。由注册表解析组合，不以大型 if/else 路由表承载全部逻辑。
4. **未知和冲突要显式失败**：识别证据不足、组件缺失、能力冲突、依赖不满足时，返回可诊断结果，不静默误路由或悄悄降级。
5. **稳定的公共协议**：Rust/Python IPC 和外部 API 使用与具体模型无关的有类型请求/结果；增量扩展字段保持向后兼容。
6. **功能正确性先于性能路径**：每个专用执行器有通用或安全 fallback；优化能力只有在正确性和门控条件通过后才启用。

## 领域模型

### 正交维度

| 维度 | 初始值示例 | 说明 |
|---|---|---|
| Task | generation, embedding, reranking, speech_to_text, text_to_speech, speech_to_speech | 一次请求要完成的操作 |
| Modality | text, image, audio, video | 输入和输出分别声明，不推断任务 |
| Architecture | Qwen3.5, Llama, Gemma, Whisper 等 | 模型结构/家族身份，可带多架构组成 |
| Structure | dense, MoE, hybrid_attention, encoder_decoder, diffusion | 对执行器有意义的结构属性 |
| Acceleration | MTP, dFlash, speculative, quantized_kv | 可选生成或缓存加速，带兼容性约束 |
| Artifact component | config, tokenizer, processor, text_weights, vision_encoder, projector, audio_encoder, decoder, vocoder, draft_weights, shard | 检查点必需/可选组件及来源文件 |
| Backend | mlx_lm, mlx_vlm, mlx_audio, minimlx_native, future | 实际加载/执行实现以及版本/硬件要求 |

### 核心实体

- `ModelDescriptor`：稳定身份、架构、配置摘要、量化、文件组件清单及发现证据；不承载执行器对象。
- `CapabilityProfile`：任务集合、输入/输出模态、结构属性、可选加速及其约束。
- `InferenceRequest`：task、带类型的消息/载荷、独立的 acceleration 集合、采样/长度选项、会话和流式选项。加速声明在 JSONL 中使用顶层 `accelerations` 字段；旧版本放在 `options.accelerations` 的请求只作为兼容输入读取，新客户端不得继续把路由语义塞入通用选项。
- `ExecutionPlan`：解析出的 task adapter、family adapter、backend、组件绑定、加速到运行时组件 provider 的映射、执行阶段、cache/scheduler 策略和 fallback 链。
- `InferenceResult`：按任务区分的结果（生成文本、向量、相关性分数、转写、音频块等）以及统一 usage/时延/诊断元数据。
- `SupportDecision`：supported、unsupported、blocked_by_dependency、missing_component、ambiguous 等结构化结果，附证据和可行动解释。

## 分层架构

```text
External APIs / GPUI / CLI
            |
        Protocol DTOs
            |
 Request Validation + Task Contract
            |
       Plan Resolver <---- Runtime Environment / Backend Catalog
            |
       Execution Plan
            |
  Task Adapter × Family Adapter × Backend Adapter
            |
    Component Loaders / Processor Registry
            |
       MLX / Native Kernels
```

### 1. Discovery

扫描本地检查点，解析 `config.json`、index/shard、tokenizer/processor 元数据和已知 sidecar。输出 facts 与 provenance（文件路径、字段/规则、置信等级）；不加载完整权重，不运行模型，不直接选择后端。

目录名和架构名仅作为证据之一。空的视觉配置、同架构文本量化模型、多用途模型、独立投影器、MTP sidecar 必须由组合规则处理。未知字段保留，不丢弃。

### 2. Capability Resolver

输入为 `InferenceRequest + ModelDescriptor + RuntimeEnvironment`。先验证任务/模态和必需组件，再选择可用 adapter/backend 组合。一个任务可有多个 backend candidate，通过显式优先级、依赖版本约束、硬件约束和 capability predicate 选出；决策结果可解释、可测试。请求可选声明 MTP/dFlash 等 acceleration；只有 provider 声明支持且运行环境具备对应 drafter/component 时才进入执行计划，否则分别返回 unsupported 或 dependency-blocked。

### 3. Adapter contracts

- `TaskAdapter`: 请求预处理、执行调用、结果转换及 task-specific validation。
- `FamilyAdapter`: 家族配置解释、权重映射、cache/state 约定和特殊 forward 语义。
- `BackendAdapter`: 后端可用性、版本/硬件条件、load/unload 与执行接口。
- `ComponentAdapter`: processor、vision/audio encoder、projector、vocoder、drafter 等组件发现、装载、生命周期和兼容性检查。
- `KernelProvider`: 可选算子实现与 shape/dtype/device 条件；不负责模型分类。

接口可以由单个适配器同时实现多个角色，但角色契约保持分离，避免形成无法替换的 giant adapter。

### 4. Task payloads

输入统一用有类型的内容 part 表达：text、image reference/bytes、audio reference/bytes、video reference/frames/timestamps、task-specific structured data。`structured` 是传输 payload kind（不是物理模态），用于 query/documents 等任务专属对象；视频帧载荷使用有时间戳的帧列表或视频引用。传输层不把 PIL、MLX array 等后端对象泄露给 API/Rust。视觉/音频 preprocessing 在后端侧执行，产生内部 `PreparedInput`。

### 5. Loading and lifecycle

模型基础权重、独立模态组件和辅助 drafter 分开描述、验证与加载。辅助加速工件必须在解析阶段按其模型身份、必需组件和声明的配对元数据（例如 hidden size、词表大小）与目标模型核验；目录存在或权重完整本身不代表可配对。

`ExecutionPlan` 是加载与执行的唯一决策来源：resolver 把已通过检查的加速工件路径及其 component provider 写入计划，family/backend adapter 只消费该计划，不得在 load 时重新读取环境变量并私自选择另一条路径。`ModelInstance` 登记基础模型与独立 processor/drafter 等资源，按依赖逆序释放；共享所有权的组件显式由主适配器释放。实例固定保存加载时使用的运行环境快照，使后续 request resolution 与已加载的 AR/MTP/backend 配置一致。模型切换时重新探测并解析；失败时保留原有实例。

当加速由环境自动启用而初始化失败时，runtime 必须移除该加速后重新解析并尝试安全的基础 backend；重新解析后的环境快照与实例计划都不得继续宣称该加速可用。显式请求的加速不得静默降级，必须返回带原因的失败。

`ModelInstance` 管理组件依赖、内存估算、warmup、unload；失败时释放已加载组件并保留可定位的错误。多模型池和内存调度建立在单模型生命周期契约之上。

### 6. Scheduling and cache

请求调度、连续 batching、单请求 decode、prefix/KV/recurrent/vision feature cache 按计划和任务能力选择。跨任务 cache 必须显式声明键、生命周期和安全边界。调度器不负责识别模型家族。

## Registry / plugin contract

每个 provider 注册 `AdapterRegistration`：名称、角色、支持 task/modality/architecture/backend、可选 acceleration、组件/依赖/硬件约束和优先级。注册匹配即纯配置 `probe`；`CapabilityResolver` 执行 validate 并产出 `ExecutionPlan`；family loader 在 plan 通过后才加载权重。内置 provider 和未来外部 plugin 使用同一契约，外部插件明确配置启用，不默认执行任意远程模型代码。

## 测试与支持状态

支持矩阵的键为 `(task, input modalities, output modalities, family, format, backend, hardware profile)`，而不是单纯模型名称。状态采用：

`discovered → classified → loadable → smoke_pass → task_correct → context_or_shape_pass → benchmarked`

另有 `unsupported`、`ambiguous`、`missing_component`、`dependency_blocked`、`hardware_blocked`。每个状态附测试版本、依赖锁、硬件和日志摘要。

测试层级：

1. descriptor/config/file manifest 单测和损坏/冲突样本；
2. resolver 矩阵测试（请求×能力×依赖×组件）；
3. adapter contract tests（输入、结果、错误、资源释放）；
4. 每个任务的 golden smoke/correctness 测试；
5. 请求协议兼容性和 round-trip 测试；
6. 经 task correctness 后才跑性能和内存基准。

## 分阶段实施

执行顺序严格分两大段：**先完成全类型架构设计与基础契约，再以 Qwen3.5 为首个垂直样本迁移验证并作为当前里程碑终点。** 架构阶段就定义全部任务/模态，不先实现图像或音频功能。Qwen3.5 之后的模型家族和任务 backend 扩充列为后续路线，不是当前里程碑的退出条件。

### 阶段 A — 全类型架构与基础设施（已完成）

#### A1. 领域与接口决策

- 固定 task/modality/structure/acceleration/component 的正交模型。
- 固定 Discovery、Capability Resolver、Task/Family/Backend/Component Adapter、Execution Plan、ModelInstance 生命周期边界。
- 固定统一有类型请求/结果、错误分类、依赖/硬件约束、状态机和 API/IPC 演进规则。
- 对照 oMLX、LM Studio、vLLM、llama.cpp 的做法，记录选择理由、反例和 MiniMLX 决策，不把某个项目的实现照搬为约束。
- 对照和设计取舍记录：[`Architecture_Decisions.md`](Architecture_Decisions.md)。
- 适配器开发模板见 [`Adapter_Development_Template.md`](Adapter_Development_Template.md)，包含注册、factory/lifecycle、验证矩阵和支持状态记录合同。
- 能力状态记录的机器可读 JSON Schema：[`model_support_record.schema.json`](model_support_record.schema.json)。
- 验收：完整覆盖所有目标任务、模态和检查点组件；主要失败模式都有表述；接口关系图和决策记录完成。

#### A2. 基础契约落地

- 落地通用 descriptor、artifact manifest、capability profile、typed request/result、support decision、adapter spec 的语言级类型。
- registry/provider 具备 `probe → validate → plan → load` 生命周期；纯配置路径可运行，不要求 GPU。
- 为全 task × modality × component × backend 组合建立 resolver 表驱动测试；未来尚未实现的条目可明确返回 unsupported/missing dependency，而非缺失类型。
- 验收：契约/协议不依赖 Qwen35 特例；现有 Python、Rust 边界兼容策略明确；静态矩阵测试通过。

#### A3. 架构评审与基线冻结

- 更新旧 runtime architecture、模型支持计划和支持矩阵入口，消除“文本先行即架构”的歧义；历史测量保留但标记历史。
- 写 adapter 开发模板、最小测试合同、支持状态 JSON Schema 和版本兼容策略。
- 验收：一份权威架构文档、一份实施路线、一份能力矩阵 schema；通过契约评审后进入 Qwen3.5 迁移。

### 阶段 B — Qwen3.5 首个垂直迁移（实现已接入，设备端正确性验收待完成）

- 把已验证的 Qwen3.5 dense/MoE、Q4/BF16、hybrid attention、MTP 和自研 kernels 接入 A 阶段契约；不重写已正确的热路径，只替换边界和路由。
- Qwen3.5 是“架构端到端验证样本”，不是架构边界本身；任何仅适用于 Qwen3.5 的字段/接口都不得上升为通用概念。
- 完成模型发现、component manifest、generation task request、执行计划、cache 生命周期、fallback 和报告闭环。
- 验收：旧 correctness/MTP boundary 测试全通过；已测量 benchmark 可复现且行为无回归；新 resolver 给出可解释计划；Qwen3.5 的 text/image-capable checkpoint 可正确区分，并完成单图、多图、base64/data URI 和视觉特征缓存验证。

### 后续路线（当前里程碑之后）

以下路线在架构中预留清晰 extension points，但不阻止当前“框架 + Qwen3.5”里程碑完成。

#### C — 扩全文本生成家族

- 先覆盖标准 dense 家族，再覆盖 MoE、hybrid/encoder-decoder 和各家族特殊 cache/采样行为。
- 每个家族优先复用 generation task 和 backend contract，仅新增必要 family adapter；未知架构进入明确的 unsupported/blocked 状态。
- 验收：oMLX 文本生成支持清单逐项有本地矩阵状态；每项经过 smoke 和 family correctness，不以成功 load 作为支持。

#### D — 视觉语言、OCR、图像/视频输入

- 实现视觉/视频 typed payload、processor、视觉 encoder/projector、embedding 注入、OCR family prompt/解码适配和视觉缓存。
- 验收：至少覆盖单图、多图、文本-only边界、sidecar 组合、OCR 固定样本和适配器 fallback；扩展至目标 VLM 家族矩阵。

#### E — Embedding、reranker、多模态检索

- 独立实现 pooling/归一化、cross-encoder、late-interaction 和图文输入任务合同。
- 验收：向量/打分输出形状和语义测试、检索排序 golden tests、API 兼容测试通过。

#### F — Audio STT/TTS/STS 与实时流

- 实现音频 payload、processor、encoder/decoder/vocoder、分块状态和实时取消/结束语义。
- 验收：转写/合成质量基线、音频可解码性、流式顺序/资源释放、后端选择和依赖报告通过。

#### G — Serving、调度、模型池与全覆盖

- 在统一任务契约上扩展 batching、streaming、model pool、内存 admission、unload、metrics 和任务间隔离。
- 建立定期 oMLX capability diff 与 backend/version/hardware 测试矩阵；新模型/任务以 provider 扩展，不修改核心调用者。
- 验收：目标 oMLX 支持条目全部被 MiniMLX 的明确状态覆盖；无未分类项；每项 supported 均附 correctness 证据。

## 当前仓库状态

当前实现已包含模型描述与 provenance、artifact manifest、能力解析、typed inference
contract、执行计划、Python/Rust JSONL DTO、adapter factory 和 unload 生命周期。
Qwen3.5 是首个注册路由，常规文本生成与 MTP 通过计划派发；新的统一入口与旧
`generate()` 保持兼容。架构单测及 Rust 协议测试通过。阶段 B 的设备端验收已完成：
Qwen3.5 4B Q4 的 128-token MTP 输出与同目标 AR 完全一致并显示
`compatibility=PASS`；`execution_route` 为 generation 且 `accelerations` 明确包含
`mtp`。强制拒绝后继续 5 轮及强制 EOS 边界均通过并与同目标 AR 对齐。
其他模型家族及
VLM/OCR、embedding/reranker、音频 backend 作为后续路线启动。

## 参考实现

- oMLX model discovery / task-engine categories: <https://github.com/jundot/omlx/blob/main/omlx/model_discovery.py>
- oMLX VLM preprocessing and embedding injection: <https://github.com/jundot/omlx/blob/main/omlx/engine/vlm.py>
- LM Studio MLX engine: <https://github.com/lmstudio-ai/mlx-engine>
- vLLM model/plugin registry contracts: <https://docs.vllm.ai/en/latest/contributing/model/registration/>
- llama.cpp multimodal artifact layout: <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md>
