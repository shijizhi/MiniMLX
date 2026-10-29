# MiniMLX 模型接入标准

本文是新增模型接入时的执行规范。它补充
[`Universal_Model_Support_Architecture.md`](Universal_Model_Support_Architecture.md)
和 [`Adapter_Development_Template.md`](Adapter_Development_Template.md)，用于
避免把“模型家族接入”“后端接入”和“性能优化”混成一件事。

## 一、先记住架构边界

MiniMLX 的模型支持不是“一个模型对应一个 Adapter 类”。一次请求的执行路径
由多个角色组合而成：

```text
ModelDescriptor
    ↓
CapabilityResolver
    ↓
ExecutionPlan
    ├── Task Adapter       请求语义、输入输出、结果转换
    ├── Family Adapter     模型结构和家族特有的权重/cache/forward 语义
    ├── Backend Adapter    mlx-lm、mlx-vlm、mlx-audio 或 MiniMLX 原生执行后端
    └── Component Adapter  processor、vision/audio encoder、projector、drafter 等
```

几个必须保持的区分：

| 概念 | 负责什么 | 不负责什么 |
|---|---|---|
| `mlx-lm` | 通用文本模型的加载和生成后端 | 不负责判断 MiniMLX 的任务能力和发布状态 |
| `mlx-vlm` | 通用视觉语言加载、预处理和生成后端 | 不代表每个 VLM 家族都已经完成验证 |
| Family Adapter | 只有模型结构确实需要时，处理家族差异 | 不应因为模型名字不同就自动新增 |
| `CapabilityResolver` | 根据请求、检查点、依赖和硬件选择计划 | 不加载权重，不执行推理 |
| `ExecutionPlan` | 固化本次加载和执行的所有选择 | 不应在 adapter 内重新猜测路由 |
| Kernel/optimization | 对已验证形状提供可选加速 | 不负责模型分类或基础兼容性 |

因此，`mlx-lm` 是通用基线能力，不等于“每个模型都创建一个模型专属的
`MLXLMAdapter`”。所有能由标准 MLX-LM 正确执行、又没有已验证专项优化的文本模型，
都应复用同一个基线 provider。性能优化层叠加在基线上，只替换或增强有证据的执行
环节；它不是另一条取代基线的模型路由。

## 二、当前仓库中的实际分工

### 1. Discovery

`describe_model()` 只读取检查点事实：

- `config.json` 和架构字段；
- tokenizer、processor、权重 shard 和 index；
- 量化、视觉、音频、视频和辅助模型元数据；
- 文件来源和发现证据。

Discovery 不加载完整模型，不导入 Metal，不根据目录名直接宣布支持。

### 2. Provider Registry 与 Resolver

`AdapterRegistration` 是声明式匹配合同，描述：

- provider 角色：task/family/backend/component；
- task 和输入输出模态；
- 模型架构证据；
- 必需组件；
- 后端依赖、版本和硬件条件；
- 可选加速和优先级。

`CapabilityResolver` 根据：

```text
InferenceRequest + ModelDescriptor + RuntimeEnvironment
```

生成 `SupportDecision + ExecutionPlan`。解析成功后，plan 是后续 load 和 execute
的唯一依据。

### 3. Runtime 与 Adapter

`Runtime` 负责模型切换、实例工厂、组件绑定、执行器派发和失败回滚。
它不应该把所有模型判断写成大型 `if/else`。

当前几个已有实现的代码位置与目标架构的关系：

- `StandardTextAdapter`（`adapters/mlx_lm_standard.py`）：MLX-LM 文本家族共用基线；Llama、Qwen2/3 等已列入通用家族目录的模型使用该基线，不因家族名称增加模型专属 adapter；
- `Qwen35Adapter`：Qwen3.5 的版本专用协调器。普通文本/VLM路径先加载 `StandardTextAdapter` 或 `MLXVLMAdapter` 基线，再按计划应用 `qwen35-performance-overlay`；自动启用的 overlay 初始化失败时 Runtime 重新解析并使用未优化基线，专项环境选项明确请求的优化失败则直接报告。`adapters/qwen35_overlay.py` 负责叠加/回退协调，`adapters/qwen35_optimization.py` 持有版本专用 hooks、视觉特征缓存、自定义生成和 profiling 算法；MTP/DFlash 仍按独立 acceleration 计划处理；
- `MLXVLMAdapter`：家族无关的 `mlx-vlm` 视觉语言基线，Qwen3.5 复用其加载、processor、消息处理和生命周期；性能 hooks 由独立 overlay 提供；
- Qwen3.8 当前检查点配置会被上游识别为 `qwen3_5`，但 `.minimlx-repo-id` 中的模型身份用于排除 Qwen3.5 专用 provider：文本走通用 MLX-LM，媒体输入走通用 MLX-VLM。这不意味着它属于 Qwen3.5 性能优化范围。Qwen3.8 专项优化必须按独立身份/版本登记；无单独测量与方法时只使用通用基线/通用 VLM 能力。

上述实现不能反过来改变架构原则：Qwen3.5 是垂直验证样本，不是所有模型的
路由模板。Qwen2.5 的既有验证记录描述的是通用基线上的模型兼容证据，不定义专属
Qwen2 优化层。

## 三、新模型接入的标准决策树

### 第一步：先判断是否已经有可复用的路线

按以下顺序检查：

1. 是否是已有 family adapter 的同一结构，只是尺寸、量化或 checkpoint 变体；
2. 是否可以走现有通用文本 task/backend provider；
3. 是否可以走现有 `MLXVLMAdapter`；
4. 是否只是缺少组件声明、架构别名或 provider registration；
5. 最后才判断是否需要新的 family adapter。

不能因为 `ModelRegistry` 当前没有返回某个模型，就直接创建一个以模型名命名的
新 adapter。先确认缺的是：

- provider registration；
- 通用 backend 的接线；
- descriptor 发现规则；
- 组件 manifest；
- 还是实际的结构适配。

### 第二步：只补最小缺口

#### 情况 A：同一家族的 checkpoint 变体

只补 discovery/config、量化或组件条件，并复用现有 family adapter。

#### 情况 B：新家族但标准后端可直接执行

复用现有 task/backend 路线，新增：

- 家族的架构证据和 provider registration；
- 必需组件和能力声明；
- resolver 测试；
- 实际模型 smoke/correctness 记录。

不要新增一个仅仅包了一层 `mlx-lm.load()` 的模型专属 adapter。

#### 情况 C：标准后端无法正确执行

只有在确认存在真实结构差异时，才新增 family adapter。必须说明：

- 上游标准路径在哪一步失败；
- 哪些权重、cache、forward 或 processor 语义不同；
- 新 adapter 如何复用 task/backend contract；
- 哪些场景仍然 fallback 到标准路径。

#### 情况 D：需要性能优化

先让标准基线通过 correctness，并记录基线性能。只有明确瓶颈、优化方法、目标指标、
适用版本/shape 和可重复 A/B 证据齐备后，才增加可选优化 provider。provider 必须
复用基线的公共请求、组件与生命周期契约，只覆盖特异化步骤；计划中同时保留基线和
优化身份及 fallback。基线仍是该模型的默认兼容路径，优化关闭或不适用时必须能独立
工作。优化代码按真实版本分开，例如 Qwen3.5 与 Qwen3.8 不共享版本专属 patch；
Llama 不得挂入 Qwen 优化层。

## 四、以 Llama 为例的正确接入流程

Llama 3.2 3B 是标准文本模型。正确顺序是：

1. Discovery 确认 `model_type=llama`、`LlamaForCausalLM`、tokenizer 和文本权重；
2. Resolver 查询现有通用文本 task/backend provider；
3. 如果只缺 Llama 的 registration，就补 registration 和 capability fixture；
4. 通过现有标准加载/生成契约执行，不创建 `MLXLMAdapter`；
5. Llama 先运行通用 MLX-LM 基线；性能优化只有在 profile 明确瓶颈、提出可验证方法并完成 A/B 后才单独叠加。只有标准后端存在正确性缺口时才设计 family adapter；
6. 将结果记录为 Llama 家族的独立支持状态，不把它错误归为 Qwen。

换句话说，Llama 不进入 Qwen 专用层，也不因为是新家族就需要自己的完整执行器。
基线负责广泛兼容，未来若有足够证据，再叠加 Llama 独立的性能优化。

## 五、每个新模型必须通过的测试层级

### A. 无设备架构测试

- descriptor/config 解析；
- 组件 manifest；
- family/provider 匹配；
- text-only 与 multimodal 请求路由；
- 缺组件、缺依赖、未知架构和冲突配置的失败状态；
- factory、unload 和失败加载回滚。

### B. 设备 smoke

- 实际加载模型；
- 固定 prompt 生成；
- 重复执行 token 或结果一致性；
- streaming；
- API/CLI 到 runtime 的完整调用链。

### C. 家族 correctness

- chat template；
- EOS/停止条件；
- 采样参数；
- 长输入截断和 context boundary；
- 视觉/音频/视频等该模型声明的特殊能力。

### D. 性能测试

只有 A～C 通过后才进行：

- 1K/4K/16K context matrix；
- 安全时再测 32K；
- TTFT、TPOT、prefill/decode throughput、E2E 和峰值内存；
- 固定硬件、依赖版本、warmup、重复次数和原始日志。

支持状态必须按以下顺序推进，不能把“能识别”直接写成“已支持”：

```text
discovered
 → classified
 → loadable
 → smoke_pass
 → task_correct
 → context_or_shape_pass
 → benchmarked
```

## 六、本次错误的复盘规则

本次错误是因为只看了当前几个具体 adapter 文件，没有先按架构文档检查：

1. `mlx-lm` 是后端，不是按模型复制的 adapter 类；
2. provider registry、resolver 和 runtime factory 才是接入链路；
3. 新模型未匹配时，要先定位 registration、descriptor、组件或 backend 接线缺口；
4. Qwen3.5 的专用路径不能被误认为通用路径缺失。

以后新增模型前，必须先完成本文件“第三步：只补最小缺口”的判断，并在变更说明
中写出“复用了哪个现有 provider、实际新增了什么结构差异”。

