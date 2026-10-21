# MiniMLX 模型覆盖与性能推进计划

> 本文是早期按 Qwen3.5 优先组织的执行计划。全类型目标、架构边界和当前实施顺序以
> [`Universal_Model_Support_Architecture.md`](Universal_Model_Support_Architecture.md) 为准；本文件中的历史基准和已完成工作记录继续保留。

## Scope clarification

本计划的目标是覆盖模型族、任务和输入模态，而不是只支持当前磁盘上的模型
目录。`~/.omlx/models` 或其他本地 checkpoint 仅作为可复现的 conformance
样本和性能证据；模型没有下载、磁盘空间不足或当前机器无法加载时，仍然可以
通过 config/schema/provider 进入 `discovered`、`planned`、`blocked` 或
`unsupported` 状态。不能把“本地没有模型”解释成“MiniMLX 不支持该模型”。

Qwen3.5 的家族级完成范围和逐项状态见
[`Qwen35_Family_Coverage.md`](Qwen35_Family_Coverage.md)。Qwen3.5 已完成首个
端到端支持起点，横向扩展现已启动：按 A2 顺序先覆盖其他 Qwen 架构，当前从
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` 开始。Qwen3.5 尚未完成的超大尺寸或
特殊格式仍留在台账中，但不再阻止横向扩展；未完成项继续明确记录状态。

基准结果统一记录在 [`Model_Benchmark_Comparison.md`](Model_Benchmark_Comparison.md)。

## 执行队列

以下是实际执行顺序。每完成一个步骤，就更新支持矩阵和状态；没有完成当前步骤的退出条件，不进入下一步。

### 阶段 1：Qwen3.5 能力目录和基线

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 1.1 | 根据模型族、任务、模态、格式和 backend 建立能力目录 | provider/capability matrix | 不依赖本地权重存在 |
| 1.2 | 对可获得的 checkpoint 运行 `verify_model_support.py` | conformance 证据 | 证据只更新状态，不定义支持范围 |
| 1.3 | 修复通用配置读取和 adapter 分发 | registry/adapter 变更 | 不依赖目录名；同类模型进入同一 provider |
| 1.4 | 建立失败样本和缺依赖回归测试 | 固定失败用例 | 同一模型得到稳定、可解释的失败分类 |

当前状态：0.8B Q4/BF16 与 4B Q4 兼容性均已通过。4B BF16 已通过 descriptor/load/4-token smoke、1K/4K/16K 短生成上下文检查，MLX 峰值内存 7.899 GiB；在 24 GB M4 上登记为 Supported，不开展性能调优。

### 阶段 2：Qwen3.5 Dense 变体

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 2.1 | 收集本机可用 Dense 变体 | 支持矩阵行 | 每种尺寸/精度/量化各一行 |
| 2.2 | 运行短 prompt smoke | 验证日志 | prefill、decode、token 输出均正确 |
| 2.3 | 运行 1K/4K/16K context matrix | 性能基线 | 记录 TTFT、TPOT、TG TPS、E2E、峰值内存 |
| 2.4 | 尝试 32K | 长上下文结果 | 通过则标记 context-pass；失败则记录内存原因 |
| 2.5 | 完成 MiniMLX 自有基准和报告 | 可复现结果表 | 保存原始重复结果、median/min/max、实际硬件和软件环境 |
| 2.6 | 处理差异与异常 | MiniMLX 性能决策记录 | 与本机历史基线相比出现稳定异常时再 profile |

阶段 2 的执行顺序以能力目录为准，而不是以本地下载队列为准。只有在需要
conformance 或性能证据时才获取一个具体 checkpoint；格式顺序为
BF16/FP16 → Q4/Q8 → 其他格式，不先写格式专用优化。

### 阶段 3：Qwen3.5 MTP 兼容性与 decode（correctness 已完成，opt-in）

MoE 权重下载等待期间，先用可取得的 Qwen3.5 MTP checkpoint 验证 drafter 接入、greedy correctness、cache rollback 和短 decode；不先做长性能矩阵。MTP 的完成状态与普通模型兼容性分开登记。

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 3.1 | 检查目标和 drafter 的实际 MTP 权重、结构及配对关系 | MTP 权重报告 | 不能仅凭 `mtp_num_hidden_layers` 元数据判定权重存在 |
| 3.2 | 建立普通 greedy 与 MTP greedy 的 correctness 对照 | 固定 token 对照 | token 序列完全一致；覆盖 EOS、短输出和接受/拒绝路径 |
| 3.3 | 适配 MTP propose/verify 与 hybrid cache 回滚 | 独立 opt-in 路径 | cache 状态、输出正确；不影响自回归 fallback |
| 3.4 | 短 decode A/B 与接受率检查 | MTP decode 结果 | 记录 tok/s、接受长度、峰值内存；只在正确后评估收益 |

当前 MTP 已通过 target/drafter 配对、AR token 等价、自然接受/拒绝、强制 EOS、强制拒绝和 cache rollback correctness 验证；仍保持显式 opt-in，普通 AR 路径为 fallback。现有 0.8B Q4 checkpoint 的 config 虽声明 `mtp_num_hidden_layers=1`，但 safetensors index 不包含 MTP/nextn 权重，不能单独开启 MTP。源码对照和验证记录见 [`MTP_Compatibility_Research.md`](MTP_Compatibility_Research.md)。

### 阶段 4：Qwen3.5 MoE 变体（已完成）

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 3.1 | 读取 expert/gate 配置 | MoE 结构报告 | experts、top-k、shared expert 信息完整 |
| 3.2 | 验证普通权重 MoE | smoke 结果 | 生成正确，峰值内存可记录 |
| 3.3 | 验证量化 MoE | 量化结果 | gate/expert 的量化状态无误判 |
| 3.4 | 运行 1K/4K/16K | MoE 基线 | 明确可运行上下文上限 |
| 3.5 | 完成 MiniMLX 自有 MoE 报告 | MoE 基准表 | 仅据本机正确性、重复性能和内存证据决定下一步 |

MoE 先完成 descriptor/load/短生成和内存分类，再决定是否值得做性能矩阵。MTP 留在本阶段之后；MoE 性能分析不与兼容性验证并行。

### 阶段 5：Qwen3.5 多模态边界

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 4.1 | 纯文本调用包含视觉权重的模型 | 文本边界结果 | 文本路径不崩溃，视觉权重不会误进入文本计算 |
| 4.2 | 检查图像输入所需接口 | capability 报告 | 明确当前是否支持视觉输入 |

视觉能力不阻塞纯文本 Qwen3.5 的发布状态，单独登记。

### 阶段 6：模型特异化性能分支

只有阶段 2～4 的同协议结果触发以下条件时才进入：

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 5.1 | 阅读其他 runtime 的公开 dispatch/settings（可选） | 设计差异说明 | 用于形成假设，不把其基准作为 MiniMLX 测量 |
| 5.2 | MiniMLX 阶段 profiling | 瓶颈报告 | 定位到可复现的 kernel/调度/内存问题 |
| 5.3 | 写独立 correctness A/B | 实验脚本 | 输出和 cache 状态通过 |
| 5.4 | 写端到端 A/B | 性能结果 | 1K/4K/16K 至少一个场景稳定收益，且无其他回归 |
| 5.5 | 集成 capability gate | 默认路径或 opt-in | 不满足形状/精度/硬件条件自动 fallback |
| 5.6 | 回归所有已通过 Qwen3.5 变体 | 回归报告 | 旧模型无兼容性或性能回归 |

### 阶段 7：横向扩展启动（Qwen2.5）

Qwen3.5 基线和统一支持框架完成后，横向扩展按 A2 开始推进；尚未验证的
Qwen3.5 checkpoint 留在单独台账，不作为暂停其他家族的门槛。

Qwen3.5 剩余 Dense/MoE 尺寸、特殊量化及 MTP/VLM 项目仍按原验收条件逐项
记录，但这些未测项目不再阻止下一家族进入实现和验证。

当前横向起点为 Qwen2.5：先完成 Qwen2 family 的 descriptor、provider route、
加载和短生成，再按标准 Dense 家族扩展 Llama/Gemma/Mistral。Qwen3.5 的剩余
canonical checkpoint 继续按 [`Qwen35_Family_Coverage.md`](Qwen35_Family_Coverage.md)
登记；本地 fixture 用于 conformance 验证，不改变候选家族范围。

## 目标

MiniMLX 的主目标是：逐步扩大主流模型的兼容覆盖，并形成可复现、可与其他公开结果公平对照的 MiniMLX 自有性能报告。本计划不运行 oMLX 基准；没有同协议的外部数据时不宣称胜过其他产品。

这里把“兼容性”和“性能”分成两条主线，但不是彼此割裂：兼容性工作负责扩大可用范围；性能工作只在有证据表明某个模型族存在重要差距或可复用收益时插队。

## 当前基线与边界

- 历史工程基线：MacBook Air M4，24 GB 统一内存。每份新基准记录实际运行机器，不得套用历史机器规格。
- 当前已验证模型：Qwen3.5-0.8B 的 MLX 4-bit 和 BF16。
- 性能结论只来自 MiniMLX 自有基准。外部产品数据须单独标注来源、机器与测试协议，不混入本机结果。
- 单机、单请求、Metal/MLX 是第一阶段的权威场景；不以未拥有的 M 系列硬件推断收益。
- 32K 及以上只在内存安全且结果稳定时测试；不能为了跑通测试让系统进入 swap 或牺牲其他测试的可信度。
- 同一进程不同时加载多个大模型；模型扫描只读取配置，真正验证按模型逐个进行。

## 主线 A：扩大模型支持面

## 第一阶段专项：Qwen3.5 全覆盖

第一阶段以完整登记 Qwen3.5 官方 Dense/MoE 配置和代表性量化/MTP 变体为目标。
本机无法安全运行的项目仍保留在家族覆盖表中，记录为硬件限制，不从范围中删除，
也不误标为软件不支持。

### Qwen3.5 变体覆盖表

| 维度 | 必测范围 | 通过要求 | 当前状态 |
|---|---|---|---|
| 架构 | Dense text | load + smoke + context matrix | 0.8B Q4/BF16 已通过 |
| 架构 | MoE text | expert/gate load + smoke + greedy decode | 待验证 |
| 输入 | text-only | 固定 prompt 生成正确 | 0.8B 已通过 |
| 输入 | multimodal text path | 无图文本请求不崩溃 | 待验证 |
| 输入 | image/text path | 图像预处理、视觉塔、文本生成 | 单独任务，不与纯文本混合验收 |
| 精度 | BF16/FP16 | 输出正确、内存记录 | BF16 0.8B 已通过 |
| 精度 | MLX Q4/Q8 | 量化元数据、输出正确 | Q4 0.8B 已通过 |
| 精度 | 其他可加载格式 | 明确标记格式，不套用 Q4 假设 | 待按本地模型验证 |
| 权重 | 普通权重 | 全量加载和生成 | 已验证 0.8B |
| 权重 | MTP 附加权重 | 基础模型生成正确；MTP 加速另行标记 | 待验证 |
| 尺寸 | 全部官方 Dense/MoE 尺寸 | 通过内存门槛和最小测试；受限时保留明确状态 | 详见家族覆盖表 |

### 第一阶段的具体执行顺序

#### Q35-1：建立一次性下载队列

候选顺序与精确仓库 ID 以 [`Qwen35_Family_Coverage.md`](Qwen35_Family_Coverage.md)
为准；优先 Qwen3.5，按模型热度与家族/架构覆盖推进。大小只记录为运行条件，不
用于筛除或降低优先级。每次只运行一个模型，验证完成后按需要清理权重并保留
配置摘要、原始测试日志、基准报告和支持矩阵记录。

当前真实队列（不得重复已测项）：

1. `mlx-community/Qwen3.5-9B-4bit`：已完成。
2. `mlx-community/Qwen3.5-27B-4bit`：已完成 descriptor、文本/图像验证和 MiniMLX 自有性能矩阵。
3. `mlx-community/Qwen3.5-35B-A3B-4bit`：下一项，canonical MoE；REAP-19B derivative 不替代它。
4. 按 `Qwen35_Family_Coverage.md` 完成代表型号的 Q4/Q8/BF16 与 MTP 配对覆盖。

122B/397B 仍保留在家族覆盖目标中；本机无法安全运行时登记为
`hardware-blocked`，不能从覆盖范围删除或称为软件不支持。

oMLX README 将 LLM 基础范围定义为 mlx-lm 支持的模型，并列出 Qwen3.5
122B-A10B 示例；mlx-community 当前可见 Qwen3.5 4B、9B 和 MTP 变体。具体
下载前仍以模型页面实际文件大小和 config 为准。

#### Q35-2：完成 Dense 精度矩阵

对每个本地 Dense Qwen3.5 变体依次运行：

1. descriptor/config 检查；
2. 权重加载；
3. 固定 prompt 的 4 token smoke；
4. 1K/4K/16K context matrix；
5. 32K 仅在峰值内存安全时运行；
6. 记录 MiniMLX 的 TTFT、TPOT、PP/TG TPS、E2E、吞吐量、峰值内存和实际生成 token 数。

Dense 变体只有完成第 1～4 项，才能标记为 `context-pass`；只加载成功只能标记
为 `loadable`。

#### Q35-3：验证 MoE 分支

MoE 不视为 Dense 的尺寸变体。需要单独确认：

- `num_experts`、`num_experts_per_tok` 和 expert intermediate size；
- gate、shared expert、expert 权重的量化状态；
- Q4/Q8 MoE 权重是否被当前 Q4 专用路径误判；
- 单 token decode 和长 prefill 的峰值内存；
- 输出与 mlx-lm 基线的 top-1/top-5 一致性。

MoE 的第一个目标是功能和内存稳定，不在未完成兼容性验证前做性能 kernel。

#### Q35-4：验证 multimodal 和 MTP 边界

多模态 Qwen3.5 分两级：

- 文本级：允许模型包含视觉权重，但验证纯文本分支可以正常运行；
- 视觉级：需要图像输入契约、预处理、视觉塔和融合结果，单独建立测试工具。

MTP 也分两级：

- 基础级：忽略或正确处理 MTP 附加权重后，主模型生成正确；
- 加速级：只有完成独立 correctness 和端到端收益测试后才实现。

#### Q35-5：建立按实际机器分类的硬件门槛

每个变体运行前先记录权重文件总大小，运行中记录峰值 Metal/统一内存。采用
以下分类：

- `supported`: 完成 smoke、1K/4K/16K，并且峰值内存有安全余量；
- `supported-limited-context`: 模型可运行，但 16K 或 32K 超出安全内存；
- `loadable-only`: 能加载，但无法完成规定生成或 context matrix；
- `hardware-blocked`: 预计或实际超过当前指定测试机器的安全内存预算，不能归咎于 adapter；
- `unsupported`: 配置、权重格式或模型行为尚未被当前 mlx-lm/MiniMLX 支持。

这样可以区分“我们没实现”和“当前测试机器无法安全承载”，避免错误判断。

### Qwen3.5 第一阶段完成定义

第一阶段完成需要同时满足：

1. 本机模型目录中所有 Qwen3.5 变体都已登记，没有未分类目录；
2. 所有在当前指定测试机器上可安全运行的 Dense 变体达到 `context-pass`；
3. 所有可安全运行的 MoE 变体至少达到 `smoke-pass`，并明确其量化和内存限制；
4. 多模态模型至少完成纯文本边界验证，并把视觉输入单独列为任务；
5. MTP 模型至少完成基础生成验证，并明确是否支持 MTP 加速；
6. 每个失败模型都有可复现的失败原因和状态，而不是笼统写成“不支持”；
7. 当 MiniMLX 自身重复结果出现稳定异常、OOM 或发现可复用优化机会时，才从兼容性队列切换到性能专项；本计划不运行 oMLX 基准。

第一阶段完成后，才进入 Llama、Gemma、Mistral、DeepSeek、GLM、MiniMax 等其他模型族。

### A0. 建立统一的支持契约

每个模型必须经过以下等级，不能只凭 `config.json` 判断“支持”：

1. 配置解析：识别 `model_type`、架构、层数、精度和量化信息。
2. 基础加载：通过 MiniMLX 自己的 registry/adapter 加载，不依赖 PyMiniMLX 资产。
3. 短请求验证：完成固定 prompt 的 prefill、greedy decode，并检查输出非空。
4. 形状验证：报告 linear/full attention、KV heads、hidden size、量化层等关键结构。
5. context matrix：至少 1K、4K、16K；32K 按当前实际机器的内存余量决定。
6. 性能基线：记录 TTFT、TPOT、TG TPS、E2E、峰值内存和失败原因。

状态只允许使用：`discovered`、`loadable`、`smoke-pass`、`context-pass`、`benchmarked`、`optimized`、`blocked`。

### A1. 先做通用基线，再做模型族适配

运行时优先采用“通用 mlx-lm 基线 + 小型模型族 adapter”的结构：

- 通用路径负责 tokenizer、权重加载、标准量化、cache、prefill/decode 和结果协议。
- adapter 只处理模型族确实不同的输入输出布局、cache 结构、特殊 attention 或 generation 约定。
- 特殊性能路径必须由 capability/shape/precision 明确触发，不能由模型目录名或单一模型硬编码触发。
- 每个特殊路径都必须保留通用 fallback，并通过同一组正确性测试。

这与 oMLX 的公开结构方向一致：它把“mlx-lm 可运行模型”作为基础范围，再对具体模型族增加加载和性能 dispatch；我们实现自己的 adapter 和 kernel，不复制其代码或实现细节。[oMLX README](https://github.com/jundot/omlx/blob/main/README.md)、[oMLX model loading dispatch](https://github.com/jundot/omlx/blob/main/omlx/utils/model_loading.py)

### A2. 模型族推进顺序

具体顺序以 [`Qwen35_Family_Coverage.md`](Qwen35_Family_Coverage.md) 的覆盖台账为准；
本地可用性用于判断能否立即验证，不决定模型是否进入范围：

1. Qwen3.5：优先完成家族尺寸、Dense/MoE、量化和 MTP 的剩余验证；严格跳过已测 checkpoint。
2. Qwen 系列其他架构：优先验证与 Qwen3.5 共用基础组件的版本。
3. Llama、Gemma、Mistral 等标准 decoder-only 架构：验证通用路径的复用能力。
4. DeepSeek、GLM、MiniMax、Step 等具有 MoE、DSA、混合 attention 或特殊 cache 的模型：逐族增加 adapter。
5. VLM、embedding、reranker、OCR 等非纯文本任务：在文本 LLM 基线稳定后建立独立任务契约，不把它们混入 causal-LM 的通过标准。

其他家族按届时的主流程度和架构覆盖价值排序；发现本地 checkpoint 只增加验证样本，不改变家族范围或已确定队列。

### A3. 精度和格式覆盖

同一模型族按以下顺序验证：BF16/FP16 → 标准 MLX 量化 Q4/Q8 → 其他量化或特殊格式。先确保推理正确和内存可控，再考虑性能专门化；不因 Q4 路径已优化就假设 BF16、FP16 或其他格式具有相同布局。

## 主线 B：按证据触发的 MiniMLX 性能优化

### B0. 何时暂停兼容性工作

出现以下任一条件，当前模型族的 MiniMLX 性能专项可以插队：

- MiniMLX 同一模型、同一环境的重复测试稳定偏离自身历史基线，且差距超过测量噪声。
- 公开资料提示某路径值得研究，并且 MiniMLX profiling 显示该阶段存在实际成本；不以其他 runtime 的性能结果作为必要条件。
- MiniMLX 在较长上下文出现 OOM、异常的内存增长或吞吐曲线恶化。
- 某个通用优化能被多个模型族复用，而不是只改善当前一个模型。

若 profiling 未发现可复现问题或通用收益，则记录“已研究/暂缓”，不为追逐外部实现增加维护成本。

### B1. 性能研究流程

1. 可阅读其他 runtime 的公开 model settings、加载 dispatch 和模型族分支，记录设计线索；该研究不等于运行对方基准。
2. 用 MiniMLX 的阶段基准确认差距位于 prefill、decode、attention、projection、cache、sampling 还是 orchestration。
3. 建立独立的 correctness A/B，再建立端到端 A/B；孤立 kernel 变快不能直接视为产品收益。
4. 方案必须有 shape、dtype、硬件能力和上下文长度 gate，并保留通用 fallback。
5. 至少验证 1K/4K/16K；长上下文方向在 32K 可运行时补测。decode、prefill、内存和输出正确性必须同时不回归。
6. 只有稳定收益达到触发标准，才合并默认路径；否则保留研究记录或 opt-in，不污染主路径。

### B2. 当前性能状态

前一轮 Qwen3.5 专项优化已经完成阶段性收口：GDN、Q4 prefill dense/QMM、FA-256、ANE、compile、KV cache quantization、sampling 和 cache growth 等项目均有记录，未证明有足够稳定的默认收益的项目不再反复测试。

从现在开始，性能调查只由新模型的对齐结果触发；不要因为旧实验仍在清单中，就自动重新打开旧节点。

## 主线 C：围绕实际测试机器的工程约束

- 以 Metal GPU 路径为默认；ANE/native extension 只作为有独立收益和可维护 fallback 的可选后端。
- 任何模型验证都记录权重、KV/cache、临时 buffer 和峰值内存；内存压力升高时降低并发和测试规模，不把 swap 时间计入“模型性能”。
- 标准回归矩阵：1K、4K、8K、16K；32K 条件回归；warm-up 32、最多生成 128 token、五次独立测量并报告原始值及 median/min/max。
- 结果保存模型仓库/revision、MiniMLX git revision、MLX/Metal 环境、实际硬件、精度、context、生成长度和测试时间，保证 MiniMLX 自身可复现，不要求与 oMLX 对齐。
- 不为服务器多并发提前引入容器、分布式或复杂调度；这些属于后续硬件和产品阶段。

## 已完成与当前工作节点

- [x] Qwen3.5-0.8B MLX 4-bit：兼容性和现有性能基线完成。
- [x] Qwen3.5-0.8B BF16：兼容性与 context matrix 完成；外部数据不计入 MiniMLX 自有性能结果。
- [x] 支持矩阵和统一 smoke verifier 已建立。
- [x] 根据公开支持范围建立一次性下载队列，不长期保存候选模型。
- [x] `Qwen3.5-4B-MLX-4bit`：descriptor/load/smoke/context matrix 已完成；保留现有本地模型，不重复下载。
- [x] `Qwen3.5-4B-MLX-bf16`：descriptor/load/4-token smoke、1K/4K/16K 短生成上下文检查 PASS；MLX 峰值内存 7.899 GiB，在 24 GB M4 上分类为 Supported。
- [x] Qwen3.5 MoE REAP-19B Q4：descriptor/load、40 层 MoE 路由与 shared expert、4-token 短生成 PASS，峰值 MLX 内存 11.586 GiB；混合 Q4/Q8 位宽统计已复核。
- [x] 对该 MoE 模型做 1K/4K/16K、4-token 兼容性检查：均通过；benchmark 输出未包含逐上下文峰值内存，故只登记 smoke 峰值 11.586 GiB；不记录或比较吞吐性能。
- [x] `mlx-community/Qwen3.5-27B-4bit`：descriptor、4-token text smoke、图像 OCR/vision-cache 两请求验证和 1K/4K/8K/16K 五次 MiniMLX context matrix 完成；suite peak 18.465 GiB。原始样本、设备/依赖版本和 EOS 情况记入 `Qwen35_Family_Coverage.md` 与 `Model_Benchmark_Comparison.md`。
- [x] Qwen3.5 MTP（框架迁移前基线）：4B Q4 opt-in 路径通过 128-token、64-token 同目标 AR token 对齐，1-token 输出边界，自然接受/拒绝轮次，以及受控强制拒绝和 EOS 边界；受控拒绝后继续 5 轮仍与 AR 完全一致。未做逐字段缓存快照断言。性能配对仅有单次样本，不作稳定结论。
- [x] Qwen3.5 MTP（统一推理框架迁移后）：4B Q4 在新 `InferenceRequest → resolver → ExecutionPlan → task executor` 路径通过 128-token 同目标 AR 全序列对齐（`mtp_matches_same_target_ar=True`、`compatibility=PASS`，execution route 明确包含 `mtp`）；受控拒绝后继续 5 轮及受控 EOS 边界均与同目标 AR 对齐，`mtp_boundary_suite=PASS`。修复了 Q4 verifier 对不同 token 维度使用不同 affine projection kernel 导致 argmax 不一致的问题，默认逐 verifier position 投影后再 argmax。
- [x] 横向扩展首项 `mlx-community/Qwen2.5-0.5B-Instruct-4bit`：该 checkpoint 的 Qwen2 adapter route/load/固定 prompt 生成通过；1K/4K/16K context matrix 五次重复通过。逐模型数据见 [`Model_Support_Matrix.md`](Model_Support_Matrix.md)；此记录不外推到其他 Qwen2 checkpoint。
- [x] `mlx-community/Qwen2.5-7B-Instruct-4bit`：两次固定 prompt token 一致；Qwen2 adapter route/load/smoke 通过；1K/4K/16K 五次重复均生成 128/128 token；32,767 prompt + 1 token 的 32K 边界检查通过。峰值 MLX 内存 5.483/6.297 GiB；设备、依赖版本和原始重复数据见 [`Model_Support_Matrix.md`](Model_Support_Matrix.md)。
- [x] `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`：descriptor、Qwen2 route/load、两次固定 prompt 和代码生成 smoke 通过；项目内固定代码题 pass@1 为 8/8（28/28 断言），MiniMLX 与原生 `mlx-lm` 在这 8 题上的 prompt/generation token 全部一致（两者共用 `mlx-lm`，属于包装层回归，不是独立质量分）；1K/4K/8K/16K 五次矩阵完成，4K/16K EOS 提前结束（中位数 69/67 tokens），32K 边界检查通过。数据和范围限制见 [`Model_Support_Matrix.md`](Model_Support_Matrix.md)。
- [ ] 对每个新模型完成 MiniMLX 自有重复基准报告；只有触发 B0 才暂停覆盖工作进入专项优化。

Qwen3.5 MTP 已在统一执行框架迁移后完成同目标 AR token 对齐及强制拒绝/EOS 边界验收。MTP 保持 opt-in，AR 仍为默认路径。MoE 的 smoke/context 验证已收口。MTP 性能配对仅有单次样本，不作稳定结论；后续性能工作按本计划生成 MiniMLX 自有重复报告。

## 本阶段完成定义

当 Qwen3.5 家族配置/格式均有明确状态、可运行代表模型通过规定测试、原始性能数据可复现且硬件受限项有记录时，第一阶段才算完成。之后按主流程度逐步扩展其他家族，并积累 MiniMLX 自有性能报告；没有同硬件同协议的外部结果时，不作跨产品性能结论。
