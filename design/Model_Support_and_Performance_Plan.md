# MiniMLX 模型覆盖与性能推进计划

基准结果统一记录在 [`Model_Benchmark_Comparison.md`](Model_Benchmark_Comparison.md)。

## 执行队列

以下是实际执行顺序。每完成一个步骤，就更新支持矩阵和状态；没有完成当前步骤的退出条件，不进入下一步。

### 阶段 1：Qwen3.5 本地库存和基线

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 1.1 | 根据公开支持范围建立一次性下载队列 | Qwen3.5 变体清单 | 每次只保留一个待测模型 |
| 1.2 | 对每个目录运行 `verify_model_support.py` | smoke 结果 | 每个模型归类为 smoke-pass、loadable-only、hardware-blocked 或 unsupported |
| 1.3 | 修复通用配置读取和 adapter 分发 | registry/adapter 变更 | 不再依赖目录名；同类模型能进入同一 adapter |
| 1.4 | 建立失败样本回归测试 | 固定失败用例 | 同一模型再次运行得到同一失败分类 |

当前状态：0.8B Q4/BF16 与 4B Q4 兼容性均已通过。4B BF16 已通过 descriptor/load/4-token smoke、1K/4K/16K 短生成上下文检查，MLX 峰值内存 7.899 GiB；在 24 GB M4 上登记为 Supported，不开展性能调优。

### 阶段 2：Qwen3.5 Dense 变体

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 2.1 | 收集本机可用 Dense 变体 | 支持矩阵行 | 每种尺寸/精度/量化各一行 |
| 2.2 | 运行短 prompt smoke | 验证日志 | prefill、decode、token 输出均正确 |
| 2.3 | 运行 1K/4K/16K context matrix | 性能基线 | 记录 TTFT、TPOT、TG TPS、E2E、峰值内存 |
| 2.4 | 尝试 32K | 长上下文结果 | 通过则标记 context-pass；失败则记录内存原因 |
| 2.5 | 与 oMLX 同协议对齐 | 对比表 | 每个模型完成一次同硬件对比 |
| 2.6 | 处理差异 | 性能决策记录 | 领先：暂缓优化；落后/异常：转阶段 5 |

阶段 2 的执行顺序固定为：一次下载一个模型，测试完成后删除，再下载下一项。
格式顺序为 BF16/FP16 → Q4/Q8 → 其他格式；不先写格式专用优化。

### 阶段 3：Qwen3.5 MTP 兼容性与 decode（当前工作）

MoE 权重下载等待期间，先用可取得的 Qwen3.5 MTP checkpoint 验证 drafter 接入、greedy correctness、cache rollback 和短 decode；不先做长性能矩阵。MTP 的完成状态与普通模型兼容性分开登记。

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 3.1 | 检查目标和 drafter 的实际 MTP 权重、结构及配对关系 | MTP 权重报告 | 不能仅凭 `mtp_num_hidden_layers` 元数据判定权重存在 |
| 3.2 | 建立普通 greedy 与 MTP greedy 的 correctness 对照 | 固定 token 对照 | token 序列完全一致；覆盖 EOS、短输出和接受/拒绝路径 |
| 3.3 | 适配 MTP propose/verify 与 hybrid cache 回滚 | 独立 opt-in 路径 | cache 状态、输出正确；不影响自回归 fallback |
| 3.4 | 短 decode A/B 与接受率检查 | MTP decode 结果 | 记录 tok/s、接受长度、峰值内存；只在正确后评估收益 |

当前 MiniMLX 基础 decode 尚未实现 MTP。现有 0.8B Q4 checkpoint 的 config 虽声明 `mtp_num_hidden_layers=1`，但 safetensors index 不包含 MTP/nextn 权重；该 checkpoint 只能作为 target baseline，不能单独开启 MTP。源码对照和当前调查记录见 [`MTP_Compatibility_Research.md`](MTP_Compatibility_Research.md)。

### 阶段 4：Qwen3.5 MoE 变体（已完成）

| 步骤 | 操作 | 产物 | 退出条件 |
|---|---|---|---|
| 3.1 | 读取 expert/gate 配置 | MoE 结构报告 | experts、top-k、shared expert 信息完整 |
| 3.2 | 验证普通权重 MoE | smoke 结果 | 生成正确，峰值内存可记录 |
| 3.3 | 验证量化 MoE | 量化结果 | gate/expert 的量化状态无误判 |
| 3.4 | 运行 1K/4K/16K | MoE 基线 | 明确可运行上下文上限 |
| 3.5 | 对比 oMLX | MoE 对比表 | 决定暂缓或转阶段 6 |

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
| 5.1 | 对照 oMLX 公开 dispatch/settings | 差异说明 | 找到具体模型阶段，不凭总耗时猜测 |
| 5.2 | MiniMLX 阶段 profiling | 瓶颈报告 | 定位到可复现的 kernel/调度/内存问题 |
| 5.3 | 写独立 correctness A/B | 实验脚本 | 输出和 cache 状态通过 |
| 5.4 | 写端到端 A/B | 性能结果 | 1K/4K/16K 至少一个场景稳定收益，且无其他回归 |
| 5.5 | 集成 capability gate | 默认路径或 opt-in | 不满足形状/精度/硬件条件自动 fallback |
| 5.6 | 回归所有已通过 Qwen3.5 变体 | 回归报告 | 旧模型无兼容性或性能回归 |

### 阶段 7：转入下一个模型族

只有以下条件全部满足，才离开 Qwen3.5 阶段：

- 本机所有 Qwen3.5 目录均已分类；
- 可安全运行的 Dense 变体达到 context-pass；
- MoE 变体至少完成 smoke 和内存分类；
- 多模态/MTP 边界有明确能力标记；
- 所有与 oMLX 的差异都有“优化/暂缓/硬件阻塞”结论；
- 支持矩阵、验证脚本和失败回归用例已更新。

下一模型族默认从 Llama/Gemma/Mistral 中选择本机实际可取得者，重复阶段 1～5。

## 目标

MiniMLX 的主目标是：在当前可验证的硬件条件下，逐步覆盖 oMLX 能运行的模型范围，并在同一模型、同一精度、同一上下文和同一测试协议下持续超过 oMLX。

这里把“兼容性”和“性能”分成两条主线，但不是彼此割裂：兼容性工作负责扩大可用范围；性能工作只在有证据表明某个模型族存在重要差距或可复用收益时插队。

## 当前基线与边界

- 主要硬件：MacBook Air M4，24 GB 统一内存。
- 当前已验证模型：Qwen3.5-0.8B 的 MLX 4-bit 和 BF16。
- 当前 Qwen3.5 BF16 已通过短生成和 1K/2K/4K/8K/16K context matrix；当前 MiniMLX 在已提供的对齐数据中 TTFT、TPOT 和 TG TPS 均优于 oMLX。
- 单机、单请求、Metal/MLX 是第一阶段的权威场景；不以未拥有的 M 系列硬件推断收益。
- 32K 及以上只在内存安全且结果稳定时测试；不能为了跑通测试让系统进入 swap 或牺牲其他测试的可信度。
- 同一进程不同时加载多个大模型；模型扫描只读取配置，真正验证按模型逐个进行。

## 主线 A：扩大模型支持面

## 第一阶段专项：Qwen3.5 全覆盖

第一阶段不以“支持一个 0.8B 模型”为目标，而是以 Qwen3.5 家族在当前
MacBook Air M4、24 GB 统一内存上能够安全验证的全部变体为目标。这里的
“全部”指：模型文件已经可获得、基础依赖支持、运行时不会依赖 swap，并且
至少能完成规定的最小测试；超过 24 GB 可用内存的变体记录为硬件阻塞，不能
伪装成软件不支持。

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
| 尺寸 | 所有能在 24 GB 内安全加载的尺寸 | 通过内存门槛和最小测试 | 目前只有 0.8B |

### 第一阶段的具体执行顺序

#### Q35-1：建立一次性下载队列

不扫描并长期保存模型库存。根据 oMLX 的公开支持范围和 mlx-community 的可下载
模型页面建立候选队列，每次只保留一个待测模型；验证完成后删除权重，只保留
配置摘要、测试日志和支持矩阵记录。

第一批候选按硬件风险排序：

1. `Qwen3.5-4B-MLX-4bit`：约 3 GB，优先验证尺寸扩展和视觉权重共存的文本路径。
2. `Qwen3.5-4B-MLX-bf16`：验证同一尺寸的 BF16 与 Q4 差异。
3. `Qwen3.5-9B-4bit`：约 6 GB，验证更大 checkpoint 在 24 GB 上的边界。
4. `Qwen3.5-9B-bf16` 或同等 BF16 变体：只有内存余量允许时下载。
5. Qwen3.5 MoE 的最小 4-bit 变体：验证 expert/gate，不默认下载大尺寸。
6. 27B/35B-A3B 的 4-bit 变体：只在前面模型清理完且内存预算允许时尝试。

122B/397B 及更大模型不进入当前 MacBook Air 24 GB 下载队列；记录为
`hardware-blocked`，不浪费下载空间。

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
6. 记录 MiniMLX/oMLX 的 TTFT、TPOT、TG TPS、E2E 和峰值内存。

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

#### Q35-5：建立 24 GB 硬件门槛

每个变体运行前先记录权重文件总大小，运行中记录峰值 Metal/统一内存。采用
以下分类：

- `supported`: 完成 smoke、1K/4K/16K，并且峰值内存有安全余量；
- `supported-limited-context`: 模型可运行，但 16K 或 32K 超出安全内存；
- `loadable-only`: 能加载，但无法完成规定生成或 context matrix；
- `hardware-blocked`: 预计或实际需要超过 24 GB，不能归咎于 adapter；
- `unsupported`: 配置、权重格式或模型行为尚未被当前 mlx-lm/MiniMLX 支持。

这样可以区分“我们没实现”和“MacBook Air 24 GB 无法承载”，避免错误判断。

### Qwen3.5 第一阶段完成定义

第一阶段完成需要同时满足：

1. 本机模型目录中所有 Qwen3.5 变体都已登记，没有未分类目录；
2. 所有 24 GB 内存可安全运行的 Dense 变体达到 `context-pass`；
3. 所有可安全运行的 MoE 变体至少达到 `smoke-pass`，并明确其量化和内存限制；
4. 多模态模型至少完成纯文本边界验证，并把视觉输入单独列为任务；
5. MTP 模型至少完成基础生成验证，并明确是否支持 MTP 加速；
6. 每个失败模型都有可复现的失败原因和状态，而不是笼统写成“不支持”；
7. 只有当同模型同协议对比 oMLX 后出现稳定落后、OOM 或可复用优化机会，才从兼容性队列切换到性能队列。

第一阶段完成后，才进入 Llama、Gemma、Mistral、DeepSeek、GLM、MiniMax 等其他模型族。

### A0. 建立统一的支持契约

每个模型必须经过以下等级，不能只凭 `config.json` 判断“支持”：

1. 配置解析：识别 `model_type`、架构、层数、精度和量化信息。
2. 基础加载：通过 MiniMLX 自己的 registry/adapter 加载，不依赖 PyMiniMLX 资产。
3. 短请求验证：完成固定 prompt 的 prefill、greedy decode，并检查输出非空。
4. 形状验证：报告 linear/full attention、KV heads、hidden size、量化层等关键结构。
5. context matrix：至少 1K、4K、16K；32K 按 24 GB 内存余量决定。
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

具体顺序由本机实际可用模型清单决定，默认按以下优先级推进：

1. Qwen3.5：4-bit/BF16 已完成，继续覆盖本地存在的尺寸和精度变体。
2. Qwen 系列其他架构：优先验证与 Qwen3.5 共用基础组件的版本。
3. Llama、Gemma、Mistral 等标准 decoder-only 架构：验证通用路径的复用能力。
4. DeepSeek、GLM、MiniMax、Step 等具有 MoE、DSA、混合 attention 或特殊 cache 的模型：逐族增加 adapter。
5. VLM、embedding、reranker、OCR 等非纯文本任务：在文本 LLM 基线稳定后建立独立任务契约，不把它们混入 causal-LM 的通过标准。

这个顺序不是对 oMLX 当前全部模型支持的承诺，而是本机可下载、可验证、可回归的工作队列。每发现一个本地模型，先进入清单，再决定是否进入实现队列。

### A3. 精度和格式覆盖

同一模型族按以下顺序验证：BF16/FP16 → 标准 MLX 量化 Q4/Q8 → 其他量化或特殊格式。先确保推理正确和内存可控，再考虑性能专门化；不因 Q4 路径已优化就假设 BF16、FP16 或其他格式具有相同布局。

## 主线 B：按证据触发的性能竞争

### B0. 何时暂停兼容性工作

出现以下任一条件，当前模型族的性能工作可以插队：

- 同模型、同硬件、同精度、同上下文和同 warm-up/repeats 协议下，MiniMLX 稳定落后 oMLX，且差距超过测量噪声（暂以连续多次中位数差异约 5% 作为研究触发线）。
- oMLX 针对该模型存在公开的专用 dispatch，而 MiniMLX 的同一阶段明显落后。
- MiniMLX 在较长上下文出现 OOM、异常的内存增长或吞吐曲线恶化，而 oMLX 能稳定完成。
- 某个通用优化能被多个模型族复用，而不是只改善当前一个模型。

如果 MiniMLX 已经领先，则记录“已研究/暂缓”，不为了追随 oMLX 而增加维护成本。

### B1. 性能研究流程

1. 先阅读 oMLX 的公开 model settings、加载 dispatch 和模型族分支，记录它解决的具体问题；只做行为和设计层面的借鉴。
2. 用 MiniMLX 的阶段基准确认差距位于 prefill、decode、attention、projection、cache、sampling 还是 orchestration。
3. 建立独立的 correctness A/B，再建立端到端 A/B；孤立 kernel 变快不能直接视为产品收益。
4. 方案必须有 shape、dtype、硬件能力和上下文长度 gate，并保留通用 fallback。
5. 至少验证 1K/4K/16K；长上下文方向在 32K 可运行时补测。decode、prefill、内存和输出正确性必须同时不回归。
6. 只有稳定收益达到触发标准，才合并默认路径；否则保留研究记录或 opt-in，不污染主路径。

### B2. 当前性能状态

前一轮 Qwen3.5 专项优化已经完成阶段性收口：GDN、Q4 prefill dense/QMM、FA-256、ANE、compile、KV cache quantization、sampling 和 cache growth 等项目均有记录，未证明有足够稳定的默认收益的项目不再反复测试。

从现在开始，性能调查只由新模型的对齐结果触发；不要因为旧实验仍在清单中，就自动重新打开旧节点。

## 主线 C：围绕 24 GB M4 的工程约束

- 以 Metal GPU 路径为默认；ANE/native extension 只作为有独立收益和可维护 fallback 的可选后端。
- 任何模型验证都记录权重、KV/cache、临时 buffer 和峰值内存；接近 24 GB 时降低并发和测试规模，不把 swap 时间计入“模型性能”。
- 标准回归矩阵：1K、4K、16K；32K 条件回归；每个模型至少 warm-up 32、重复 5 次取 median。
- 结果必须保存模型路径、git revision、MLX/Metal 环境、精度、context、生成长度和测试时间，保证与 oMLX 对齐。
- 不为服务器多并发提前引入容器、分布式或复杂调度；这些属于后续硬件和产品阶段。

## 已完成与当前工作节点

- [x] Qwen3.5-0.8B MLX 4-bit：兼容性和现有性能基线完成。
- [x] Qwen3.5-0.8B BF16：兼容性、context matrix 和 oMLX 对齐数据完成。
- [x] 支持矩阵和统一 smoke verifier 已建立。
- [x] 根据公开支持范围建立一次性下载队列，不长期保存候选模型。
- [x] `Qwen3.5-4B-MLX-4bit`：descriptor/load/smoke/context matrix 已完成；保留现有本地模型，不重复下载。
- [x] `Qwen3.5-4B-MLX-bf16`：descriptor/load/4-token smoke、1K/4K/16K 短生成上下文检查 PASS；MLX 峰值内存 7.899 GiB，在 24 GB M4 上分类为 Supported。
- [x] Qwen3.5 MoE REAP-19B Q4：descriptor/load、40 层 MoE 路由与 shared expert、4-token 短生成 PASS，峰值 MLX 内存 11.586 GiB；混合 Q4/Q8 位宽统计已复核。
- [x] 对该 MoE 模型做 1K/4K/16K、4-token 兼容性检查：均通过；benchmark 输出未包含逐上下文峰值内存，故只登记 smoke 峰值 11.586 GiB；不记录或比较吞吐性能。
- [x] Qwen3.5 MTP：4B Q4 opt-in 路径通过 128-token、64-token 同目标 AR token 对齐，1-token 输出边界，自然接受/拒绝轮次，以及受控强制拒绝和 EOS 边界；受控拒绝后继续 5 轮仍与 AR 完全一致。未做逐字段缓存快照断言。性能配对仅有单次样本，不作稳定结论。
- [ ] 对每个新模型与 oMLX 做同协议对齐；只有触发 B0 才暂停覆盖工作进入专项优化。

因此，Qwen3.5 MTP correctness 节点已收口，下一步按队列转入 9B 与多模态边界。MTP 已通过已测提示和受控 EOS/拒绝场景的同目标 AR token 对齐；单次配对 decode 测量显示有收益迹象，但不登记为稳定性能结论。MTP 保持 opt-in，AR 仍为默认路径。MoE 的 smoke/context 验证已收口；用户提供的 MiniMLX 128-token 数据优于此前 oMLX 截图，但截图测试批次为 2、MiniMLX 为单请求，故只作为参考，不登记为同协议性能结论。每次只推进一个验证节点；不是扫描本地库存，也不是重新打开旧的 Qwen3.5 性能分支。

## 本阶段完成定义

当本机候选模型都至少达到 `smoke-pass`，主要模型族各有一个 `context-pass` 基线，并且每个落后 oMLX 的模型都有已登记的性能决策（优化、暂缓或受硬件限制），第一阶段才算完成。最终目标仍然是：在可验证的 M4/24 GB 场景中，支持面不明显落后 oMLX，并在相同测试协议下整体性能超过 oMLX。
