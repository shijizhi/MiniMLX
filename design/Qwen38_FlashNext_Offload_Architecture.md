# Qwen3.8-Flash-Next 分层加载架构

## 目标

为 MiniMLX 增加一条独立的 Qwen3.8-Flash-Next 执行路径，使总权重超过当前统一内存时
仍可以在 Apple Silicon 上运行，同时保持通用 MLX-LM/MLX-VLM 路径不变。

这条路径综合两个公开方向：

- Strata：专家权重分为 GPU 热缓存、RAM 常驻区和 SSD 流式区；
- oMLX：针对 `qwen4_exp` 的 PLE/N-gram SSD offload、专家共享缓冲区和异步预取。

不复制任何外部实现。MiniMLX 只借鉴可验证的架构思想，并使用自己的 MLX/Metal
执行、缓存和生命周期契约。

## 模型事实与设计判断

Qwen 官方资料显示，Qwen3.8-Flash-Next 是混合 Gated DeltaNet + QSA 的 MoE 模型，
主模型约 125B 参数、每 Token 激活约 6B 参数，并额外包含约 51B 参数的 N-gram
Embedding 表。官方明确说明 N-gram 表可以卸载到主机内存，并通过异步预取与模型计算
重叠。

因此，模型总大小不能再作为“必须常驻内存”的判断条件。需要区分：

```text
常驻部分 = dense backbone + shared expert + KV/state + 热点缓存
流式部分 = 冷专家 + PLE/N-gram 冷数据
运行时峰值 = 常驻部分 + 当前 Token 所需的预取窗口 + 临时 workspace
```

## 不变的架构边界

Qwen3.8-Flash-Next 不进入 Qwen3.5 专用 provider，也不修改通用 MLX-LM 的全局类。
新增组件应保持现有分层：

```text
ModelDescriptor
    ↓
CapabilityResolver
    ↓
Qwen38FlashNextExecutionPlan
    ├── Qwen4ExpFamilyAdapter       结构与 checkpoint 语义
    ├── ResidentMemoryPlanner       常驻预算与峰值预算
    ├── MmapWeightStore              SSD shard / PLE / expert 读取
    ├── ExpertCache                  GPU/RAM 热点缓存
    ├── PrefetchScheduler            路由后的异步预取
    └── MLX/Metal Executor           GDN、QSA、MoE、KV/state 执行
```

通用路径仍然是默认 fallback。offload 路径只有在模型身份、权重布局、硬件能力和
correctness probe 全部通过时才启用。

## 分层内存模型

### 1. GPU 常驻层

优先放置：

- GDN/QSA 所需的 dense projection；
- shared expert；
- router、norm、HyperConnection 和输出头；
- 当前上下文的 KV cache 与 GDN recurrent state；
- 根据专家热度选择的 GPU expert cache。

GPU 层不能因为缓存不足而加载整个模型；缺失专家必须回退到 RAM 或 SSD 层。

### 2. RAM 常驻层

RAM 预算由用户可配置，但必须保留系统安全余量：

```text
ram_budget = min(
    configured_budget,
    total_memory - os_reserve - runtime_reserve - kv_reserve
)
```

RAM 层保存：

- 高频专家的解码格式；
- PLE/N-gram 热点页；
- SSD 读取的页缓存和预取窗口；
- 必要的 CPU fallback workspace。

预算不足时，只减少热点缓存，不把整个模型加载失败；但如果连 dense backbone、当前
workspace 和最小专家窗口都无法满足，则安全拒绝启动。

### 3. SSD 流式层

SSD 层保存完整的模型分片，使用只读 mmap 或等价的页粒度读取。第一版不允许把
整个 SSD 文件复制为第二份 RAM buffer，也不允许对全量专家执行 eager materialize。

SSD 流式层需要记录：

- shard、offset、长度和 dtype；
- tensor/专家到文件区间的索引；
- checksum、checkpoint revision 和布局版本；
- page fault、读取字节数和预取命中率。

## 两阶段实现顺序

### 阶段 A：PLE/N-gram offload

先只处理额外的 N-gram Embedding 表，不改专家执行：

1. 识别 `qwen4_exp` 配置和 PLE tensor；
2. 将 PLE 映射为只读 shard；
3. 根据 token hash 计算页索引；
4. 在 CPU/文件层发起异步读取；
5. 将页结果送入 MLX 计算，并记录命中与等待时间；
6. 无法使用 mmap 或预取失败时回退为普通加载，不能静默产生错误输出。

阶段 A 的价值是验证“超出内存的结构化表可以安全流式读取”，风险低于直接改 MoE。

### 阶段 B：专家分层缓存

在阶段 A 稳定后，再增加专家缓存：

```text
router output
    → 当前层 expert ids
    → GPU cache lookup
    → RAM cache lookup
    → SSD prefetch/read on miss
    → expert compute
    → async cache promotion / eviction
```

缓存必须按层管理，不能把所有层的专家混在一个无界 LRU 中。第一版使用简单的
LRU/频率混合策略；后续再根据稳定的 expert profile 做预热。

### 阶段 C：Qwen4Exp 专用执行器

最后接入完整的 GDN、QSA、四路 residual/HyperConnection、PLE 和 MoE 执行计划。
所有专用 kernel 必须有：

- 单层 reference 对照；
- 固定 prompt token 对照；
- prefill/decode 分开 benchmark；
- offload 关闭时的同模型 baseline；
- 任何缓存 miss、预取失败和 OOM 的可诊断错误。

## 预取与调度原则

预取不能阻塞 router，也不能无限制读取 SSD。每个 decode step 只允许有限的 in-flight
窗口：

```text
当前层路由完成
    → 合并并排序 expert ids
    → 去重 cache hit
    → 对 miss 生成 bounded prefetch batch
    → 等待当前层所需最小集合
    → 计算并同时准备下一层
```

第一版宁可等待 SSD，也不能为了吞吐无限扩大 RAM 或文件缓存。调度器要把以下时间
分开记录：router、cache hit、SSD read、dequant/pack、GPU compute、同步等待。

## 用户可见的内存策略

新增模型设置：

- `Resident memory budget`：自动 / 固定 GiB；
- `SSD offload`：自动 / 开启 / 关闭；
- `Expert cache policy`：保守 / 平衡 / 性能；
- `Maximum context`：仍然参与 KV/state 预算；
- `Allow below recommended budget`：显式确认后才允许启动。

默认策略为自动：优先保证系统响应和正确性，不为追求加载成功而耗尽统一内存。

UI 应显示：

- 总权重大小；
- 预计 GPU/RAM 常驻大小；
- 预计 SSD 流式大小；
- 当前 context 对 KV/state 的额外占用；
- 预取命中率、SSD 读取速率和因内存压力暂停的请求数。

## 安全门控与回退

以下任一条件不满足，offload 路径不得自动启用：

- checkpoint 布局没有完整索引或 shard checksum 不匹配；
- `qwen4_exp` 结构字段不完整；
- Metal/MLX 版本不满足已验证范围；
- mmap、异步读取或 pinned buffer 不可用；
- correctness probe 与 baseline 不一致；
- 最小常驻预算无法满足。

回退顺序：

1. offload + 专用执行器；
2. offload + reference executor；
3. 普通 MLX 加载路径（仅当完整权重适合内存）；
4. 明确拒绝并解释需要的内存、磁盘和 context 预算。

## 验收指标

### 正确性

- 4K、16K、32K context 的固定 prompt 输出与 reference 对齐；
- PLE offload 开关只影响性能，不改变 token 结果；
- expert cache 容量变化不改变 greedy 输出；
- 连续 1000 token decode 无 cache corruption、死锁或无界内存增长。

### 资源

- 记录常驻 RAM、峰值 RAM、GPU active/cache、SSD 读取量；
- 在 32/48/64/96 GB Apple Silicon 上分别记录可运行条件；
- 把 context、PLE cache、expert cache 和 KV/state 预算分开报告。

### 性能

- SSD offload 的首 token、decode TPS、prefill TPS；
- cache hit/miss 比例；
- 预取隐藏比例；
- 和同一 checkpoint 的完整加载 baseline 对比；
- 不以“能启动”单独作为成功标准。

## 第一批工作项

1. 增加独立 `qwen4_exp` descriptor，不复用 Qwen3.5 专用身份。
2. 读取一个真实 Qwen3.8-Flash-Next checkpoint 的 config、tensor index 和 PLE 布局，
   暂不下载完整权重。
3. 建立 `OffloadPlan`、`ResidentBudget` 和 `MmapWeightStore` 的无设备单元测试。
4. 实现 PLE shard 的只读映射与页级统计。
5. 用小型 fixture 验证 PLE lookup 的 reference/streaming 一致性。
6. 再决定是复用 mlx-vlm 的 qwen4_exp 执行器，还是建立 MiniMLX family adapter。

## 参考资料

- Qwen 官方模型仓库：<https://github.com/QwenLM/Qwen3.8-Flash-Next>
- Strata 工作原理：<https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md>
- Strata Qwen3.8 分层加载：<https://github.com/Niko1221/Strata/blob/main/docs/UNSLOTH_Q4.md>
- oMLX 发布说明：<https://github.com/jundot/omlx/releases>
- oMLX Qwen3.8 支持讨论：<https://github.com/jundot/omlx/discussions/3191>
