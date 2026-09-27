# MiniMLX performance optimization roadmap

目标：在相同模型、硬件和 benchmark 条件下超过 oMLX。所有优化都必须保留通用模型能力；模型特化只能通过严格的能力检测、形状门控和 fallback 生效。

## Checkpoint — 2026-09-21

这是当前性能优化循环的唯一恢复点。原计划清单已经闭环；后续工作必须从“新计划项”
继续，不得重新打开已完成或已否决的节点，除非出现新的硬件、MLX 版本或正确性证据。

### 已完成并封项

- GDN Prefill custom kernel：正确性通过，完整路径已确认约节省 50–70 ms TTFT。
- Q4 dense MLP Prefill：1K/4K/16K/32K A/B 已完成；4K、16K、32K 仍为正收益，
  保留严格长度和推理门控。
- FA-256 fused attention：隔离算子有收益，但完整生产路径回退，保持 opt-in。
- ANE 单投影和阶段级 offload：边界成本抵消收益，不进入默认路径。
- Prefix/session cache：已接入 opt-in；SSD/paged 长上下文缓存明确延后。
- Prefill LM-head suppression：生产路径已有惰性裁剪，实际收益不足，不接入。
- Decode `mx.compile`：固定单 token cached Decode 已通过 correctness，但 stock 6.405 ms
  vs compiled 6.401 ms，收益约 0.06%，关闭，不进入生产路径。
- KV cache quantization：4K/16K/32K 均节省约 34%/43%/45% cache bytes，但 decode
  分别慢约 7.3%/5.7%/37.0%；top-1 保持一致，关闭性能路径，仅保留未来显存模式候选。

### 已研究并否决/暂不适用

- 独立 native-Q4 tile：慢于 MLX 原生 QMM。
- QxA8/W4A8：当前 M4/MLX 组合没有可复用入口，暂不写运行时路径。
- 自适应 Prefill chunk：当前没有优于显式 1024 的证据。
- 长上下文 O(L) attention 原型：现有独立 kernel 常数过大，未超过 stock SDPA。

### 当前运行基线

`config/minimlx.local.json` 当前为：Qwen3.5-0.8B-MLX-4bit、4096 prompt tokens、
128 generated tokens、warmup 32、repeats 5、prefill step 1024、Q4 dense enabled。

### 原计划状态：已闭环

长上下文请求调度与 prefill 图边界已经完成研究：Full-Attention prefill force-fused、
allocator-cache 保留、clear cadence、orchestration 和 input embedding 预计算均已完成
正确性与跨长度 A/B，均没有形成稳定的默认路径收益；KVCache growth policy 已完成长度门控。
因此原计划没有剩余的未测节点，不再把这些实验入口当作“下一项”。

不得重新打开 decode SDPA、已完成的 force-fused、采样、compile、KV 量化或 GDN Prefill
节点。若继续追求性能，必须先把公开实现证据或新的硬件测量写成新的计划项，再进入实验。

Prefill orchestration 对照也已完成：直接复现 MLX-LM 的 chunk/eval/clear 调度在 4K、
16K、32K 没有稳定优势，不能把 Python generator 当作生产优化点。整段 input embedding
预计算实验随后完成并已封项，工具为 `tools/experiment_prefill_input_embeddings.py`。

Input embedding 预计算也已完成：4K/16K/32K 分别比 token-id 路径慢约 10/141/815 ms，
虽然 token correctness 通过，但额外 embedding 物化和同步成本随长度放大；不进入运行时。
当前 prefill 图边界分支封项，后续候选必须来自新的公开实现证据或新的硬件测量。
根据 oMLX 对 Qwen MLP per-call gate 的公开实现说明，曾尝试 Q4 MLP dispatch wrapper
去重；当前 4K 完整 benchmark 未显示收益且方向回退，因此不进入主线，后续只有在严格
paired A/B 证明稳定收益时才重新考虑。
单请求 Full-Attention KVCache 预分配暂不作为当前主线：oMLX 的 KV cache 优势主要来自
paged/block cache、prefix sharing、Copy-on-Write、SSD tiering 和 continuous batching，
不是单请求 `KVCache.step` 预分配。已有 `tools/experiment_full_attention_cache_growth.py`
可保留作诊断，但在单请求路径没有新的证据前不继续投入。

历史实验入口：`tools/experiment_prefill_clear_cadence.py`。该实验已完成：保持每个 chunk 的
`mx.eval(cache.state)` 不变，仅比较 `mx.clear_cache()` 每 1/2/4/8 个 chunk 执行一次。
4K、16K、32K 均为每 chunk 清理最快，延迟清理没有降低峰值显存，token correctness 全部通过；
因此保持默认每 chunk 清理，不再重复测试该分支。


## 执行规则

1. 当前只比较 MiniMLX 与 oMLX；PyMiniMLX 不再参与性能决策。
2. 每项优化先研究公开算法/实现，再做独立实现，不复制 oMLX 代码。
3. 每项优化都要验证短上下文、4K、16K、32K+（适用时）以及 decode 不回退。
4. 只要某个支持场景稳定有收益，就通过 dispatch 保留；不能因为 4K 收益小而直接删除。
5. 每项完成前必须有：正确性、shape/硬件门控、A/B benchmark、fallback 和长期记录。

## 历史候选顺序

## 当前执行状态

这张表是唯一的推进依据；未列入“当前工作项”的方向不得临时插入。

| 项目 | 状态 | 结论 | 当前动作 |
|---|---|---|---|
| FA-256 full-attention prefill | 已完成/否决默认启用 | 原生 fused 隔离路径未改善完整生产路径 | 保留严格 opt-in，不继续调参 |
| Q4 prefill dense MLP | 已完成 | 1K 基本无收益，4K/16K 约 2.9%，32K 仍有约 0.66% Prefill 加速 | 保留严格门控，进入下一项 |
| QxA8/W4A8 | 能力研究完成/暂不适用 | M4 上暂无可直接复用的 W4A8 路径 | 不进入当前运行时，保留硬件升级后的复查入口 |
| ANE 阶段 offload | 已完成/否决 | 隔离算子有收益，但阶段/整机边界成本抵消收益 | 保留实验代码，不进入默认路径 |
| GDN Prefill custom kernel | 已完成 | 正确，完整路径约节省 50–70 ms TTFT | 封项，不再重复测 block size |
| 自适应 Prefill 调度 | 阶段性否决 | 当前长度规则未证明优于显式 chunk | 保留 opt-in，不作为当前目标 |
| Prefix/KV cache | 首轮已落地 | session prefix reuse 有显著收益 | 只补遥测/淘汰；SSD 长上下文延后 |
| Prefill LM-head suppression | 已验证但否决 | 生产路径本已有惰性裁剪，实际仅约 0.8% | 不进入运行时 |
| Decode `mx.compile` | 已完成/否决 | correctness PASS；6.405 vs 6.401 ms，约 0.06% | 不进入默认路径 |
| KV cache quantization | 已完成/否决性能路径 | 4K/16K/32K 都无速度收益，只降低 cache bytes | 不进入默认性能路径，保留显存模式候选 |
| Decode sampling/postprocess | 端到端否决 | 算子级仅约 0.45%/0.22% 正收益，但自有同步生成循环使 TPOT 约 5.87→6.85 ms，E2E 回退约 66.7 ms | 默认关闭；保留实验代码，若研究异步调度再复评 |
| Full-Attention decode / SDPA dispatch | 已完成/否决默认启用 | 隔离约快 1.2%–4.5%，端到端 paired A/B 仅约 0.2% | 保留实验能力，不进入默认路径 |
| Full-Attention decode Q/K/V projection boundary | 阶段性否决 | 阶段数据表现为近似固定成本，未显示随 context 增长的主瓶颈 | 不作为当前工作项 |
| Full-Attention decode KVCache growth policy | 已验证/长度门控保留 | 1K/4K/16K 保持 256；32K 使用 1024，decode 约快 0.4%，TTFT 方向稳定但受长上下文波动影响 | 默认按 context 自动选择 |
| Long-context Full-Attention prefill force-fused | 已完成/生产路径否决 | 4K/16K 隔离约 1.7x/1.8x，32K stock 单算子 OOM，但当前 1024 分块完整路径无稳定收益 | 保留严格 opt-in，不进入默认路径 |
| Prefill allocator cache policy | 已完成/否决 | 独立 16K/32K 和 32K 完整路径均显示保留 `mx.clear_cache()` 更快；跳过清理增加耗时 | 保持默认清理，不进入运行时 |
| Prefill clear cadence | 已完成/否决 | 4K/16K/32K 中每 chunk 清理均最快；延迟清理无显存收益且更慢 | 封项，不再重复测试 |
| Prefill input embedding precompute | 已完成/否决 | 4K/16K/32K 均慢，额外物化/同步成本随长度放大 | 封项，不再重复测试 |
| Q4 MLP dispatch wrapper 去重 | 已尝试/暂不采用 | 当前 4K benchmark 未显示收益且 TPOT/TG TPS 回退 | 不进入默认路径 |
| Full-Attention KVCache preallocation | 降级/暂缓 | oMLX 的主要 KV 优化是服务层分页、共享和 SSD tiering；单请求预分配不是其核心差异点 | 保留诊断工具，不作为当前主线 |
| GDN Q4 prefill projection native path | 暂缓/未来候选 | 两个 JIT probe 均不等价于 oMLX 的 compiled MLX primitive；但当前 MiniMLX 与 oMLX 的 Prefill 差距约 2%–4%，native extension 的潜在收益不足以覆盖 MLX ABI、版本适配和维护成本 | 不进入当前优化循环；保留研究记录，未来在 MLX 版本稳定或出现更大收益证据时复评 |

### 原计划完成状态

Q4 dense MLP 的跨长度完整验证已经完成。QxA8 能力探针、Decode 固定图编译、KV cache
quantization 和 greedy skip-logsumexp 都已完成独立 A/B；其中采样优化在算子级略快，
但自定义循环破坏异步调度，完整 benchmark 回退，因此默认关闭。Long-context request
scheduling / prefill graph boundary 也已经完成研究，不改变默认路径。decode SDPA、prefill
force-fused 和 allocator-cache 保留已验证但不进入默认路径：隔离测试虽有
小幅收益，真实 paired A/B 为 TPOT 5.90 vs 5.91 ms、E2E 2565.2 vs 2569.5 ms，
不足以归因于稳定性能提升。原计划至此完成，不再继续重复这些实验。

### 最终 MiniMLX / oMLX 对齐结果

当前统一口径为 Qwen3.5-0.8B-MLX-4bit、4K prompt、128 generated tokens、warmup 32、
正式运行 5 次取 median、prefill step 1024。

| 实现 | TTFT | TPOT | TG TPS | 备注 |
|---|---:|---:|---:|---|
| MiniMLX（本次串行复测） | 1891.2 ms | 6.26 ms | 159.75 | decode 更快；本机波动较大 |
| oMLX（已提供的同口径结果） | 1802.4 ms | 6.70 ms | 151.4 | 作为对照 |

MiniMLX 本次 TPOT 低约 6.6%，TG TPS 高约 5.5%；TTFT 高约 4.9%。此前 MiniMLX
同配置也出现过约 1800 ms TTFT，因此不能把单次 TTFT 差异归因于实现。综合现有
多次结果，MiniMLX 与 oMLX 端到端处于同一水平，decode 路径有明确优势，但尚不能
宣称所有场景都稳定超过 oMLX。

本轮优化结论：原计划中的可验证优化项已完成；没有新的稳定收益证据时，不再继续
修改当前推理热路径，转入下一阶段工作。

### Context matrix baseline

MiniMLX 单进程矩阵（Qwen3.5-0.8B-MLX-4bit，warmup=32，repeats=5，median，
generated=128，prefill step=1024）已在交互式 Metal 环境完成：

| Context | TTFT | TPOT | TG TPS | E2E |
|---:|---:|---:|---:|---:|
| 1K | 445.5 ms | 5.50 ms | 181.70 | 1149.9 ms |
| 2K | 882.7 ms | 5.66 ms | 176.77 | 1606.8 ms |
| 4K | 1866.2 ms | 6.10 ms | 164.06 | 2646.4 ms |
| 8K | 4059.6 ms | 6.71 ms | 148.95 | 4919.0 ms |
| 16K | 9186.2 ms | 7.81 ms | 128.05 | 10185.8 ms |

对应的 oMLX 单请求结果（用户提供的吞吐量基准截图）为：1K 465.9/6.7/149.9、
4K 1824.3/7.0/144.7、8K 3893.4/7.4/136.5、16K 8937.8/8.6/117.4
（分别为 TTFT/TPOT/TG TPS，oMLX 的 E2E 在界面中按 0.1 秒显示）。对齐结论是：
MiniMLX 的 TPOT 在四档分别低约 17.9%/12.9%/9.3%/9.2%，TG TPS 高约
21.2%/13.4%/9.1%/9.1%；oMLX 的 TTFT 则低约 4.4%/2.3%/4.1%/2.7%。
两者 E2E 在当前显示精度下接近，MiniMLX 的 decode 优势抵消了部分 prefill 差距，
但不能据此宣称 MiniMLX 在所有上下文长度上端到端稳定超过 oMLX。

公开依据：MLX 的 `mx.compile` 会合并和融合计算图；MLX 社区对 Decode 优化的讨论
也将 per-layer `mx.compile`、dispatch reduction 列为通用方向：
<https://ml-explore.github.io/mlx/build/html/usage/compile.html>
和 <https://github.com/ml-explore/mlx/issues/3789>。

KV cache 方向的公开依据：MLX-LM 已提供 `--kv-bits`、`--kv-group-size` 和
`--quantized-kv-start`；MLX 官方文档也将预分配和 cache layout 作为 decode 性能
关键点，见 <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/cache_prompt.py>、
<https://ml-explore.github.io/mlx/build/html/usage/kv_cache.html>。
上述公开依据对应的实验也已经完成；它们不能作为下一项的隐含指令。新的优化方向必须
单独登记为新的计划项，并重新指定目标、适用场景和 A/B 验收标准。

### 1. FA-256 full-attention prefill（已完成阶段性否决）

针对 Qwen3.5/3.6 的 `head_dim=256` causal prefill，优先使用 MLX 原生 `force_fused=True`；只有官方 fused 路径不足时才继续自写 Steel/FlashAttention kernel。当前 0.8B 模型有 6 个 full-attention 层。

### 2. Q4 prefill native QMM tile（阶段性否决）

针对长 prefill 的 Q4 projection，研究面向序列维度的专用量化矩阵乘法 tile；decode 继续使用现有路径。第一版独立 kernel 已证明明显慢于 MLX 原生 QMM，暂不进入生产路径。

### 3. QxA8 prefill

研究 INT8 activation × Q4 weight 的 prefill kernel。在 M4/M5 等硬件上分别检测能力，验证速度、误差和模型兼容性。

### 4. 阶段级 ANE GDN/MLP offload

将当前单投影 ANE 探针升级为减少边界转换的 GDN/MLP 阶段级 offload，研究共享边界、投影组合、activation 融合、双 ANE 和 fraction 调度。

### 5. 长上下文 O(L) attention

研究 head_dim=256 长上下文的 tiled O(L) attention，用于 16K/32K+，同时降低 O(L²) 临时内存压力。

### 6. 自适应 prefill 调度

建立 context length × hardware × chunk size × backend 的测量表，自动选择 prefill chunk、tail padding 和 accelerator dispatch。

## 当前状态

- [x] ANE 单投影可行性和端到端边界成本已验证
- [x] GDN Prefill custom kernel 已完成正确性和完整路径验证；当前默认路径保留，
  不再把 block size 微调当作新的工作项
- [x] FA-256 full-attention prefill 研究：已确认模型形状匹配，完成原生 `force_fused` 隔离 A/B、真实模型命中验证，以及 stock/fused 多 chunk 对照；孤立算子更快但整机所有已测 chunk 均未超过默认 stock 1024，保持 opt-in，不进入默认路径
- [x] Q4 prefill dispatch: BF16 dequantized dense MLP is integrated behind strict inference/shape gates; 1K/4K/16K/32K enabled-vs-disabled TTFT is 446.5/448.0 ms, 1808.2/1861.4 ms, 9118.0/9392.3 ms, and 23786.8/23944.7 ms. Native QMM tile first experiment is rejected (4K gate/up ~72 ms vs stock ~14 ms).
- [x] QxA8 capability research: current MLX exposes no reusable W4A8 matmul entry; custom kernel remains conditional on hardware proof
- [x] Decode fixed-graph compilation: explicit token/cache-state graph compiles and correctness passes, but 6.405 ms vs 6.401 ms has no meaningful gain; not integrated.
- [x] Full-Attention decode SDPA dispatch: cached one-token force-fused SDPA is correct and isolated 1K/4K/16K/32K is 1.2%–4.5% faster, but paired end-to-end A/B is only TPOT 5.90 vs 5.91 ms; not integrated by default.
- [x] Full-Attention decode stage profile: Q/K/V projections and normalization/layout are approximately fixed-cost; real KVCache expansion is boundary-dependent while steady writes stay roughly flat, and SDPA is the length-scaling stage.
- [x] Full-Attention KVCache growth policy: 256/512/1024/2048 isolated A/B shows 256 best at 1K/4K/16K and 1024 best at 32K; production 32K paired runs show TPOT 10.03 vs 10.07 ms and TG TPS 99.67 vs 99.27, with matching token correctness. Runtime now auto-selects 256 below 32K and 1024 at/above 32K.
- [x] Long-context Full-Attention prefill force-fused: isolated SDPA is 1.7x/1.8x faster at 4K/16K and stock 32K standalone SDPA exceeds the Metal buffer limit, but the real 1024-chunk production path shows no stable end-to-end gain; retained opt-in only.
- [x] Prefill allocator cache policy: isolated 16K/32K runs show `mx.clear_cache()` faster than retaining the allocator cache (10.56 vs 11.42 s; 24.69 vs 27.60 s), and the 32K production run with retention regressed to 25.75 s; default clearing remains enabled.
- [x] KV cache quantization: 4K/16K/32K cache bytes drop 34%/43%/45%, but decode regresses 7.3%/5.7%/37.0%; no default performance integration.
- [x] Decode sampling/postprocess: greedy skip-logsumexp is numerically correct and slightly faster in isolation, but the custom synchronous loop regresses end-to-end TPOT/E2E; default disabled.
- [x] 阶段级 ANE GDN/MLP offload feasibility: isolated projection and recurrence/consumer prototypes are numerically correct, but the Core ML `while_loop` scales poorly (1024-token recurrence 337.6 ms; full consumer 1099.8 ms) and full-model dispatch regresses; no production enablement
- [x] 长上下文精确 tiled attention：scalar/query-tile 已在 4K/8K 否决；独立 SIMD-group MMA 原型已验证网格映射，但矩阵 lane 布局仍不正确，未进入生产
- [x] 自适应 prefill 调度（阶段性否决）：已加入显式 opt-in 的 `--prefill-step-size 0`，但当前仅按长度选择 1024/2048/4096，没有证明优于显式 chunk；4K 实测也未带来收益，因此不启用默认路径。后续若有硬件/后端测量表，再以独立实验重新评估。

### 7. Prefix/KV cache reuse

已完成首轮验证并接入 opt-in runtime：4K 扩展请求约 6.4x，16K 扩展请求约
28.3x；缓存状态采用显式克隆，保留冷请求路径。缓存不是冻结；当前只延后长上下文
SSD/paged cache。命中/未命中遥测和淘汰策略属于后续工程项，不插入当前性能优化循环。
