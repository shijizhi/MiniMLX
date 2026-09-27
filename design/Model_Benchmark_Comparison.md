# MiniMLX / oMLX 模型综合基准

本文档是模型支持阶段的长期性能记录。每个模型完成测试后追加一节，保留
MiniMLX 和 oMLX 的原始指标，不用单一吞吐量替代 TTFT、TPOT 和内存信息。

## 固定测试协议

- 硬件：MacBook Air M4，24 GB 统一内存。
- 请求：单请求、代码上下文、生成 128 tokens。
- Warm-up：32；正式重复：5；统计：median。
- MiniMLX：prefill step size = 1024。
- 指标：TTFT、TPOT、PP TPS、TG TPS、E2E、峰值内存。
- 标准上下文：1K、4K、8K、16K；32K 仅在内存安全时测试。
- oMLX 测试必须使用相同模型、相同上下文、相同输出长度和相同 warm-up 约定。
- 如果两边的测试条件不完全一致，结果只能标记为“参考”，不能用于性能结论。

## 结果解释

- TTFT 越低越好，主要反映 prefill。
- TPOT 越低越好，主要反映 decode 单 token 延迟。
- PP TPS 越高越好，等价于 prefill 吞吐。
- TG TPS 越高越好，等价于 decode 吞吐。
- E2E 受 TTFT、TPOT 和生成长度共同影响。
- “MiniMLX 领先”必须至少在同一指标的多个上下文长度上稳定出现，不能根据一次运行判断。
- 性能专项只有在 oMLX 有模型特异化处理，或对齐测试出现稳定明显差距时才建立。

## 汇总表

| 模型 | 格式 | 兼容性 | 1K | 4K | 8K | 16K | 性能结论 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | Q4 | PASS | 已测 | 已测 | 已测 | 已测 | MiniMLX decode 优势，prefill 接近 |
| Qwen3.5-0.8B | BF16 | PASS | 已测 | 已测 | 已测 | 已测 | MiniMLX TTFT/TPOT/TG TPS 均领先 |
| Qwen3.5-4B | BF16 | PASS | 已测 | 已测 | 已测 | 已测 | 当前数据 MiniMLX TTFT/TPOT/E2E 更快；需留意 oMLX 批处理设置 |

## Qwen3.5-0.8B-MLX-4bit

### MiniMLX

来源：MiniMLX `benchmark_context_matrix.py`，用户提供的终端结果。

| Context | TTFT (ms) | TPOT (ms) | TG TPS | E2E (ms) |
|---:|---:|---:|---:|---:|
| 1K | 445.5 | 5.50 | 181.70 | 1149.9 |
| 2K | 882.7 | 5.66 | 176.77 | 1606.8 |
| 4K | 1866.2 | 6.10 | 164.06 | 2646.4 |
| 8K | 4059.6 | 6.71 | 148.95 | 4919.0 |
| 16K | 9186.2 | 7.81 | 128.05 | 10185.8 |

### oMLX

来源：用户提供的 oMLX 基准界面结果。界面只显示了部分 E2E，故这里只记录可对齐指标。

| Context | TTFT (ms) | TPOT (ms) | TG TPS |
|---:|---:|---:|---:|
| 1K | 465.9 | 6.7 | 149.9 |
| 4K | 1824.3 | 7.0 | 144.7 |
| 8K | 3893.4 | 7.4 | 136.5 |
| 16K | 8937.8 | 8.6 | 117.4 |

### 对比结论

| Context | MiniMLX TTFT 优势 | MiniMLX TPOT 优势 | MiniMLX TG TPS 优势 |
|---:|---:|---:|---:|
| 1K | 4.4% | 17.9% | 21.2% |
| 4K | -2.3% | 12.9% | 13.4% |
| 8K | -4.1% | 9.3% | 9.1% |
| 16K | -2.7% | 9.2% | 9.1% |

结论：MiniMLX 的 decode 优势稳定存在；prefill 在当前协议下与 oMLX 接近，不能仅凭这组数据宣称所有端到端场景都稳定领先。

## Qwen3.5-0.8B-MLX-bf16

### MiniMLX

来源：MiniMLX `benchmark_context_matrix.py`，用户提供的终端结果。

| Context | TTFT (ms) | TPOT (ms) | TG TPS | E2E (ms) |
|---:|---:|---:|---:|---:|
| 1K | 448.8 | 16.68 | 59.97 | 2583.3 |
| 2K | 871.4 | 16.75 | 59.70 | 3015.6 |
| 4K | 1752.8 | 17.10 | 58.50 | 3941.0 |
| 8K | 3829.2 | 17.54 | 57.02 | 6074.0 |
| 16K | 8939.7 | 18.73 | 53.40 | 11336.9 |

### oMLX

来源：用户提供的 oMLX 基准界面结果。

| Context | TTFT (ms) | TPOT (ms) | TG TPS |
|---:|---:|---:|---:|
| 1K | 501.4 | 18.7 | 53.8 |
| 4K | 2506.1 | 19.0 | 53.1 |
| 8K | 4798.6 | 19.7 | 51.2 |
| 16K | 10992.6 | 21.2 | 47.5 |

### 对比结论

| Context | MiniMLX TTFT 优势 | MiniMLX TPOT 优势 | MiniMLX TG TPS 优势 |
|---:|---:|---:|---:|
| 1K | 10.5% | 10.8% | 11.5% |
| 4K | 30.0% | 10.0% | 10.2% |
| 8K | 20.2% | 11.0% | 11.4% |
| 16K | 18.7% | 11.7% | 12.4% |

结论：BF16 在当前对齐结果中 MiniMLX 的 TTFT、TPOT 和 TG TPS 均领先 oMLX。

## Qwen3.5-4B-MLX-4bit

### MiniMLX compatibility

- Architecture: `Qwen3_5ForConditionalGeneration`
- Layers: 32 total, 24 linear-attention, 8 full-attention
- Quantization: affine Q4, group size 64
- Q4 MLP layers: 32
- Smoke tokens: `[11, 353, 2688, 4313]`
- Text prefix: `", I'm trying"`
- Prefill: 281.376 ms
- Decode: 86.879 ms
- Result: `compatibility=PASS`

### oMLX 基线

来源：用户提供的 oMLX 基准界面结果；warm-up 32、生成 128 tokens。

| Context | TTFT (ms) | TPOT (ms) | PP TPS | TG TPS | E2E (s) | Peak memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 2914.7 | 29.1 | 351.3 | 34.7 | 6.6 | 3.29 GB |
| 4K | 11439.9 | 30.5 | 358.0 | 33.0 | 15.3 | 3.75 GB |
| 8K | 23763.2 | 32.7 | 344.7 | 30.8 | 28.0 | 3.98 GB |
| 16K | 48617.0 | 33.5 | 337.0 | 30.1 | 52.9 | 4.38 GB |

### oMLX 连续批处理基线

来源：用户提供的 oMLX 批次结果；同一模型，批次 1x/2x/4x。

| Batch | TG TPS | PP TPS | 平均 TTFT (ms) | E2E (s) | 相对加速 |
|---:|---:|---:|---:|---:|---:|
| 1x | 34.7 | 351.3 | 2914.7 | 6.6 | 1.00x |
| 2x | 75.1 | 216.3 | 6321.1 | 12.9 | 2.16x |
| 4x | 150.9 | 183.5 | 12659.3 | 25.7 | 4.35x |

### MiniMLX

| Context | TTFT (ms) | TPOT (ms) | TG TPS | E2E (s) |
|---:|---:|---:|---:|---:|
| 1K | 2986.9 | 28.20 | 35.46 | 6.596 |
| 2K | 5842.0 | 27.12 | 36.87 | 9.314 |
| 4K | 12806.2 | 27.27 | 36.67 | 16.296 |
| 8K | 29107.0 | 29.89 | 33.46 | 32.933 |
| 16K | 64705.6 | 35.43 | 28.22 | 69.241 |

连续批处理结果待补充。单请求结果和批处理结果分开比较，不混合计算。

### 单请求对比

| Context | TTFT 差异 | TPOT 差异 | TG TPS 差异 | E2E 差异 |
|---:|---:|---:|---:|---:|
| 1K | MiniMLX 慢 2.5% | MiniMLX 快 3.1% | MiniMLX 快 2.2% | 基本持平 |
| 4K | MiniMLX 慢 11.9% | MiniMLX 快 10.6% | MiniMLX 快 11.1% | MiniMLX 慢 6.5% |
| 8K | MiniMLX 慢 22.5% | MiniMLX 快 8.6% | MiniMLX 快 8.6% | MiniMLX 慢 17.6% |
| 16K | MiniMLX 慢 33.1% | MiniMLX 慢 5.8% | MiniMLX 慢 6.2% | MiniMLX 慢 30.9% |

### 当前结论

兼容性通过，但性能不能判定为通过：MiniMLX 在 1K～8K 的 decode 略快，然而
prefill 从 4K 开始明显落后，16K 的 TTFT 落后 33.1%，E2E 落后 30.9%。
该模型进入性能专项，不继续下载下一个模型。

下一步先核对 oMLX 是否为该 Qwen3.5 变体启用了 ANE prefill、QxA8 prefill、
custom kernel 或其他专用 dispatch，再决定 MiniMLX 的独立实验方向。

### 专项调查记录：Q4 prefill projection coverage

公开源码核对后，当前最值得验证的差异不是先假设 ANE，而是 Q4 projection 的
覆盖范围：

- oMLX 的 Qwen3.5 Q4 patch 对长 prefill 的 MLP gate/up/down 使用 native QMM；
- 同一 patch 还覆盖 full-attention 的 q/k/v/o projection，以及 GDN 的
  `in_proj_qkv`、`in_proj_z`、`in_proj_a`、`in_proj_b`、`out_proj`；
- oMLX 的默认长 prefill gate 通常从 2048 tokens 开始，decode/短输入回退到
  原始路径；
- MiniMLX 当前已集成 Q4 MLP fusion/dense 研究路径，但没有等价覆盖所有
  attention/GDN Q4 projection 的独立 native backend。

因此，4B 当前的首要假设是“Q4 projection coverage gap”，不是“必须启用 ANE”。
这只是研究假设，必须按以下顺序验证：

1. 对 MiniMLX 4B 在 4K/16K 做 real prefill stage profile，确认 projection、
   MLP、GDN 和 full-attention 各自占比；
2. 分别关闭 MiniMLX 的 Q4 MLP dense/fusion，确认已有路径实际贡献；
3. 选择一个代表性 Q4 projection 做独立 correctness/A-B，验证 native QMM
   backend 是否有足够收益；
4. 再扩展到 attention/GDN projection，并做全模型端到端验证；
5. 最后才检查 ANE。QxA8 暂不作为 M4 的首要方向，因为 oMLX 公开设置将其
   限定为具备原生 INT8 tensor 能力的硬件路径。

### MiniMLX 4K real prefill profile

当前路径：Q4 dense enabled、Q4 MLP fusion enabled、warmup=1。

| 项目 | 时间 |
|---|---:|
| Grouped total | 13527.7 ms |
| Linear attention | 8195.2 ms |
| Full attention | 3403.4 ms |
| Unattributed | 1929.0 ms |
| Actual prefill | 11738.7 ms |
| Actual prefill chunks | 11440.0 ms |
| Actual final token | 298.6 ms |

本次 actual prefill 为 11.74 s，与 oMLX 4K TTFT 11.44 s 接近；此前端到端矩阵
中的 12.81 s 包含完整请求边界和重复运行调度成本，不能直接等同于 grouped profile。
下一项是保持 4K、关闭 Q4 MLP fusion，测量其对 linear/full attention 和 actual
prefill 的真实影响。

### Q4 MLP fusion A/B

| 配置 | Linear attention | Full attention | Actual prefill | Actual chunks |
|---|---:|---:|---:|---:|
| fusion enabled | 8195.2 ms | 3403.4 ms | 11738.7 ms | 11440.0 ms |
| fusion disabled | 8213.3 ms | 3393.3 ms | 11759.8 ms | 11585.8 ms |

补充复测：Q4 dense enabled、Q4 MLP fusion disabled 时，grouped total 为
13767.7 ms，linear attention 为 8260.2 ms，full attention 为 3438.8 ms，
actual prefill 为 12513.3 ms，actual chunks 为 12194.3 ms。由于这次结果与前一轮
fusion disabled 且 dense disabled 的 11809.0 ms 不一致，单次 profile 不足以把
774.6 ms 全部归因于 fusion；需要恢复生产配置后重新建立同条件基线，再进行至少
3 次中位数复测。

关闭 fusion 仅使 actual prefill 增加 21.1 ms，约 0.18%；在当前单次 profile
噪声范围内，不能认为 fusion 是 4B 主要瓶颈。下一项改测 Q4 dense fallback。

### 结论

待 MiniMLX 完成 1K/4K/8K/16K 测试后计算差异。若 MiniMLX 明显落后，先检查
oMLX 是否启用了 Qwen3.5 专用 ANE/QxA8/MTP 路径，再决定是否进入性能专项。

## Qwen3.5-4B-MLX-bf16

### MiniMLX

来源：用户提供的 `benchmark_context_matrix.py` 终端结果。单次测量，2K context
warm-up 生成 32 tokens；测量请求各生成 128 tokens；prefill step 1024。该结果
用于建立 BF16 参考，不属于固定协议的 5 次 median，不能单独用于性能结论。

| Context | TTFT (ms) | TPOT (ms) | TG TPS | E2E (ms) |
|---:|---:|---:|---:|---:|
| 1K | 2574.9 | 84.99 | 11.77 | 13453.9 |
| 4K | 10004.5 | 86.58 | 11.55 | 21087.3 |
| 8K | 20111.8 | 87.69 | 11.40 | 31335.8 |
| 16K | 40419.4 | 91.66 | 10.91 | 52152.2 |

### oMLX

来源：用户提供的 oMLX 基准界面截图。同一模型，warm-up 32、输出 128 tokens。
截图中连续批处理 2/4 处于选中状态；PP/TG TPS 和内存列可能受其影响。

| Context | TTFT (ms) | TPOT (ms) | PP TPS | TG TPS | E2E (s) | Peak memory (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 2738.6 | 94.6 | 373.9 | 10.7 | 14.8 | 8.65 |
| 4K | 11270.3 | 92.4 | 363.4 | 10.9 | 23.0 | 9.27 |
| 8K | 24873.6 | 98.4 | 329.3 | 10.2 | 37.4 | 9.43 |
| 16K | 60125.4 | 99.7 | 272.5 | 10.1 | 72.8 | 9.85 |

### 对照（单次结果）

| Context | TTFT Mini / oMLX (ms) | TPOT Mini / oMLX (ms) | TG TPS Mini / oMLX | E2E Mini / oMLX (s) |
|---:|---:|---:|---:|---:|
| 1K | 2574.9 / 2738.6 | 84.99 / 94.6 | 11.77 / 10.7 | 13.45 / 14.8 |
| 4K | 10004.5 / 11270.3 | 86.58 / 92.4 | 11.55 / 10.9 | 21.09 / 23.0 |
| 8K | 20111.8 / 24873.6 | 87.69 / 98.4 | 11.40 / 10.2 | 31.34 / 37.4 |
| 16K | 40419.4 / 60125.4 | 91.66 / 99.7 | 10.91 / 10.1 | 52.15 / 72.8 |

按表面数值，MiniMLX 的 TTFT 低约 6%/11%/19%/33%，TPOT 低约
6%–10%，E2E 低约 8%/8%/16%/28%。这仍是单次对单次的参考对照；oMLX 截图中
2/4 批次选项被选中，而 MiniMLX 是单请求矩阵，不能据此下稳定性能结论。
MiniMLX 本轮没有 PP TPS 或各 context 的峰值内存数据，不能比较这两项。

### 结论

MiniMLX 完成单次 1K/4K/8K/16K 测量，表面上全项优于截图所列 oMLX 单请求指标；
但批次设置未完全对齐，且双方都不是多次中位数。结论仅作初步参考，不触发新性能专项。

## 新模型记录模板

复制下面小节，替换模型信息和结果；测试完成后再删除本地模型权重。

```markdown
## <model-id>

- 下载来源：<Hugging Face model id>
- 格式：<BF16/FP16/Q4/Q8/...>
- 架构：<model_type / architecture>
- 模型规模：<parameter count>
- 峰值内存：<MiniMLX / oMLX>
- MiniMLX 兼容性：<PASS / limited-context / hardware-blocked / unsupported>
- oMLX 特殊处理：<none / feature names>
- 触发专项优化：<yes/no，原因>

### MiniMLX

| Context | TTFT | TPOT | TG TPS | E2E | Peak memory |
|---:|---:|---:|---:|---:|---:|
| 1K | | | | | |
| 4K | | | | | |
| 8K | | | | | |
| 16K | | | | | |

### oMLX

| Context | TTFT | TPOT | TG TPS | E2E | Peak memory |
|---:|---:|---:|---:|---:|---:|
| 1K | | | | | |
| 4K | | | | | |
| 8K | | | | | |
| 16K | | | | | |

### 结论

<兼容性结论、性能结论、是否进入专项优化>
```
