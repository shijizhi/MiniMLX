# MiniMLX / oMLX 模型综合基准

本文档是模型支持阶段的长期性能记录。每个模型完成测试后追加一节，保留
MiniMLX 和 oMLX 的原始指标，不用单一吞吐量替代 TTFT、TPOT 和内存信息。

## 固定测试协议

- 硬件按每次运行单独记录；历史数据未记录的，不从主机名或其他运行推断。
- 请求：单请求、代码上下文、生成 128 tokens。
- Warm-up：32；正式重复：5；统计：median，并保留全部原始样本。
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
| Qwen3.5-9B | MLX affine Q4, group 64 | PASS | 已测 | 已测 | 已测 | 已测 | MiniMLX-only 三次中位数；1K/16K 提前 EOS，硬件信息待确认 |
| Qwen3.5-27B | MLX affine Q4, group 64 | PASS | 已测 | 已测 | 已测 | 已测 | 已记录 MiniMLX 五次中位数与用户提供的 oMLX 截图；对照条件未完全对齐 |
| Llama 3.2 3B Instruct | MLX affine Q4, group 64 | PASS | 已测 | 已测 | 已测 | 已测 | oMLX 单请求时延在 4K/8K/16K 较低；MiniMLX 1K TTFT 较低；批处理 TPS 未对齐 |
| Mistral Nemo Instruct 2407 | MLX affine Q4, group 64 | PASS | 已测 | 已测 | 已测 | 已测 | MiniMLX 通用 MLX-LM 基线，未与 oMLX 对照 |

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

## Qwen3.5-9B-4bit (MiniMLX-only)

- 下载来源：`mlx-community/Qwen3.5-9B-4bit`
- 格式：MLX affine Q4，group size 64
- 架构：`Qwen3_5ForConditionalGeneration`，32 layers（24 linear-attention、8 full-attention）
- 参数量：descriptor 未提供；不从 repo 名称推断
- 兼容性：文本生成 PASS；单图 OCR PASS（输出 `Swift / Transformers`）
- 基准：MiniMLX `benchmark_context_matrix.py`；warm-up 32、max tokens 128、step 2048、每 context 3 次；仅自有 MiniMLX 结果，本条没有运行 oMLX 对照测试。
- 整体峰值内存：7.680 GiB。运行机器未由本次日志独立确认。
- 注意：1K 实际生成 29 tokens，16K 实际生成 16 tokens；这两行的 TG TPS/E2E 受提前停止影响，保留实测数字但不视为满 128-token 基准。

| Context | TTFT median (ms) | TPOT median (ms) | PP TPS median | TG TPS median | Generated | E2E median (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 620.7 | 18.14 | 1,649.8 | 57.11 | 29/128 | 1,429.6 |
| 4K | 2,387.1 | 18.55 | 1,715.9 | 54.34 | 128/128 | 5,035.6 |
| 8K | 4,835.0 | 19.12 | 1,694.3 | 52.72 | 128/128 | 7,584.4 |
| 16K | 10,321.5 | 20.34 | 1,587.4 | 52.43 | 16/128 | 10,944.0 |

每个上下文的三次原始样本、运行命令和协议记录在
[`Qwen35_Family_Coverage.md`](Qwen35_Family_Coverage.md)。

## Qwen3.5-27B-4bit (MiniMLX result + user-supplied oMLX reference)

- 下载来源：`mlx-community/Qwen3.5-27B-4bit`
- 缓存记录 revision：`45797d2985a12c55e6473686e9ea91b95e959553`
- 格式/架构：MLX affine Q4，group size 64；`Qwen3_5ForConditionalGeneration`，Dense，64 layers；3 个权重分片均在 index 中且文件齐全。
- 环境：Apple M5 Pro，64 GiB unified memory；Python 3.13.12、MLX 0.32.2、mlx-lm 0.31.3、mlx-vlm 0.7.2；MiniMLX revision `dbad46b`，工作区包含其他未提交修改。
- Text smoke：PASS，4/4 tokens；smoke peak 16.830 GiB。
- Vision smoke：PASS；`Benchmarks/fixtures/swift-t-banner.png` 两次均输出 `Swift / Transformers`，每次 4,155 prompt tokens / 5 completion tokens。视觉缓存由 miss/store (1) 变为 hit (1)，保持 1 条 entry。
- 基准：MiniMLX `benchmark_context_matrix.py`；warm-up 32、max tokens 128、step 1024、每 context 5 次；MiniMLX 数值为自有测试，没有运行 oMLX 基准。Suite peak memory 18.465 GiB。
- 1K 自然提前 EOS，生成 111/128；4K/8K/16K 均生成 128/128。1K 的 TG TPS/E2E 保留为观测，不当成满长 decode 比较。

| Context | TTFT median (ms) | TPOT median (ms) | PP TPS median | TG TPS median | Generated | E2E median (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 2,182.0 | 57.23 | 469.3 | 17.63 | 111/128 | 8,813.1 |
| 4K | 9,161.0 | 58.31 | 447.1 | 17.28 | 128/128 | 16,875.4 |
| 8K | 22,013.1 | 69.24 | 372.1 | 14.56 | 128/128 | 31,245.3 |
| 16K | 42,153.1 | 63.91 | 388.7 | 15.77 | 128/128 | 50,584.8 |

五次原始样本（每项顺序 TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms）和精确运行命令见
[`Qwen35_Family_Coverage.md`](Qwen35_Family_Coverage.md)。

### 用户提供的 oMLX 对照截图

以下 oMLX 数值由用户提供的 GUI 截图转录（页面标记“单请求结果”“试验：4”）。这不是 MiniMLX 运行得到的 oMLX 基准；截图没有完整说明重复统计、warm-up、prefill step 和硬件/版本，因此作为参考，不视为严格同条件结论。MiniMLX 五次中位数见本节前表；峰值内存是 MiniMLX 全 suite peak，非逐上下文值。

为贴近截图中的单请求形式，以下主对照采用 MiniMLX 随后追加的一次单次矩阵结果（每个 context 各 1 次），不再用五次中位数充当单次对照。原五次中位数仍保留在上表作为稳定性参考。单次运行命令与结果见 [`Qwen35_Family_Coverage.md`](Qwen35_Family_Coverage.md)。

| Context | Runtime | TTFT ms | TPOT ms | PP TPS | TG TPS | E2E s | Peak memory |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1K | MiniMLX (single run) | 2,176.0 | 57.20 | 470.6 | 17.64 | 8.79* | 18.464 GiB suite peak |
| 1K | oMLX screenshot | 2,118.6 | 58.2 | 483.3 | 17.3 | 9.5 | 15.9 GB |
| 4K | MiniMLX (single run) | 8,378.5 | 58.04 | 488.9 | 17.36 | 16.06 | 18.464 GiB suite peak |
| 4K | oMLX screenshot | 7,969.5 | 57.8 | 514.0 | 17.4 | 15.3 | 17.0 GB |
| 8K | MiniMLX (single run) | 17,513.0 | 59.28 | 467.8 | 17.00 | 25.35 | 18.464 GiB suite peak |
| 8K | oMLX screenshot | 16,638.3 | 61.3 | 492.4 | 16.5 | 24.4 | 17.3 GB |
| 16K | MiniMLX (single run) | 37,967.9 | 61.78 | 431.5 | 16.31 | 46.13 | 18.464 GiB suite peak |
| 16K | oMLX screenshot | 36,426.9 | 60.4 | 449.8 | 16.7 | 44.1 | 17.8 GB |

方向性上，截图中的 oMLX 在四个上下文的 TTFT、PP TPS 都更好，4K/8K/16K 的 E2E 也较低。TPOT/TG TPS 则是混合结果且差距较小：MiniMLX 在 1K/8K 的 TPOT 和 TG TPS 较好，oMLX 在 4K/16K 较好。MiniMLX 1K 实际生成 111/128，而截图未显示 oMLX 的实际生成数，所以该行 E2E/TG 不宜直接比较。单次样本显示的性能差距小于先前五次中位数表给人的印象；仍需统一设备、版本、warm-up、重复次数、step 和生成 token 数后才能下严格结论。峰值内存口径及 GB/GiB 也未对齐，不做严格胜负判断。

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

## Llama 3.2 3B Instruct MLX 4-bit

测试设备为同一台 MacBook Pro M5 Pro、64 GB。MiniMLX 使用
`mlx-community/Llama-3.2-3B-Instruct-4bit` 的本地副本；OMLX 使用刚复制到
`~/.omlx/models/mlx-community/Llama-3.2-3B-Instruct-4bit` 的副本。两边截图/运行
均为 Python 代码语料、1K/4K/8K/16K contexts、32-token quick warm-up 和 128-token
生成。MiniMLX 为五次中位数；用户提供的 OMLX 截图标记“试验：4”，但其聚合统计方式
未显示，故这些数字是同机参考对照，不是严格同统计口径的胜负结论。

| Context | Mini TTFT ms | OMLX TTFT ms | Mini TPOT ms | OMLX TPOT ms | Mini PP TPS | OMLX PP TPS | Mini TG TPS | OMLX TG TPS | Mini E2E s | OMLX E2E s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 259.9 | 319.1 | 8.88 | 7.9 | 3940.3 | 3209.2 | 113.52 | 127.9 | 1.692 | 1.3 |
| 4K | 1083.2 | 1043.5 | 10.42 | 9.3 | 3781.2 | 3925.1 | 96.74 | 108.8 | 2.709 | 2.2 |
| 8K | 2289.6 | 2109.1 | 11.29 | 11.2 | 3577.9 | 3884.2 | 89.31 | 89.9 | 3.988 | 3.5 |
| 16K | 5998.9 | 4985.7 | 16.63 | 14.9 | 2731.2 | 3286.2 | 60.62 | 67.8 | 8.395 | 6.9 |

### 结论

- **TTFT / prefill：** MiniMLX 在 1K 快约 18.6%；OMLX 在 4K、8K、16K 分别快约
  3.8%、8.6%、20.3%。16K 的差距最明显。
- **TPOT / decode：** OMLX 四个 context 的 TPOT 均较低；差距在 1K、4K、8K、16K
  分别约为 12.4%、12.0%、0.8%、11.6%。8K 基本持平。
- **PP/TG TPS：** PP TPS 两边交错：MiniMLX 1K 较高，OMLX 4K/8K/16K 较高。单请求
  TG TPS OMLX 略高，8K 基本持平。
- **E2E：** 按截图显示的一位小数，OMLX 四档均较短；由于 OMLX 只显示到 0.1 秒且
  重复统计未说明，差距只作方向性参考。
- **内存：** OMLX 截图显示 2.46/2.75/3.19/3.90 GB；MiniMLX 显示
  2.460/2.691/3.181/3.986 GiB。数值接近，但单位标签/换算方式不一致，不能据此
  宣称精确内存胜负。
- **连续批处理吞吐：** OMLX 页面描述该 TPS 为 continuous batching，并选择 batch
  size 2 和 4；MiniMLX 本次报告是单请求自回归矩阵，没有同配置的 continuous
  batching 结果。截图的吞吐量 868.0/1896.7/2350.6/2400.3 不与 MiniMLX TG TPS
  或由单请求 E2E 算出的速率直接比较。

整体看，当前数据表明 MiniMLX 在 1K prefill 有优势；4K 以上 OMLX 的 prefill、decode
和 E2E 更好，尤其 16K。**Llama 暂时保持通用 MLX-LM 基线，不据此直接新增 Llama
优化层。** 先确认两边的重复统计、采样和吞吐口径后再做严格结论；若要对比连续批处理，
需要 MiniMLX 也跑 batch size 2/4 的同协议测试。

来源：MiniMLX 五次 context matrix 见 [`Model_Support_Matrix.md`](Model_Support_Matrix.md)
的 Llama 记录；oMLX 指标来自用户提供的吞吐量基准截图，试验次数标注为 4。

### Llama 性能专项复测（2026-10-25）

为评估此次优化候选，在 M5 Pro 64 GB 上按相同 MiniMLX 路径重建了 step=1024 的五次
中位数基线。所有请求实际生成 128 tokens；使用标准 `mlx-lm`、Q4 affine/group64，关闭
KV 量化，未启用 Qwen 专属优化。

| Context | TTFT | TPOT | PP TPS | TG TPS | E2E | Peak MLX memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 236.0 ms | 7.92 ms | 4,339.0 | 127.25 | 1,513.9 ms | 2.460 GiB |
| 4K | 982.7 ms | 9.44 ms | 4,167.9 | 106.82 | 2,455.6 ms | 2.691 GiB |
| 8K | 2,180.4 ms | 11.23 ms | 3,757.2 | 89.77 | 3,885.6 ms | 3.181 GiB |
| 16K | 5,645.8 ms | 14.84 ms | 2,902.0 | 67.91 | 7,811.4 ms | 3.986 GiB |

与上方 oMLX 截图对照，当前 MiniMLX 在 1K/4K 的 TTFT 更低，8K 接近，16K 仍较高；
四档 TPOT 基本相当。oMLX 的重复统计、实际 prefill step 和环境负载不可见，故该对照
仅供定位方向，不能视作同条件胜负。此次记录替代旧 MiniMLX 样本用于后续优化决策，旧值
保留为历史基准。

另外对 1024/2048 prefill step 做了平衡顺序五次 A/B。4K 下 2048 的 TTFT/E2E 中位数
分别改善 2.43%/1.21%；8K 改善 2.61%/1.35%；token 输出完全一致。16K 没有稳定改善。
应用标准生成默认 step 已为 2048，因此无需新增 Llama 叠加实现。显式 step=2048 的另一轮
整套矩阵出现显著异常慢速与离散样本，该轮不作为有效基线。

Q4 MLP BF16 预解量化和 gate/up 融合 A/B 结果及不纳入生产路径的原因见
[`Optimization_Roadmap.md`](Optimization_Roadmap.md) 的 Llama 专项记录。

## Mistral Nemo Instruct 2407 MLX 4-bit（MiniMLX）

- Checkpoint：`mlx-community/Mistral-Nemo-Instruct-2407-4bit`；通用 `StandardTextAdapter` / `mlx-lm-standard-runtime`，未启用模型专属优化。
- 环境：Apple M5 Pro、64 GB unified memory；通过交互式 macOS Metal 会话运行。
- 协议：代码语料，warm-up 32 tokens，输出上限 128，五次中位数，prefill step 1024，EOS 开启；所有样本均生成 128/128 tokens。
- 运行配置：Q4 native prefill、native FA256 开启；Q4 dense prefill 与 Q4 MLP fusion 关闭；KV cache 未量化；autoregressive decode。套件峰值内存 9.469 GiB。
- Tokenizer：启用 `fix_mistral_regex=True`，修复该 checkpoint 已知的正则兼容问题。

| Context | TTFT median (ms) | TPOT median (ms) | PP TPS median | TG TPS median | Generated | E2E median (ms) | Peak MLX memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1,253.0 | 38.42 | 817.3 | 26.23 | 128/128 | 6,442.1 | 7.290 GiB |
| 4K | 5,222.1 | 41.58 | 784.4 | 24.24 | 128/128 | 10,830.7 | 7.592 GiB |
| 8K | 10,885.2 | 45.34 | 752.6 | 22.23 | 128/128 | 16,965.1 | 8.264 GiB |
| 16K | 24,561.7 | 53.60 | 667.1 | 18.80 | 128/128 | 31,706.4 | 9.469 GiB |

五次原始样本（每项顺序 TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms）：

- 1K：`1248.1/38.42/820.5/26.23/6456.2`；`1253.0/38.27/817.3/26.34/6424.9`；`1254.0/38.35/816.6/26.28/6442.1`；`1256.7/38.48/814.8/26.19/6476.1`；`1245.3/38.43/822.3/26.23/6438.4`。
- 4K：`5064.0/41.65/808.9/24.20/10663.0`；`5232.6/41.51/782.8/24.28/10830.7`；`5272.2/41.58/776.9/24.24/10885.1`；`5222.1/41.69/784.4/24.17/10846.9`；`5197.5/41.43/788.1/24.33/10782.6`。
- 8K：`10957.6/45.51/747.6/22.15/17059.8`；`10885.2/45.31/752.6/22.24/16965.1`；`10872.6/45.34/753.5/22.23/16957.7`；`10867.1/45.34/753.8/22.23/16955.2`；`10930.2/45.39/749.5/22.20/17022.1`。
- 16K：`24347.7/53.32/672.9/18.90/31440.4`；`24468.9/53.40/669.6/18.87/31584.6`；`24568.6/53.60/666.9/18.80/31706.4`；`24561.7/53.64/667.1/18.79/31715.4`；`24699.3/53.61/663.3/18.80/31840.0`。

短测通过，精确输出 `Mistral test passed`（6 tokens）。结果支持将该 checkpoint 归入通用 MLX-LM 基线的已测文本模型。尚无相同 checkpoint 的 oMLX 对照，因此此处不作跨运行时速度结论。详细兼容性记录见 [`Model_Support_Matrix.md`](Model_Support_Matrix.md) 的 Mistral Nemo 条目。

命令：`PYTHONPATH=python .venv/bin/python tools/benchmark_context_matrix.py <model-dir> Resources/BenchmarkCorpus/code_python.txt --warmup 32 --max-tokens 128 --repeats 5 --step 1024 --contexts 1024 4096 8192 16384`。

### Mistral 单次复测（对照 oMLX 截图）

按要求每档只测一次，仍使用 32-token warm-up、128-token 输出、step 1024、代码语料，EOS 开启；每档生成均为 128/128。下表 MiniMLX 为本次单次结果，oMLX 为用户截图中标注“试验：4”的结果，重复统计方式不明，差异仅作参考。

| Context | TTFT Mini / oMLX (ms) | TPOT Mini / oMLX (ms) | PP TPS Mini / oMLX | TG TPS Mini / oMLX | E2E Mini / oMLX (s) |
|---:|---:|---:|---:|---:|---:|
| 1K | 1,085 / 881.3 | 32.09 / 24.9 | 943.8 / 1,161.9 | 31.41 / 40.4 | 5.44 / 4.1 |
| 4K | 4,358 / 3,484.9 | 34.59 / 27.0 | 939.8 / 1,175.4 | 29.14 / 37.3 | 9.03 / 6.9 |
| 8K | 9,152 / 7,281.6 | 37.85 / 29.8 | 895.1 / 1,125.0 | 26.63 / 33.8 | 14.25 / 11.1 |
| 16K | 21,344 / 16,276.7 | 44.60 / 35.1 | 767.6 / 1,006.6 | 22.60 / 28.7 | 27.30 / 20.7 |

此单次复测中 MiniMLX 的 E2E 比截图 oMLX 高约 28%–33%，低于此前五次中位数对照所得约 53%–57% 的差距；单样本波动明显改变结论幅度，因此不能把先前差距视作稳定性能差距。后续需统一两边的单次/重复统计和 warm-up 后再下优化结论。MiniMLX 原始单次结果：`/tmp/minimlx-mistral-single-bench.log`。
