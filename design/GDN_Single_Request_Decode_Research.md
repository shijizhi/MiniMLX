# GDN 单请求 Decode 优化研究

## 当前结论

当前 MiniMLX 与 oMLX 的差距首先没有显示在 Prefill：同机 Qwen3.8-27B 记录中，
1K 的 MiniMLX Prefill 为 2144.3 ms，oMLX 为 2148.5 ms；4K 分别为 8241.0 ms 和
8061.3 ms，MiniMLX 暂时慢约 2.2%。两组尚非严格配对，不足以做性能结论。

Decode 数据目前不能作端到端比较：MiniMLX 在 1K 请求只生成 23 tokens 后遇到 EOS；
4K 只生成 6 tokens。4K 的 21.13 tok/s 由 6 tokens 得出，时长太短，且输出数不同，
不应用来判断胜负。先前在 GDN 函数内强制同步计时所得的 `sum_call_ms` 也无效；
MLX lazy evaluation 会把上游工作归到这些同步点。现有合法诊断仅为调用计数：4K 这次
生成 6 tokens 记录到 336 个单步 GDN 调用，不代表每次调用的 GPU 耗时。

因此，下一轮聚焦普通单 token 自回归 Decode；MTP 和 APC 不在本优化项内。

## 公开实现证据与可移植点

- [Qwen3.8-27B-MLX](https://github.com/ggluo/Qwen3.8-27B-MLX) 报告针对 gated-delta recurrence
  的融合 Metal kernel，将 recurrent state 更新的多步运算合并，报告 plain decode 约 +4.8%。
  项目规模很小，数值只作为方向线索，不作为 MiniMLX 的预期收益。
- [libmir 的 Qwen3.6 Metal 研究记录](https://github.com/mirmir-tech/libmir/blob/main/AGENTS.md)
  给出了更细的消融：head-wide 单步 recurrence 中直接计算 FP32 decay/update，减少一个
  dispatch 和两个临时张量；其报告在 15 样本中 97.04 tok/s 对 94.33 tok/s，并要求完整
  recurrent state 和 greedy token digest 一致。其后把精确 Q/K RMSNorm 融进 recurrence，
  128-token digest 保持一致，交替测试报告 91.91/92.02 对 90.38/90.34 tok/s。不同阶段
  的基线与硬件/模型配置不能混作同一组收益。
- [Qwen3.6 的可复现 MLX 优化研究](https://github.com/fbzz/qwen3.6-mlx-inference) 也说明
  DeltaNet 单层边界的 QKV 与 recurrence 耗时接近，且多项局部快核没有转化为完整 Decode
  收益；因此本项目必须在完整 token 步和端到端中验证，不能按理论 FLOPs 推断。

## MiniMLX 现状

Qwen GDN 的普通单步更新在 MLX-LM 0.31.3 已经进入 `mx.fast.metal_kernel`，不是 Python
逐 token recurrence。MiniMLX 现有 `python/minimlx_runtime/kernels/gdn.py` 优化的是长序列
Prefill（`tokens >= 16`），单步 Decode 会回退到上游 kernel。Qwen3.8-27B 的该层有 48 个
GDN 层，所以每生成一步会跨过多个独立 GDN dispatch；整体仍受 4-bit 权重读取主导，单层
融合的总收益有上限。

### 本机 Qwen3.8-27B GDN 单请求 A/B（2026-10-13）

相同 MiniMLX、MTP/APC 关闭、greedy 固定生成 128 tokens、warmup=1、repeats=3。自定义
recurrence 保留 MLX 原始 gate 计算；两个上下文的 token digest 均与基线完全一致。

| Context | Baseline decode | Custom recurrence | TPS 变化 | Digest |
|---:|---:|---:|---:|---|
| 1K | 7116.2 ms / 17.99 tok/s | 7098.4 ms / 18.03 tok/s | +0.22% | 一致 |
| 4K | 7236.5 ms / 17.69 tok/s | 7231.7 ms / 17.70 tok/s | +0.06% | 一致 |

自定义路径路由 38,448 次、fallback 1,152 次（包含正式运行、warmup 和调用计数诊断）。
收益低于本机运行噪声，不能视为性能提升；自定义 recurrence 只证明数值一致，没有达到
保留运行时优化分支的收益门槛。

原始 gate 融合候选未通过数值验证：同一组 upstream `g/beta` 输入下，recurrence 与 MLX
reference 的完整 state 和输出完全一致；Metal 中手写 gate 计算仍有 BF16 偏差，两步
synthetic state 对照最大误差约 0.048。普通 float、BF16 cast、显式 round-to-nearest-even
均未复现 MLX gate，因此 gate 融合不进入运行时，仅保留诊断证据。

## 后续

不继续微调同一 recurrence 的线程映射。若重开 GDN 优化，应有新的公开实现证据或硬件
profile 指向不同瓶颈，例如 projection/QK normalization 边界融合；单独做数值回归，保留
上游 epsilon、精确倒平方根、输入 dtype 中间舍入和缩放顺序。任何候选都必须通过完整 state、
greedy digest 和端到端 paired A/B；MTP/APC 仍为独立能力，不纳入本项。

不改模型适配器架构，不新增 Qwen3.8 专属路由；能力应由张量形状、dtype、state layout、
硬件和序列长度识别，所有不匹配情况透明 fallback。本轮已在本机 Metal 验证，但未证明
生产性能收益。

### Q/K RMSNorm 融合正确性（2026-10-14）

按 libmir 公开实现记录的精确 Metal `rsqrt` 与两阶段输入 dtype 舍入实现独立实验。4 步
BF16 recurrence 对照中，融合路径 output 与完整 FP32 state 对 mlx-vlm reference 均逐位一致；
未融合的同归一化输入 recurrence 控制组也逐位一致。定位到的必要语义是：Q/K RMSNorm 先
输出 BF16，再将缩放系数转换成 BF16 后相乘；直接用 FP32 缩放系数会导致 K 单元素 ULP
差异，并令 recurrent state 漂移。修正后最大 output/state 绝对误差均为 0。

该候选已通过独立 opt-in 路由接入单请求 benchmark：只对单 token、推理态的 Qwen3.5 GDN
层绕过上游 Q/K norm，并在兼容的 Metal recurrence 内融合；prefill 继续使用原归一化，
不兼容及 fallback 路径恢复上游归一化。它不改变默认执行。

### Q/K RMSNorm 融合端到端 A/B（2026-10-14）

Qwen3.8-27B、MTP/APC 关闭、greedy 固定生成 128 tokens，warmup=1、repeats=3；依次运行
stock、recurrence-only、fused-QK-norm。三种策略在 1K 与 4K 的 token digest 均一致。
融合路径在计时/预热诊断中共路由 38,448 次、fallback 0 次，确认完整走了候选路径。

| Context | Stock decode | Recurrence-only | Fused Q/K norm | 融合相对 recurrence | Digest |
|---:|---:|---:|---:|---:|---|
| 1K | 7174.0 ms / 17.84 tok/s | 7172.6 ms / 17.85 tok/s | 7343.1 ms / 17.43 tok/s | -2.35% | 完全一致 |
| 4K | 7316.7 ms / 17.49 tok/s | 7353.3 ms / 17.41 tok/s | 7219.7 ms / 17.73 tok/s | +1.84% | 完全一致 |

融合相对 stock 分别为 -2.30% 与 +1.37%。上下文间方向相反，且与 recurrence-only 的差距较小，
当前不能区分真实收益与运行噪声；结论是“正确性通过、性能收益未证实”，不是优化成功。
候选继续保持 opt-in，不启用默认，也不应以这组数据宣称 MiniMLX 整体优于 oMLX。
后续若要裁决性能，应采用策略顺序平衡的配对复测；若仍无稳定收益，则停止维护该融合分支，
将 profiling 转向每 token 的量化权重投影/dispatch 等更可能支配 27B decode 的成本。

## Unsloth MLX Release / 源码调研（2026-10-14）

### 与普通单请求 decode 直接相关的已实现工作

Unsloth 的 [MLX decode convolution + SiLU fusion PR #1191](https://github.com/unslothai/unsloth-zoo/pull/1191)
融合的是单 token GDN 的 depthwise convolution 与 SiLU，并保持原生累加、dtype cast、激活顺序；
它**没有**把 cache window 拼接、QKV projection、Q/K/V 切分或 GDN recurrence 全部融合进该 dispatch。
实现只在兼容的推理单行路径接管；prefill、训练、分布式、unsupported dtype/shape 或 upstream
代码变化时回退原生路径。其在 Qwen3.5
0.8B BF16（18 个 eligible recurrent layers）上做六组 AB/BA 配对、每请求生成 512 tokens，报告
decode +3.20%，95% 配对区间 +3.05% 至 +3.34%；Gemma E2B 无 eligible 层的对照为无可分辨收益。
这是真实 Apple Silicon/Metal 结果，但不是 Qwen3.8-27B 结果，也不能线性外推。

Unsloth 的 [residual norm fusion PR #1246](https://github.com/unslothai/unsloth-zoo/pull/1246)
将单行 decode 的 RMSNorm、residual add 和可选 scalar scale 融成一次 dispatch，并严格复现原生
reduction/dtype rounding。其三个不同模型的 paired decode 结果约 +1.89% 到 +3.00%，且 ABBA
观察范围均为正；Qwen3.8-27B 未在该 PR 的测量集里。它能覆盖 decoder 每 token 的多个通用 norm/residual
边界，值得优先于只融合 GDN 内部 norm 的候选做适配性核对。

Unsloth 的 [MoE router fusion PR #1252](https://github.com/unslothai/unsloth-zoo/pull/1252) 将
Qwen/Gemma 路由链的多个小操作融合为一次 dispatch；三组模型报告约 +1.54% 至 +2.90% decode，
验证了减少 per-layer 小 kernel dispatch 的普遍价值。不过 Qwen3.8-27B 是 dense 模型，不含 MoE
router/expert 路径，因此这项不是当前 27B 的优化候选；可留给后续 A3B/MoE 型号。

### Release headline 的边界

Unsloth [v0.1.906 release](https://github.com/unslothai/unsloth/releases) 宣称 MLX speculative decoding
最高 2.8x，并列出 MTP、DFlash、DSpark、EAGLE-3 draft；[v0.1.907 release](https://github.com/unslothai/unsloth/releases)
提及 Qwen3.5/3.6/3.8 更快 prefill 和 speculative decoding。前者的 speculative 是 draft/verify 多 token
路径，不是关闭 draft 后的普通单-token decode；按项目约定，MTP/DFlash 作为独立、显式开启的加速能力
单独评估。后一条 release 摘要没有给出 uncached single-request cold-prefill 的独立数值和控制条件；
Unsloth 对长对话/后续请求的缓存收益也不能归因成首次单请求 prefill 加速。故这些 release headline
是定位相关代码的索引，不作为本项目普通 decode 或冷 prefill 的性能证据。

### 对 MiniMLX 的可迁移判断

Qwen3.8-27B 是 dense Qwen3.5 架构，且该项目中有 48 个 GDN 层。当前 MiniMLX 的融合候选只把 Q/K
RMSNorm 移进 recurrence；Unsloth #1191 所覆盖的前置 recurrent convolution/SiLU 是不同的
dispatch 边界，不能将本轮 Q/K 融合结果当作它的验证，也不能照抄其 monkey-patch/AST 重建手段。
从公开数据看，最可借鉴的是“减少每个 GDN 层的单行 dispatch 数量、保留原生算术顺序、按代码契约
精确识别、其他输入透明 fallback、实测 kernel route”的方法和融合边界本身。

但需要先做代码映射：确认当前 mlx-vlm 版本中 Qwen3.5/Qwen3.8 的 recurrent conv 实际调用链，比较
其与 #1191 的 source contract、缓存状态布局、dtype 舍入及 QKV 切片语义，再清点每 token 实际被删除
的 dispatch 和临时数组。只有确认适用后，才值得构造一个候选；验收应先做 Metal bitwise/tolerance 与
greedy token/state 对照，再做固定输出长度、关闭 MTP/APC 的 paired AB/BA 或 ABBA 测量。不得因为外部
Qwen3.5 0.8B 的 +3.2% 宣称本模型必然有收益。

**调研结论：有明确、可验证的普通 decode 候选——GDN recurrent conv/SiLU 融合；其次是通用
residual RMSNorm/add 融合。MoE 路由融合不适用于当前 dense 27B；speculative 属于另一项独立能力。
本次只完成源码和证据调研，未改运行时或启动新的实验。**

### 锁定版本源码映射（本机 MiniMLX `.venv`）

已核对实际安装版本：`mlx==0.32.2`、`mlx-lm==0.31.3`、`mlx-vlm==0.7.2`、
`transformers==5.17.0`。因此以下不是按浮动 `main` 推断，而是对本机实际运行代码进行比对；
MLX-VLM 同版本源码也可从 [v0.7.2 language.py](https://github.com/Blaizzy/mlx-vlm/blob/v0.7.2/mlx_vlm/models/qwen3_5/language.py)
与 [v0.7.2 gated_delta.py](https://github.com/Blaizzy/mlx-vlm/blob/v0.7.2/mlx_vlm/models/qwen3_5/gated_delta.py)
复核。

两条 Qwen3.5 GDN 路径的数据流在运算语义上高度一致：四个独立线性投影（QKV、z、b、a）→
读取/初始化 3-row 卷积状态 → 拼接状态窗口与新 QKV → depthwise causal conv → SiLU → 按
`[key_dim, 2*key_dim]` 切分并 reshape Q/K/V → Q/K RMSNorm 与缩放 → `gated_delta_update` →
gated norm → 输出投影。差别在 cache API：mlx-vlm 通过 `cache.update_window(...)` 和
`cache.update_recurrent(...)` 管理 hybrid cache；mlx-lm 在 GDN layer 内直接读写 `cache[0]`、
`cache[1]`。因此不能复制一份完整 Qwen3.8 `__call__` 做特例；优化 kernel 应只接收已经形成的
`conv_input` 和 conv weight，cache 状态更新继续交给各自上游模型。

当前 mlx-vlm 已有单 token BF16/FP16 depthwise conv 快路：先用 `mx.compile` 的
`_qwen3_5_decode_depthwise_conv` 物化卷积输出，再由外层 `nn.silu` 处理。mlx-lm 0.31.3
当前 `Qwen3_5GatedDeltaNet.__call__` 仍写作 `nn.silu(self.conv1d(conv_input))`。两者的共同
可复用边界是“单 token recurrent depthwise conv + SiLU”，不是类名或具体模型 ID；且 MiniMLX 已有
`mtp_gdn_prework.py` 是另一路径（MTP verifier 的 3–9 token prework），不应当作普通 decode 的
实现或复用证明。

**实现决策：** 若开始实现，写一个 MiniMLX 共用的单行 GDN prework kernel/helper，只融合 conv 与
SiLU，输入输出保持上游要求的 shape/dtype，并逐运算匹配 accumulation/cast/SiLU 次序。分别用
严格的类/属性/函数签名契约，在 mlx-vlm 和 mlx-lm GDN 类上安装窄作用域 hook；不替换完整模型
`__call__`，不改 cache API，不触碰 MTP、prefill、training 或 GDN recurrence。所有未知实现、mask、
batch>1、dtype/shape 不符都走上游路径，并报告 eligible/route/fallback 计数。这样共用的是一个
张量能力，不是 Qwen3.8 专属适配器；上游契约变动时安全失效。Unsloth PR 对源契约拒绝和 fallback
做了明确测试，但也收到 reviewer 关于 AST 重写脆弱性的提醒；MiniMLX 不应移植其 AST 重建方式，
应优先用版本/属性契约并将模型调用代码留给上游。

这个候选有实测支持但收益预期应保守：Unsloth 的 M3 Max / Qwen3.5-0.8B BF16 配对测试报告
decode +3.20%（95% 区间 +3.05% 到 +3.34%）；同一 PR 在没有可应用层的 Gemma E2B 对照上没有
可分辨收益。该结论不能外推为 27B 收益。实现后必须先对真实 27B Qwen3.8 做单层输出与缓存输入
对照，再做固定 128-token greedy digest 和端到端 paired A/B；MTP/APC 关闭，且保留统一 fallback。

### 候选实现状态

已新增 `python/minimlx_runtime/kernels/gdn_conv_silu.py`：一个共享 Metal conv+SiLU kernel，按
实例契约分别接入 mlx-vlm 0.7.2 与 mlx-lm 0.31.3 的 Qwen GDN layer，cache 更新仍由各自上游代码
执行；只在 `MINIMLX_GDN_DECODE_FUSED_CONV_SILU=1` 时安装，不接入 MTP。新增
`tools/verify_gdn_conv_silu.py` 同时对照 mlx-vlm 的 compiled depthwise-conv 路径与 mlx-lm 的
原生 Conv1d；单层 output 默认要求 bitwise 相同。端到端 benchmark 会打印 eligible layer、route、
fallback 计数。

当前只完成静态语法检查。执行环境没有可用 Metal device，且虚拟环境未安装 pytest，所以本机尚未
通过 kernel 编译、数值核验或端到端运行；不得将候选视为已验证或默认启用。待 Apple Silicon 上先跑
数值脚本通过，再跑固定 128-token greedy digest 与 MiniMLX 的 27B 单请求 benchmark。
