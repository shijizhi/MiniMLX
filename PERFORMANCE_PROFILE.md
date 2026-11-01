# MiniMLX / PyMiniMLX 性能全景

这份记录只整理已经完成的测量，不包含本轮新增跑分。两边使用同一
Qwen3.5-0.8B-MLX-4bit 模型、原始 token `[9419]`（`Hello`）、prefill 1
token、decode 16 token；warmup 3 次、正式测量 5 次、取中位数。

## 1. 端到端基线

| 指标 | MiniMLX Swift | PyMiniMLX | Swift / Py |
|---|---:|---:|---:|
| Prefill | 27.4 ms | 14.0 ms | 1.96x |
| Decode（16 tokens） | 490.1 ms | 212.8 ms | 2.30x |
| Decode throughput | 32.65 tok/s | 75.19 tok/s | 0.43x |

当前需要解释的主要差距是 decode，而不是输出结果或 token 逻辑：Swift
每个 token 的平均耗时约 30.63 ms，PyMiniMLX 约 13.30 ms，多出约
17.33 ms/token。

## 2. 解码层级分布

以下是一次真实 cached decode 的层级测量，均为单层耗时，测量边界在
每层之后强制完成。它们用于定位，不应当和其它“模块 stage”表直接相加。

| 层/类型 | MiniMLX Swift | PyMiniMLX | Swift / Py |
|---|---:|---:|---:|
| L0 isolated full-attention layer | 1.038 ms | 0.687 ms | 1.51x |
| L3 isolated full-attention layer | 0.974 ms | 0.616 ms | 1.58x |
| L3 cached attention mixer | 1.101 ms | 0.763 ms | 1.44x |

已知模型中 full-attention 为 6 层；其余层是 Gated DeltaNet。full-attention
单层确实更慢，但仅凭这个绝对差值还不足以解释整个 2.30x 的端到端差距，
所以不能把结论简单归因于 attention。

## 3. Gated DeltaNet 阶段对照

下面是已做过 warmup 的首个 GDN 阶段 profile（单位 ms）。这些 stage 是
分别强制完成的边界测量，存在共享中间结果，不能直接求和。

| 阶段 | Swift | PyMiniMLX | 观察 |
|---|---:|---:|---|
| qkv | 0.257 | 0.257 | 基本一致 |
| z | 0.175 | 0.174 | 基本一致 |
| beta | 0.187 | 0.125 | Swift 较慢 |
| a | 0.175 | 0.158 | 接近 |
| convolution | 0.212 | 0.227 | Swift 略快 |
| SiLU | 0.155 | 0.126 | Swift 较慢 |
| q norm | 0.239 | 0.179 | Swift 较慢 |
| k norm | 0.201 | 0.177 | Swift 较慢 |
| delta recurrence | 0.282 | 0.245 | 接近 |
| gated RMS norm | 0.248 | 0.278 | Swift 略快 |
| output projection | 0.462 | 0.185 | Swift 明显较慢 |

GDN 的大部分核心数学已经在同一数量级；当前最醒目的 GDN 单点是
`output projection`，但它需要和 full-attention 的同类 projection 一起
看，不能只凭这一张 stage 表修改 recurrence。

## 4. Full-attention 内部阶段

这是目前最集中的可疑区域。Swift 和 PyMiniMLX 的代码顺序已经逐项对齐：
Q/K/V projection、Q/K norm、partial RoPE、KV append、GQA repeat、
score、causal mask、softmax、value matmul、gate、output projection。

| 阶段 | Swift | PyMiniMLX | Swift / Py |
|---|---:|---:|---:|
| q projection（A/B 对照） | wrapper 0.212 ms；直接 quantizedMM 0.221 ms | 0.222 ms | 接近 |
| k projection | 0.12–0.18 ms | 0.163 ms | 接近/有波动 |
| v projection | 0.12–0.20 ms | 0.194 ms | 接近/有波动 |
| q norm | 0.18–0.22 ms | 0.200 ms | 接近 |
| k norm | 0.18–0.20 ms | 0.196 ms | 接近 |
| RoPE（Swift 合并 q/k） | 0.49–0.54 ms | q rope 0.260 + k rope 0.259 ms | 边界不同，不能直比 |
| KV append | 0.12–0.20 ms | k 0.176 + v 0.169 ms | 边界不同 |
| score + softmax + value | 0.223 ms | 0.240 ms | Swift 略快 |
| output projection | float32 输入：wrapper 0.365 ms、直接 quantizedMM 0.387 ms；bfloat16 输入：0.213 ms | 0.232 ms | float32 路径约 1.6x；bfloat16 已接近 |

注意：这是“强制完成每个阶段”的诊断 profile，不能把所有行相加来重建
490.1 ms；它的用途是指出下一段代码审查位置。当前最值得共同审查的
代码点因此是：

```text
MiniMLXAttention.callAsFunction
    let scores = matmul(q, cachedKeys.transposed(...))
    let probabilities = softmax(which(...))
    let result = matmul(probabilities, cachedValues)
    return outputProjection(merged * sigmoid(gate))
```

以及 PyMiniMLX 对应的 `GatedAttention.__call__` 中同样的三段 matmul 和
`o_proj(output)`。

## 5. 已排除或暂不成立的解释

| 方向 | 证据 | 当前结论 |
|---|---|---|
| Q/gate 用 `split` 导致主要损失 | 改成 prefix/suffix slice 后 golden 正确，但基线仍约 32.65 tok/s | 不是主要原因 |
| fused SDPA 会自动解决 | 显式 mask + `forceFused` 实验反而降到约 33 tok/s | 暂不采用 |
| `.contiguous()` 能解决 MLP 慢 | MLP 前后增加 contiguous 诊断没有改善 | 不是简单布局拷贝问题 |
| packed gate/up fusion | golden 正确，但一次基线由约 39 降到约 35.8 tok/s，已撤回 | 不采用 |
| Swift 的 safetensor `.default` 加载 | Swift 当前 safetensor GPU load 报 `Load::eval_gpu Not implemented`；mlx-swift-lm 的 loader 也明确使用 `.cpu`，因为 safetensor load 没有 GPU 实现且 Apple Silicon 使用统一内存 | 不是当前差距的主因；代码保持 `.cpu` |
| MLP 是端到端主因 | 隔离 MLP projection：Swift 约 0.319 ms、Py 约 0.326 ms | 不是首要差异 |

## 6. q projection 的测量修正

原先记录的 Swift `q projection=0.75–0.82 ms` 不能作为有效对比：PyMiniMLX
在测量前执行了 5 次同边界 warmup，而 Swift 旧诊断路径没有执行 warmup，
因此可能把首次 Metal pipeline 初始化计入 q projection。

当前代码增加了只测 L3 `q_proj + reshape` 的命令：

```text
minimlx native-profile-q <model-directory>
```

它在 `Qwen35NativeModel.swift:706`
中执行与 PyMiniMLX 相同的 5 次 warmup，并在同一次运行中对 wrapper 和直接
`quantizedMM` 做 A/B 对照。两条 Swift 路径分别为 0.212 ms 和 0.221 ms，
PyMiniMLX 为 0.222 ms；因此旧的单次 0.75–0.82 ms、0.393 ms、0.272 ms
都不能作为当前 q projection 差异的证据。

## 7. 当前结论

1. 端到端差距已确认：Swift decode 比 PyMiniMLX 慢约 2.30x。
2. GDN recurrence 本身没有显示出同等数量级的差距。
3. 修正 warmup 和计时边界后，q projection 的 A/B 对照为 Swift wrapper
   0.212 ms、Swift 直接 quantizedMM 0.221 ms、PyMiniMLX 0.222 ms。
   `MiniMLXQuantizedLinear` 封装没有造成可见损失，q projection 不是当前
   Swift/Py 性能差距的有效解释。
4. `score/softmax/value` 的 Swift 耗时为 0.223 ms，低于 PyMiniMLX 的
   0.240 ms，因此该段不是差距来源。
5. 当前 full-attention 内已确认的差异集中在 `output projection` 的输入
   dtype：Swift 的真实输入为连续 `[1,2048] float32`，对应路径约
   0.365–0.387 ms；同一输入临时转换为 `bfloat16` 后为 0.213 ms，已接近
   PyMiniMLX 的 0.232 ms。wrapper 与直接 `quantizedMM` 没有显著差异，
   因而下一步应追溯 Swift 图中导致 float32 的具体算子，并与 PyMiniMLX
   的 dtype 推导对齐。
