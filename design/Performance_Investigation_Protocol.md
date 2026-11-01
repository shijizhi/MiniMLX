# MiniMLX 性能排查准则

> **现行协议（优先级高于本文后面的历史记录）**
>
> MiniMLX 的性能目标是超过 oMLX。PyMiniMLX 不属于当前性能流程，不得作为性能目标、性能排序或优化取舍依据；它最多只能作为历史数值参考，后续不再使用。
>
> 优化必须覆盖场景曲线，而不是只针对当前 4K 测试点。至少考虑短 Prefill、4K、16K、32K 及更长上下文、单 token Decode，以及未来的多 token verify/decode。只要一条路径在任一受支持场景中稳定产生正收益，并且正确性、内存和稳定性可接受，就应纳入运行时体系，由硬件、模型格式和序列长度进行 dispatch；不能因为当前 4K 收益较小而放弃。
>
> 每轮工作必须执行：
>
> 1. 先测 MiniMLX 与 oMLX 的同配置端到端结果。
> 2. 找出 MiniMLX 实际落后的完整场景。
> 3. 只研究该场景中有公开算法、论文或成熟实现依据的方向。
> 4. 在 MiniMLX 中独立实现，不复制 oMLX 代码。
> 5. 用短、中、长上下文和 Decode 重新测量收益曲线。
> 6. 只有确认收益、正确性和 fallback 后，才纳入正式路径。
>
> ANE 特别要求同时测量编译时间、编译缓存后的稳定时间、MLX↔ANE 数据搬运、同步等待、峰值内存和数值误差。ANE 不应被无条件启用；应使用固定 shape、长度阈值、硬件能力检测和 GPU fallback。不能只因为 4K 收益小就否定 ANE，也不能只因为长上下文有收益就忽略短上下文退化。

下面旧章节保留为历史记录；其中涉及 Swift、PyMiniMLX 的流程已经废止，不得作为当前操作规范。

本文件是性能工作开始前必须阅读的工作约束。目标是通过证据逐步消除最大性能差距，而不是依靠猜测或反复试验碰运气。

## 核心循环

每一轮只执行以下循环：

1. 在 Swift 与 PyMiniMLX 完全相同的条件下测量同一个边界。
2. 计算差值：`Swift 耗时 - PyMiniMLX 耗时`。
3. 在当前已测模块中选择差值最大的模块作为唯一当前目标。
4. 将该模块继续拆成更小的阶段，直到范围收敛到可以直接阅读和对照的代码范围。
5. 逐行对照 Swift 与 PyMiniMLX 的实现、输入形状、数据类型、状态布局和调用的 MLX 算子。
6. 只修改当前目标段，并重新测量当前目标段。
7. 只有确认当前目标段的差值确实变小，且结果和状态仍然正确，才允许进入下一段。
8. 当前目标改善后，重新按差值排序，选择新的最大差值目标。

## 严格禁止

- 没有差值证据时，不得凭经验猜测瓶颈。
- 当前目标尚未解释或改善时，不得跳到其他模块。
- 局部测试不得被全局 benchmark 替代；全局 benchmark 只用于阶段性确认。
- 不得因为某次单次测量波动就改变目标；需要使用同一边界和稳定统计方式。
- 不得把“换一种测试方式”当成优化本身。
- 不得把 MLX 或系统版本差异作为默认解释；必须先排除实现、边界、形状、dtype 和同步方式差异。
- 不得同时修改多个独立模块，否则无法判断收益来源。

## 测量要求

每个局部结果必须记录：

- Swift 时间；
- PyMiniMLX 时间；
- 差值及每次调用平均差值；
- 完全一致的输入 token、context、batch、warmup、repeats 和同步边界；
- 输入输出 shape、dtype、状态布局；
- 本次改动前后的结果正确性。

如果两边的测试边界不一致，结果只能标记为“不可比较”，不能用于决定优化方向。

## 当前工作状态与校正后的测量台账

以下内容覆盖此前所有旧结论；旧数据保留为历史记录，但不能与下面的新边界混用。

### A. 全局 benchmark：只看最终吞吐

当前完整优化版本最近一次结果：

```text
prefill=22.9 ms
decode=370.8 ms / 16 tokens
43.15 tok/s
```

此前 PyMiniMLX 同一 raw token、16 token 结果约为 `99.75 tok/s`。两者相差约
`2.31x`，但这只能说明全局差距，不能告诉我们差距落在哪个阶段。

### B. 可用于全局排序的完整算法边界

此前曾有一组同步到每层的类型汇总：

```text
                    Swift          PyMiniMLX        差值
linear_attention    620.034 ms     158.733 ms       +461.301 ms
full_attention      201.909 ms      53.524 ms       +148.385 ms
```

这组数据是在旧代码状态下测得，不能直接当作当前版本数值；它只能证明当时
`linear_attention` 的完整类型边界差距最大。今后如果要重新排序，必须重新测量
当前代码的完整类型边界，不能用局部 profile 外推。

### C. 当前真实 L17 linear mixer profile

现在使用的 real profile 会在生产顺序中逐阶段同步，并且 Swift/PyMiniMLX 都有
`warmup=3`。最近一组可对齐数据如下：

```text
阶段                         Swift       PyMiniMLX       差值
qkv_projection               0.268 ms    0.245 ms        +0.023 ms
z_projection                 0.229 ms    0.208 ms        +0.021 ms
beta_projection              0.192 ms    0.183 ms        +0.009 ms
a_projection                 0.189 ms    0.187 ms        +0.002 ms
convolution                  0.241 ms    0.211 ms        +0.030 ms
activation                   0.178 ms    0.152 ms        +0.026 ms
qkv_preparation              0.301 ms    0.236 ms        +0.065 ms
recurrence                  0.302 ms    0.271 ms        +0.031 ms
gated_rms_norm               0.758 ms    0.651 ms        +0.107 ms
output_projection            0.206 ms    0.216 ms        -0.010 ms
```

其中 `gated_rms_norm` 的 Swift/Py 差值 `+0.107 ms` 是其五个子阶段差值之和；
它不能与之前未拆分的 `0.299/0.238 ms` 结果混用。

同一组拆分 profile 的总和为：

```text
Swift       2.864 ms
PyMiniMLX   2.560 ms
差值        0.304 ms / layer
```

这个总和包含每个子阶段的强制同步开销，只能用于“阶段内部定位”，不能乘以层数
后当成 production decode 的真实总耗时。因此此前的 `0.388 ms × 18 × 16 ≈ 110 ms`
结论作废。

### D. 已完成的局部优化，只记录局部事实

这些结果证明对应局部 kernel 有改善，但没有自动证明全局收益：

```text
z projection       约 0.31 → 约 0.16–0.20 ms
beta projection    约 0.31 → 约 0.18–0.19 ms
qkv projection     约 0.414 → 约 0.25–0.27 ms
a projection       约 0.375 → 约 0.19–0.21 ms
output projection  real profile 约 0.366 → 约 0.21 ms
```

每个局部优化都必须再通过当前完整 benchmark 验证，不能把局部节省时间直接乘
调用次数作为全局收益。

### E. 当前正确的决策规则

1. 全局排序只使用完整 production 边界的 Swift/Py 差值。
2. real profile 只用于把一个已确认的大模块继续定位；它的强制同步总和不能外推。
3. 单个子阶段若按调用次数估算不能达到用户设定的收益阈值，就不能作为下一步优化目标。
4. 每次改动后先验证当前目标边界确实改善，再运行一次完整 benchmark。
5. 如果 local profile 与 global benchmark 不能解释，结论必须标记为“测量不充分”，
   不能继续猜测。

### F. 当前统一测量入口

先用完整 layer 类型边界建立当前基线。两边都使用：

```text
raw token = [9419]
steps = 16
warmup = 3
repeats = 5
statistic = median
```

计时范围严格是：单个 decoder layer 调用加 `eval(layerOutput)`。embedding、最终
RMSNorm、lm_head、argmax 和 host bookkeeping 均不计入 linear/full layer 时间。
每次 repeat 都重新创建 request state，并完整执行 prefill 和 16-token cached decode。

Swift：

```text
minimlx native-py-profile-attention-types <model-directory> 16 3 5
```

PyMiniMLX：

```text
../pyminimlx/.venv/bin/python \
  tools/profile_pymimix_attention_types.py \
  <model-directory> 16 3 5
```

这个结果只用于确定完整 `linear_attention` / `full_attention` layer 的差值。
确认目标后，才使用 `native-py-profile-real-linear-stages` 和对应的
`profile_pymimix_cached_linear_stages.py --real` 拆解该目标；拆解结果仍然只用于
定位，不得把各个强制同步阶段的时间直接乘层数外推成全局收益。

## 每次开始工作前的自检

开始任何性能操作前，先回答：

1. 当前使用的是全局边界、完整模块边界，还是强制同步定位边界？
2. Swift 与 PyMiniMLX 是否使用完全相同的边界和 warmup？
3. 这个差值能否在不混用其它 profile 的情况下解释全局差距？
4. 本次修改是否只触及当前目标？
5. 修改后如何证明当前目标改善且结果正确？

如果答不清这些问题，先整理测量，不进行新的性能猜测。
