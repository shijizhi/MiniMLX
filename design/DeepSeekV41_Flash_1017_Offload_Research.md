# DeepSeek V4.1 Flash 1017 分层加载研究

## 结论

DeepSeek V4.1 Flash 1017 可以复用 MiniMLX 的 SSD 分层、专家缓存、mmap
索引和预取抽象，但不能复用 Qwen3.8-Flash-Next 的执行器。两者的权重命名、
注意力、路由和推理协议不同，应建立独立的 `deepseek41` family adapter。

## 模型事实

- 1017 是 2026-10-17 的权重更新，配置和架构与首发 V4.1 Flash 保持兼容。
- 552B backbone，约 8B prompt active、16B output active。
- Causal Encoder–Decoder：20 层 causal encoder + 20 层 decoder。
- 约 196B 参数的 Engram/N-gram 表，两个主要表约 183 GiB，不能作为普通常驻权重处理。
- 使用 compressed sparse attention / 双级 indexer，而不是 Qwen3.8 的 QSA 实现。
- 使用 4 路 Hyper-Connections。
- DSpark 是三阶段、每层 128 experts、每 token 激活 3 个的 draft head。
- 1017 checkpoint 是混合 MXFP4 experts、MXFP8 dense weights、UE8M0 scales，embedding
  和 LM head 保持 BF16。

## 与 Qwen3.8 的共同层

```text
ModelDescriptor
  → offload capability detection
  → ResidentMemoryPlanner
  → MmapWeightStore
  → HotExpertCache / EngramPageCache
  → PrefetchScheduler
  → family-specific executor
```

共同的安全原则：完整常驻权重能放入安全预算时直接加载；只有预算不足时才启用
SSD offload。Engram 表必须有独立的 page budget，不能和普通 expert cache 共用一个
不透明的容量值。

## 必须新增的执行组件

1. `DeepSeek41TensorLayout`：识别 encoder、decoder、sparse-attention indexer、Engram
   table、Hyper-Connections 和 DSpark tensors。
2. `EngramPageStore`：按 hash row/page 读取 N-gram 表，支持 page-level LRU 和异步预取。
3. `DeepSeek41Router`：实现 384 experts / top-6 主路由、图像 token 专用 routing bias
   和 shared expert 合并。
4. `DeepSeek41CEDCache`：分别管理 encoder output、压缩 KV 和 decoder global KV。
5. `DSparkExecutor`：独立于 Qwen MTP，先作为可选加速组件接入。

## 当前阶段

本阶段只完成模型事实识别和能力声明，不让现有 `mlx-lm` 或 Qwen executor 错误加载
DeepSeek V4.1。真实执行器需要先验证 Apple MLX 是否已有 `deepseek_v41` backend；
若没有，应先做 CPU/headless tensor-layout fixture，再实现 MLX/Metal 层。

## 参考

- DeepSeek V4.1 Flash 1017 model card：<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>
- vLLM recipe：<https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml>
- DeepSeek 官方介绍：<https://www.deepseek.com/en/news/deepseek-v4-1-flash/>
- Strata guide：<https://www.stratat.com/en/guides/deepseek-v4-1-flash-1017>
