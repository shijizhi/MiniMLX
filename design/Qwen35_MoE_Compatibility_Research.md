# Qwen3.5 MoE compatibility research

## Candidate checkpoint facts

Checkpoint: `mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit-REAP-19B`.

Static inspection of the local `config.json` and safetensors index found:

- `model_type=qwen3_5_moe`, architecture `Qwen3_5MoeForConditionalGeneration`.
- 40 decoder layers; all 40 have routed-expert and shared-expert tensors.
- 128 retained experts and `num_experts_per_tok=8`.
- Expert projections are mixed precision, not uniformly Q4: 35 layers have
  Q4 gate/up/down projections; two have Q8/Q8/Q8, two Q4/Q8/Q8, and one
  Q8/Q4/Q4 (gate/up/down order). Quantization group size is 64.
- Shared expert and shared-expert gate tensors are present for all layers.
- The checkpoint is REAP-pruned; its 128 experts must not be presented as the
  original model's full expert count.

## Runtime implementation path

The installed MLX-LM source registers `mlx_lm.models.qwen3_5_moe.Model`, whose
text model inherits Qwen3.5's hybrid decoder. With `num_experts > 0`, each
decoder layer constructs `Qwen3NextSparseMoeBlock`: a dense router selects
top-k expert IDs, `SwitchGLU` evaluates the selected gate/up/down projections
through `mx.gather_mm`, and a separate shared MLP plus sigmoid gate is added.
The MoE model's `sanitize` maps its checkpoint namespaces to this runtime
layout and removes vision weights for text-only loading.

MiniMLX's existing Q4 dense MLP fusion/native dispatch expects ordinary
`mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj`. The MoE projections instead
live under `mlp.switch_mlp.*`, with the shared MLP under
`mlp.shared_expert.*`; therefore the dense MLP patch does not match the routed
expert module. MoE expert-kernel optimization is explicitly out of scope until
load, routing, shared-expert execution, and short greedy generation pass.

## Current MiniMLX changes

- The model descriptor now advertises `moe` and carries expert/top-k and
  intermediate-size metadata.
- `verify_model_support.py` reports MoE layer count, expert count, top-k,
  expert tensor dimensions, observed expert quantization bits, and shared
  expert presence separately from dense Q4 MLP status.
- Interactive Metal smoke passed: the model loaded, all 40 MoE blocks were
  constructed, and a 4-token greedy generation completed. The observed model
  had 30 linear-attention and 10 full-attention layers, generated tokens
  `[11, 353, 1044, 264]`, and reported 11.586 GiB peak MLX memory.
- These timings are smoke diagnostics, not a performance result. The first
  verifier output exposed a reporting bug: expert bit width was sampled only
  from layer 0 (Q8), while the separate Q4 layer count was 38. The verifier
  now aggregates bit widths and per-layer bit patterns across all MoE layers.
  The corrected rerun confirmed `4/4/4=35`, `4/8/8=2`, `8/4/4=1`,
  `8/8/8=2` (gate/up/down), and 38 layers with at least one Q4 expert
  projection. It again passed 4-token generation at 11.586 GiB peak.

## Short-context smoke

Run from an interactive macOS terminal/RustRover process with GPU access:

```bash
cd /Users/jizhishi/mycode/ai/minimlx
./.venv/bin/python tools/verify_model_support.py \
  /Users/jizhishi/.omlx/models/mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit-REAP-19B \
  --max-tokens 4
```

The 1K, 4K, and 16K prompts all completed four-token generation successfully.
Reported TTFT was 2384.0 / 8715.7 / 34984.6 ms, and E2E was 2479.4 /
8815.3 / 35101.8 ms. These are single-run compatibility diagnostics, not
benchmark comparisons. This command did not report peak memory for each
context; the only measured peak is 11.586 GiB from the short smoke.

Compatibility status: smoke-pass and context-pass through 16K. Per-context
peak memory and longer contexts remain unverified. No throughput comparison
is needed before closing this compatibility step.
