# ANE projection experiment

This is an isolated research probe. It is not enabled in the MiniMLX runtime.

## Reproduce the comparison

Build a fixed-shape Core ML model from the real Qwen3.5 L17 `in_proj_z` weights:

```sh
./.venv/bin/python tools/build_ane_linear_probe.py \
  /tmp/minimlx-qwen35-l17-z-4096.mlmodel \
  --tokens 4096 \
  --model-dir /Users/jizhishi/.omlx/models/mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --layer 17
```

Compile it with Xcode's `coremlcompiler`, run `ane_linear_probe`, then compare
the saved ANE output with MiniMLX's real Q4 path:

```sh
./.venv/bin/python tools/profile_ane_vs_minimlx_q4.py \
  /Users/jizhishi/.omlx/models/mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --layer 17 --tokens 4096 \
  --ane-output /tmp/minimlx-qwen35-l17-z-4096-ane-output.bin
```

The input is deterministic and shared with `ane_linear_probe.swift`.

## Observed result on the M4 test machine

For the real L17 `in_proj_z` projection at 4096 tokens:

| Tokens | MiniMLX MLX Q4 | Core ML ANE | Core ML `.all` |
|---:|---:|---:|---:|
| 2,048 | 5.376 ms | 3.514 ms | 3.529 ms |
| 4,096 | 8.258 ms | 7.181 ms | 6.793 ms |
| 8,192 | 15.860 ms | 15.802 ms | 16.335 ms |
| 16,384 | 31.450 ms | 28.286 ms | 29.244 ms |

Against the MLX Q4 output, the ANE output was stable across the tested shapes:
max absolute error about `1.71484053e-3`, mean absolute error about
`2.087e-4`, and relative L2 error about `1.780e-3`.

The relative speed advantage versus MLX Q4 was approximately 35% at 2K,
13% at 4K, effectively break-even at 8K, and 10% at 16K. The `.all` path is
not consistently better than ANE-only, so dispatch should be benchmark-driven
per device and shape rather than assuming `.all` is optimal.

This is promising but not a production decision yet. The next gate is an
end-to-end long-prefill experiment that accounts for model compilation,
input/output boundary copies, scheduling, and fallback dispatch. The current
probe intentionally does not modify the production runtime.

## Next active direction

The isolated ANE advantage is real, but the single-projection bridge spends
most of it at the Python/Core ML boundary. The next experiment is therefore
stage-level rather than another single projection: measure one complete
prefill stage with a shared input/output boundary, then determine whether
MLP gate/up/down or a linear-attention projection group can amortize the
bridge cost. The existing single-projection bridge remains opt-in and is not
enabled by default.

## Runtime bridge result

The opt-in bridge is now implemented in:

- `tools/ane_projection_bridge.swift`
- `python/minimlx_runtime/ane.py`
- `python/minimlx_runtime/adapters/qwen35.py`

It is enabled only with `MINIMLX_ENABLE_ANE=1` and requires an explicit
compiled model, library, and fixed token shape. Unsupported shapes fall back
to MLX Q4. The first end-to-end test used the production 1024-token prefill
chunk and only L17 `in_proj_z`:

| Path | TTFT |
|---|---:|
| MLX Q4 baseline | 1995.1 ms |
| ANE bridge | 1987.0 ms |

The bridge was confirmed to execute. Its measured accumulated Core ML bridge
time was about 7–8 ms per request, which consumes most of the isolated kernel
advantage. This means the next useful design is phase-level fusion or a
zero-copy/shared-buffer boundary, not adding more individual ANE projection
calls. The opt-in path remains valuable as a correctness and integration
 baseline, but it is not enabled by default.

## GDN stage-level bridge experiment

The next experiment combines the four GatedDeltaNet input projections into one
fixed-shape Core ML model, keeping one Python/Core ML boundary per stage:

- `in_proj_qkv`: 6144 outputs
- `in_proj_z`: 2048 outputs
- `in_proj_b`: 16 outputs
- `in_proj_a`: 16 outputs
- combined output: 8224 values per token

The probe is implemented by `tools/build_ane_gdn_stage_probe.py` and
`tools/profile_ane_gdn_stage.py`. The comparison uses the same deterministic
input and measures the complete Python ctypes bridge, not only the ANE kernel.

| Tokens | MLX four Q4 projections | ANE kernel | Python bridge total | Net saving |
|---:|---:|---:|---:|---:|
| 1,024 | 10.053 ms | 5.086 ms | 8.744 ms | 1.309 ms (13.0%) |
| 4,096 | 37.576 ms | 18.090 ms | 32.057 ms | 5.519 ms (14.7%) |

The bridge remains faster at both sizes, and the absolute saving grows with
context length. This is the first ANE result that survives the actual Python
boundary with a material stage-level benefit. It is still an isolated result:
the next gate is strict opt-in GDN dispatch, splitting the combined output
into the four original tensors, then full-model correctness plus 1K/4K/16K/32K
prefill A/B tests. Decode continues to use MLX because this fixed-shape bridge
is not appropriate for a single-token request.

## Full-model dispatch result

The strict experimental dispatch was implemented and exercised on the real
Qwen3.5 benchmark. It correctly falls back for decode and unsupported shapes,
but the full model regressed despite the isolated stage win:

| Path | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| MLX baseline | 1853.3 ms | 5.89 ms | 2607.0 ms |
| GDN ANE stage, first attempt | 2313.3 ms | 8.50 ms | 3400.7 ms |
| GDN ANE stage, one shared dtype conversion | 2136.0 ms | 7.15 ms | 3050.7 ms |

The result identifies the real remaining cost: the Core ML output is produced
in host memory and then re-entered into the MLX graph. The isolated bridge
benchmark already includes one upload, but the full GDN graph adds dependency
synchronization and prevents the surrounding MLX operations from staying in
their normal fused/device-resident path. The stage dispatch therefore remains
opt-in research code and is not accepted as a production optimization.

The next ANE experiment is a shared/zero-copy device boundary (or a Core ML
stage that includes the consumer-side operations), not further Python slicing
or more individual ANE calls. Any future ANE path must beat the MLX baseline
in full-model TTFT and TPOT before it can be enabled.

## Compute-unit selection check

The bridge accepts `MINIMLX_ANE_COMPUTE_UNITS=ane|gpu|cpu` for controlled
comparison; the default remains `.all`. On the same 4K/128-token benchmark:

| Bridge mode | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| MLX baseline | 1853.3 ms | 5.89 ms | 2607.0 ms |
| Core ML `.all` | 2135.1 ms | 7.13 ms | 3048.0 ms |
| Core ML `.cpuAndNeuralEngine` | 2131.9 ms | 7.16 ms | 3048.5 ms |
| Core ML `.cpuAndGPU` | 2160.9 ms | 7.12 ms | 3072.7 ms |

Selecting the ANE explicitly changes almost nothing end to end, while GPU
selection is worse. This rules out Core ML compute-unit selection as the main
regression. The unresolved cost is the host/Core ML/MLX boundary and the
subsequent MLX graph synchronization. The next work item is therefore a
shared-buffer or consumer-fused boundary experiment; the stage bridge remains
opt-in and is not enabled by default.
