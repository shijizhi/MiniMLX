# ANE consumer-fusion experiment

## Why this is the next experiment

The four-projection ANE stage is faster in isolation, but full-model dispatch
is slower because its result crosses from Core ML host memory back into the
MLX graph. Changing Core ML compute units does not remove that regression:

| Path | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| MLX baseline | 1853.3 ms | 5.89 ms | 2607.0 ms |
| Core ML `.all` | 2135.1 ms | 7.13 ms | 3048.0 ms |
| Core ML `.cpuAndNeuralEngine` | 2131.9 ms | 7.16 ms | 3048.5 ms |
| Core ML `.cpuAndGPU` | 2160.9 ms | 7.12 ms | 3072.7 ms |

The next prototype must therefore reduce the tensor crossing the boundary,
not only make the projection kernel faster.

## Proposed fixed-shape model

For a prefill-only GDN stage, the Core ML model should consume the normalized
hidden states and the four projection weights, then perform:

1. QKV convolution and SiLU.
2. Q/K reshaping, normalization, and scaling.
3. Gated-delta state recurrence over the fixed token axis.
4. Gated RMS normalization using `z`.
5. Output projection.

The external output should be only the stage result with shape
`[1, tokens, hidden_size]` plus the final recurrent state if the caller needs
to retain it. The current bridge returns 8224 values per token; the fused
stage should return 1024 values per token. This is the key boundary reduction.

## Implementation order

1. Build a numerical MIL prototype for the recurrence with explicit loop-carried
   state and compare it against the existing MLX GDN reference.
2. Add convolution, normalization, and output projection.
3. Compile fixed shapes for 1024 and 4096 tokens and verify Core ML output
   against MLX with a documented tolerance.
4. Measure Core ML kernel time and complete bridge time separately.
5. Integrate only as a strict long-prefill opt-in. Decode and unsupported
   shapes remain on MLX.
6. Accept the path only if full-model TTFT and TPOT beat the MLX baseline;
   otherwise retain the prototype as research code.

Core ML tooling provides loop/control-flow primitives for this design, but
their ANE performance and numerical behavior must be measured rather than
assumed. This is an independent implementation direction and does not copy
oMLX.

## Recurrence prototype result

The first MIL `while_loop` prototype is now validated at two sizes:

| Shape | Core ML time | State max error | Output max error |
|---|---:|---:|---:|
| 8 tokens, 2 heads, 8x4 | 0.107 ms | 0 | 1.86e-9 |
| 16 tokens, 16 heads, 128x128 | 7.583 ms | 1.34e-7 | 1.79e-7 |

The reference is an independent NumPy sequential recurrence. The real-size
case compiled successfully with `xcrun coremlcompiler` and executed through
the Swift Core ML runner. This proves that fixed-shape Core ML control flow
can represent the recurrence accurately; it does not yet prove an end-to-end
speedup. The next work item is to add the GDN normalization and output
projection inside the same model, then compare the complete consumer-fused
stage against the existing MLX path.
## Consumer-fusion prototype result

The complete consumer prototype now keeps the recurrent sequence in Core ML,
applies gated RMS normalization, and performs the output projection before
returning to the caller. At 16 tokens, 16 heads, 128 key/value dimensions,
and hidden size 1024, it compiled successfully and executed in `8.930 ms`.
Against an independent NumPy reference, maximum absolute errors were
`7.45e-7` for the projected sequence and `1.34e-7` for the final state.

This proves the complete fused consumer path is representable and numerically
correct. It remains a research prototype with float32 weights and synthetic
inputs; the next measurement replaces those with the model's actual q/k/v/gate
tensors and dequantized output matrix, then measures bridge-inclusive time at
1024 and 4096 prefill lengths.

## Real output-weight validation

The model-specific output projection was then extracted from layer 17 of the
Qwen3.5 MLX checkpoint and dequantized into the consumer model's expected
`[2048, 1024]` matrix. The same synthetic recurrence inputs were retained so
that this isolates the weight path from model-input capture. The compiled
consumer model ran in `6.872 ms`; maximum absolute errors against the NumPy
reference were `3.58e-7` for the projected output and `1.34e-7` for the final
state.

This removes the remaining concern that the consumer prototype only worked
with random weights. It still is not an end-to-end result: the q/k/v/gate
inputs are synthetic, and the bridge-inclusive cost and full model integration
must be measured before enabling any production dispatch.

## Real GDN tensor validation

The input generator was then changed to run the actual layer-17 Qwen3.5 GDN
projections, depthwise convolution, SiLU, Q/K RMS normalization and
`A_log`/`dt_bias` gate construction. The hidden input is deterministic rather
than captured from a full model request, so this test isolates the real GDN
parameter and layout path without introducing tokenization or unrelated layer
state. The output projection remains the dequantized layer-17 `out_proj`.

For 16 tokens, the Core ML consumer result matched the independent NumPy
reference with:

| Result | Maximum absolute error | Mean absolute error |
|---|---:|---:|
| Projected output | 3.54e-8 | 2.78e-9 |
| Final recurrent state | 1.49e-8 | 9.09e-11 |

The compiled consumer execution was `9.495 ms` for this real-parameter input.
This closes the numerical-validation gap for the real GDN tensor path. It is
still not an optimization decision: the next experiment must build matching
1024- and 4096-token consumer shapes and compare Core ML kernel time,
bridge-inclusive time, and full-model TTFT against the MLX baseline.

## Long-sequence decision

The 1024-token experiment provides that first scale check. With the same real
layer-17 GDN-derived tensors and the real dequantized output projection, the
consumer model remained numerically aligned (`1.19e-7` output max error and
`7.45e-8` state max error), but took `1099.781 ms`. The recurrence-only model
at the same shape took `337.603 ms`.

Therefore the current Core ML `while_loop` implementation is not a viable
long-prefill optimization. The slowdown is inside the Core ML execution
strategy for the recurrent loop and its dense consumer, not merely in the
Python/Swift bridge. Do not enable this path in production. Retain the probe
as a correctness fixture, and redirect optimization work to algorithms or
kernels that preserve MLX's efficient batched execution rather than expanding
the recurrence into a Core ML control-flow graph.
