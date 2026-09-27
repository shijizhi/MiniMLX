# QxA8 prefill research

## Initial conclusion

QxA8/W4A8 is a credible prefill direction, but it is not an immediate drop-in MLX feature for the current M4 target. Current MLX Python quantization exposes activation quantization only for `nvfp4` and `mxfp8`; the ordinary affine Q4 `quantized_matmul` path is weight-only. Therefore a W4A8 path would require a separately validated Metal kernel and a hardware capability gate, rather than changing the existing QMM call.

## Evidence

- MLX's quantized layer documentation explicitly limits `quantize_input=True` to `nvfp4` and `mxfp8`: <https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py>
- QQQ studies 4-bit weights plus 8-bit activations as a way to improve compute-bound prefill, while noting that weight-only and activation-quantized paths have different tradeoffs: <https://arxiv.org/abs/2406.09904>
- QServe describes W4A8 as a system co-design option, but its gains depend on a hardware-native integer path and calibrated scales: <https://arxiv.org/abs/2405.04532>
- Public oMLX discussion reports an INT8-activation prefill kernel specifically on newer Apple hardware with Neural Accelerators, not as a universal MLX fallback: <https://www.reddit.com/r/oMLX/comments/1wemv9q/i_added_int8activation_prefill_kernels_to_omlx/>

## Implication for MiniMLX

The local capability probe confirms that this MLX build can quantize tensors as affine INT8, MXFP8, and NVFP4 with their required group sizes, but all results are packed weight-style tensors (`uint32` payloads). `mx.quantized_matmul` exposes no separate activation-quantization input or W4A8 mode, and `mx.fast` exposes no ready-made integer matmul symbol. Thus these modes do not provide a reusable W4A8 implementation by themselves.

The next experiment must answer three separate questions:

1. Does this Mac expose a usable native INT8 matrix path for the relevant Metal/MLX version?
2. Can Q4 weights and per-token/per-group activation scales be consumed without an extra conversion kernel dominating the saved GEMM time?
3. Does the approximation preserve model outputs sufficiently for all supported model adapters, not only Qwen3.5?

The implementation should therefore be an opt-in experimental kernel with a strict predicate over device capability, quantization metadata, matrix shapes, sequence length, and dtype. Decode must continue to use the existing Q4 path. If the hardware gate is false, no activation quantization or extra copy is performed.

## Planned experiment

Build a standalone W4A8-vs-BF16-Q4-dequant benchmark for the exact prefill shapes already collected in `Q4_Prefill_Research.md` (M=1024, 4096, 16K, 32K). Measure separately:

- activation quantization and scale generation;
- W4A8 matrix kernel;
- output dequantization/rescaling;
- end-to-end MLP projection;
- max error against the BF16 dense reference.

Do not integrate it into the default runtime until the complete pipeline is faster than the existing BF16 dense prefill path and has a non-regression fallback.

## Status

`[research]` Public evidence supports investigating W4A8, but current MLX API support and M4 hardware availability are not yet sufficient to claim a viable implementation. The first action is a capability/shape microbenchmark, not a runtime change.
