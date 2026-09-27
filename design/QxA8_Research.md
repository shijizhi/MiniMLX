# QxA8 prefill research

## Current evidence

MLX exposes `quantized_matmul(x, w, scales, biases, ...)`, but its public
contract is a floating-point activation multiplied by a packed quantized
weight. MLX-LM's activation-quantization path does not provide an arbitrary
W4A8 route: it converts supported `mxfp8`/`nvfp4` modules to `QQLinear`, and
rejects other quantization modes for activation quantization.

The current model uses affine Q4 weights with BF16/FP32 activations. Therefore
the first experiment must not assume that an INT8 activation path is already
available in MLX. It needs a capability gate and an independent kernel.

Primary references:

- MLX `quantized_matmul` API:
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul
- MLX-LM activation quantization conversion:
  https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py
- MLX numerical precision notes:
  https://ml-explore.github.io/mlx/build/html/usage/precision.html
- MLX quantized matmul performance discussion, including scale/bias bandwidth:
  https://github.com/ml-explore/mlx/issues/3251

## Experiment boundary

Compare the existing Q4 projection against an independent W4A8 implementation
only at supported prefill shapes. Measure dequantization, activation
quantization, matrix multiply, and the combined path separately. Require the
same output error gate, short/4K/16K coverage, and no decode dispatch change.
If activation quantization plus conversion is slower than MLX Q4 QMM, retain
the capability probe only and do not add a production route.
