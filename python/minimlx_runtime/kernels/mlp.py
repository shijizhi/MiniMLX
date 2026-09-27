"""Inference-only Q4 MLP prefill fusion for Qwen3.5-shaped MLX modules."""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn


def _qwen35_mlp_types():
    """Return the MLP classes used by the loaded Qwen3.5 implementation.

    Qwen3.5 is not Qwen3-Next: mlx-lm defines its decoder MLP as
    ``mlx_lm.models.qwen3_5.MLP``.  Keeping this lookup in one place avoids
    installing a patch on an unrelated architecture class.
    """
    types = []
    try:
        from mlx_lm.models import qwen3_5

        types.append(qwen3_5.MLP)
    except (ImportError, AttributeError):
        pass
    # The MTP target is loaded by mlx-vlm, whose Qwen3.5 decoder uses a
    # separate class with the same gate/up/down contract. OMLX patches both
    # implementations; include the VLM class so MiniMLX's opt-in Q4 MLP path
    # is available to the MTP target as well.
    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

        if Qwen3_5MLP not in types:
            types.append(Qwen3_5MLP)
    except (ImportError, AttributeError):
        pass
    # Compatibility with older mlx-lm model layouts.
    try:
        from mlx_lm.models import qwen3_next

        if qwen3_next.Qwen3NextMLP not in types:
            types.append(qwen3_next.Qwen3NextMLP)
    except (ImportError, AttributeError):
        pass
    return types


def install_q4_mlp_prefill_fusion(model, min_tokens: int = 2048) -> bool:
    """Fuse Qwen MLP gate/up Q4 projections for long prefill inputs.

    The fused path still uses MLX's official quantized_matmul.  It only joins
    two projections with identical input and quantization metadata into one
    output matrix.  Decode and short inputs use the original implementation.
    """
    if os.environ.get("MINIMLX_DISABLE_Q4_MLP_FUSION") == "1":
        return False

    layers = getattr(getattr(model, "language_model", None), "model", None)
    layers = getattr(layers, "layers", None)
    if layers is None:
        return False

    installed = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        if gate is None or up is None:
            continue
        required = ("weight", "scales", "biases", "group_size", "bits", "mode")
        if any(not hasattr(gate, name) or not hasattr(up, name) for name in required):
            continue
        metadata = (gate.group_size, gate.bits, gate.mode)
        if metadata != (up.group_size, up.bits, up.mode):
            continue
        if gate.biases is None or up.biases is None:
            continue
        if gate.weight.shape[1] != up.weight.shape[1]:
            continue
        if gate.scales.shape[1] != up.scales.shape[1]:
            continue
        mlp._minimlx_fused_gate_up = (
            mx.concatenate([gate.weight, up.weight], axis=0),
            mx.concatenate([gate.scales, up.scales], axis=0),
            mx.concatenate([gate.biases, up.biases], axis=0),
            gate.weight.shape[0],
            metadata,
            int(min_tokens),
        )
        installed += 1

    if installed == 0:
        return False

    for mlp_type in _qwen35_mlp_types():
        if getattr(mlp_type, "_minimlx_q4_fusion_installed", False):
            continue
        stock_call = mlp_type.__call__

        def fused_call(self, x, _stock_call=stock_call):
            spec = getattr(self, "_minimlx_fused_gate_up", None)
            if spec is not None and not self.training and x.ndim >= 2:
                weight, scales, biases, split, metadata, threshold = spec
                if x.shape[-2] >= threshold:
                    dense_spec = getattr(self, "_minimlx_prefill_dense", None)
                    if dense_spec is not None and x.shape[-2] >= dense_spec[3]:
                        return _stock_call(self, x)
                    gate_up = mx.quantized_matmul(
                        x,
                        weight,
                        scales=scales,
                        biases=biases,
                        transpose=True,
                        group_size=metadata[0],
                        bits=metadata[1],
                        mode=metadata[2],
                    )
                    gate, up = gate_up[..., :split], gate_up[..., split:]
                    return self.down_proj(nn.silu(gate) * up)
            return _stock_call(self, x)

        mlp_type.__call__ = fused_call
        mlp_type._minimlx_q4_fusion_installed = True
    return True


def install_q4_mlp_prefill_dense(model, min_tokens: int = 1024) -> bool:
    """Use cached BF16 weights for large Q4 prefill MLPs.

    This is intentionally separate from the Q4 fusion path.  Quantized
    weights remain the production decode representation; the dense copies are
    used only when the sequence dimension is large enough for prefill GEMM to
    amortize the extra memory footprint.
    """
    if os.environ.get("MINIMLX_DISABLE_Q4_PREFILL_DENSE") == "1":
        return False

    layers = getattr(getattr(model, "language_model", None), "model", None)
    layers = getattr(layers, "layers", None)
    if layers is None:
        return False

    cached = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if gate is None or up is None or down is None:
            continue
        projections = (gate, up, down)
        if any(
            getattr(proj, "mode", None) != "affine"
            or getattr(proj, "biases", None) is None
            or not hasattr(proj, "weight")
            for proj in projections
        ):
            continue
        gate_weight, up_weight, down_weight = (
            mx.dequantize(
                proj.weight,
                proj.scales,
                proj.biases,
                group_size=proj.group_size,
                bits=proj.bits,
                mode=proj.mode,
                dtype=mx.bfloat16,
            )
            for proj in projections
        )
        # Gate and up consume the same activation. Concatenating their output
        # rows turns two prefill GEMMs into one larger GEMM.
        gate_up_weight = mx.concatenate([gate_weight, up_weight], axis=0)
        mlp._minimlx_prefill_dense = (
            gate_up_weight,
            gate_weight.shape[0],
            down_weight,
            int(min_tokens),
        )
        cached.extend((gate_up_weight, down_weight))

    if not cached:
        return False
    mx.eval(*cached)

    for mlp_type in _qwen35_mlp_types():
        if getattr(mlp_type, "_minimlx_prefill_dense_installed", False):
            continue
        stock_call = mlp_type.__call__

        def dense_call(self, x, _stock_call=stock_call):
            spec = getattr(self, "_minimlx_prefill_dense", None)
            if spec is not None and not self.training and x.ndim >= 2:
                gate_up_w, split, down_w, threshold = spec
                if x.shape[-2] >= threshold:
                    x_dense = x.astype(mx.bfloat16)
                    gate_up = x_dense @ gate_up_w.T
                    gate, up = gate_up[..., :split], gate_up[..., split:]
                    return (nn.silu(gate) * up) @ down_w.T
            return _stock_call(self, x)

        mlp_type.__call__ = dense_call
        mlp_type._minimlx_prefill_dense_installed = True
    return True
