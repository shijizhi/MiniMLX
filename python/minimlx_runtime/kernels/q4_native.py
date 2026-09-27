"""Strictly gated MiniMLX Q4 prefill dispatch.

The native primitive is intentionally narrower than MLX's general quantized
linear implementation. Unsupported projections and decode calls remain on
the original MLX path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _load_native_qmm():
    root = Path(__file__).resolve().parents[3]
    native_root = root / "native"
    if str(native_root) not in sys.path:
        sys.path.insert(0, str(native_root))
    from minimlx_q4_native import affine_qmm

    return affine_qmm


def _eligible(projection: Any) -> bool:
    return (
        getattr(projection, "bits", None) == 4
        and getattr(projection, "group_size", None) == 64
        and getattr(projection, "mode", None) == "affine"
        and getattr(projection, "biases", None) is not None
        and getattr(projection, "scales", None) is not None
        and getattr(projection, "weight", None) is not None
        and projection.weight.shape[0] % 64 == 0
        and projection.weight.shape[1] * 8 % 64 == 0
    )


def install_q4_native_prefill(model, min_tokens: int = 2048) -> int:
    """Enable the MiniMLX Q4 primitive on eligible loaded projections."""
    if os.environ.get("MINIMLX_ENABLE_Q4_NATIVE_PREFILL") != "1":
        return 0

    affine_qmm = _load_native_qmm()
    layers = getattr(getattr(model, "language_model", None), "model", None)
    layers = getattr(layers, "layers", None)
    if layers is None:
        return 0

    installed = 0
    projection_type = None
    for layer in layers:
        candidates = []
        mixer = getattr(layer, "linear_attn", None)
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        for owner, names in (
            (mixer, ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")),
            (attention, ("q_proj", "k_proj", "v_proj", "o_proj")),
            (mlp, ("gate_proj", "up_proj", "down_proj")),
        ):
            if owner is not None:
                candidates.extend(getattr(owner, name, None) for name in names)
        for projection in candidates:
            if projection is None or not _eligible(projection):
                continue
            projection._minimlx_q4_native_prefill = True
            projection._minimlx_q4_native_min_tokens = int(min_tokens)
            projection._minimlx_q4_native_affine_qmm = affine_qmm
            projection_type = type(projection)
            installed += 1
    if projection_type is None:
        return 0
    if getattr(projection_type, "_minimlx_q4_native_installed", False):
        return installed

    stock_call = projection_type.__call__

    def dispatch(projection, value, *args, **kwargs):
        if (
            getattr(projection, "_minimlx_q4_native_prefill", False)
            and not args
            and not kwargs
            and getattr(value, "ndim", 0) == 3
            and value.shape[-2]
            >= getattr(projection, "_minimlx_q4_native_min_tokens", min_tokens)
        ):
            return projection._minimlx_q4_native_affine_qmm(
                value,
                projection.weight,
                projection.scales,
                projection.biases,
                8,
            )
        return stock_call(projection, value, *args, **kwargs)

    projection_type.__call__ = dispatch
    projection_type._minimlx_q4_native_installed = True
    return installed


def install_q4_native_mlp(model, min_tokens: int = 2048) -> bool:
    """Route the Qwen MLP through direct QMM calls for long prefill.

    oMLX patches the model MLP itself: gate/up QMM, SiLU*up, then down QMM.
    Keeping that boundary here avoids routing the three projections through
    the generic QuantizedLinear dispatcher and mirrors the reference call
    graph without copying its implementation.
    """
    if os.environ.get("MINIMLX_ENABLE_Q4_NATIVE_PREFILL") != "1":
        return False

    affine_qmm = _load_native_qmm()
    layers = getattr(getattr(model, "language_model", None), "model", None)
    layers = getattr(layers, "layers", None)
    if layers is None:
        return False

    installed = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(mlp, name, None)
            if projection is not None and _eligible(projection):
                projection._minimlx_q4_native_affine_qmm = affine_qmm
                installed += 1

    if installed == 0:
        return False

    from mlx import nn
    from .mlp import _qwen35_mlp_types

    mlp_types = _qwen35_mlp_types()
    if not mlp_types:
        return False
    installed_types = 0
    for mlp_type in mlp_types:
        if getattr(mlp_type, "_minimlx_q4_native_mlp_installed", False):
            installed_types += 1
            continue
        stock_call = mlp_type.__call__

        def native_call(self, value, _stock_call=stock_call):
            if value.ndim >= 3 and value.shape[-2] >= min_tokens and not self.training:
                gate = getattr(self, "gate_proj", None)
                up = getattr(self, "up_proj", None)
                down = getattr(self, "down_proj", None)
                if all(
                    projection is not None
                    and getattr(projection, "_minimlx_q4_native_affine_qmm", None)
                    is not None
                    for projection in (gate, up, down)
                ):
                    if not getattr(type(self), "_minimlx_q4_native_mlp_logged", False):
                        print(
                            "[mlp-native] route "
                            f"x={tuple(value.shape)} gate_up="
                            f"({gate.weight.shape[0] * 2}) down={down.weight.shape[0]}",
                            flush=True,
                        )
                        type(self)._minimlx_q4_native_mlp_logged = True
                    qmm = gate._minimlx_q4_native_affine_qmm
                    gate_value = qmm(
                        value, gate.weight, gate.scales, gate.biases, 8
                    )
                    up_value = qmm(
                        value, up.weight, up.scales, up.biases, 8
                    )
                    hidden = nn.silu(gate_value) * up_value
                    return qmm(
                        hidden, down.weight, down.scales, down.biases, 8
                    )
            return _stock_call(self, value)

        mlp_type.__call__ = native_call
        mlp_type._minimlx_q4_native_mlp_installed = True
        installed_types += 1
    return installed_types > 0
