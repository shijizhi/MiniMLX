#!/usr/bin/env python3
"""Validate the isolated MLX C++ Q4 extension against a real Qwen3.5 layer."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

from minimlx_q4_native import affine_qmm  # noqa: E402


def median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def timed(fn, repeats: int = 3, evaluate: bool = True) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        if evaluate:
            mx.eval(value)
        samples.append((time.perf_counter() - start) * 1000.0)
    return median(samples)


def main() -> None:
    config = json.loads((ROOT / "config" / "minimlx.local.json").read_text())
    model, _ = load(config["model_dir"])
    layers = model.language_model.model.layers
    cases = []
    for layer_index, layer in enumerate(layers):
        if layer.is_linear:
            mixer = layer.linear_attn
            cases.extend(
                (f"L{layer_index}.linear.{name}", getattr(mixer, name))
                for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
                if getattr(mixer, name, None) is not None
            )
            break
    for layer_index, layer in enumerate(layers):
        if not layer.is_linear:
            attention = layer.self_attn
            cases.extend(
                (f"L{layer_index}.full.{name}", getattr(attention, name))
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                if getattr(attention, name, None) is not None
            )
            break
    mlp = layers[0].mlp
    cases.extend(
        (f"L0.mlp.{name}", getattr(mlp, name))
        for name in ("gate_proj", "up_proj", "down_proj")
        if getattr(mlp, name, None) is not None
    )
    projection = cases[0][1]

    print("MiniMLX native Q4 extension probe")
    print(
        f"bits={projection.bits} group_size={projection.group_size} "
        f"mode={projection.mode} output_dim={projection.weight.shape[0]}"
    )
    print("production_dispatch=false")
    print("implementation=minimlx_adapted_q4_primitive")

    for case_name, projection in cases:
        input_width = projection.weight.shape[1] * 32 // projection.bits
        print(f"case={case_name} input_dim={input_width} output_dim={projection.weight.shape[0]}")
        output_width = int(projection.weight.shape[0])
        if input_width % 64 or input_width % 32 or output_width % 64:
            print("fallback=shape_gate")
            continue
        for tokens in (2048, 4096, 16384):
            x = mx.random.normal((1, tokens, input_width), dtype=mx.bfloat16)
            mx.eval(x)
            stock = projection(x)
            mx.eval(stock)
            native = affine_qmm(
                x, projection.weight, projection.scales, projection.biases, 0
            )
            mx.eval(native)
            error = float(
                mx.max(mx.abs(stock.astype(mx.float32) - native.astype(mx.float32)))
            )
            if case_name == cases[0][0] and tokens == 2048:
                delta = mx.abs(stock.astype(mx.float32) - native.astype(mx.float32))
                flat_delta = delta.reshape(-1)
                flat_stock = stock.astype(mx.float32).reshape(-1)
                flat_native = native.astype(mx.float32).reshape(-1)
                max_index = int(mx.argmax(flat_delta))
                print(
                    "diagnostic="
                    f"flat_index={max_index} "
                    f"stock={float(flat_stock[max_index]):.8e} "
                    f"native={float(flat_native[max_index]):.8e} "
                    f"stock_abs_max={float(mx.max(mx.abs(flat_stock))):.8e} "
                    f"native_abs_max={float(mx.max(mx.abs(flat_native))):.8e}"
                )
            stock_ms = timed(lambda p=projection: p(x))
            native_ms = timed(
                lambda: affine_qmm(
                    x, projection.weight, projection.scales, projection.biases, 0
                )
            )
            print(
                f"tokens={tokens} max_abs_error={error:.8e} "
                f"stock={stock_ms:.3f} ms minimlx_native={native_ms:.3f} ms"
            )


if __name__ == "__main__":
    main()
