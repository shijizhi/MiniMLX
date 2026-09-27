#!/usr/bin/env python3
"""Independent GDN Q4 prefill projection experiment.

This is a research probe only.  It measures the existing MLX quantized
matmul against MiniMLX's independent correctness-first affine-Q4 Metal
kernel on the real Qwen3.5 GDN projections.  It does not patch mlx-lm and
does not copy oMLX implementation code; the motivation is only the public
shape/dispatch behavior observed in oMLX.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_q4_mlp_prefill import (
    custom_qmm,
    custom_simd_tiled_qmm,
    q4_affine_kernel,
    q4_affine_simd_tiled_kernel,
    timed,
)


PROJECTION_NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def projection_output(mixer):
    for name in ("out_proj", "o_proj", "output_projection"):
        candidate = getattr(mixer, name, None)
        if candidate is not None:
            return name, candidate
    fields = sorted(k for k in dir(mixer) if "proj" in k.lower())
    raise AttributeError(f"cannot find GDN output projection; fields={fields}")


def as_affine_layer(weight, scales, biases, source):
    return SimpleNamespace(
        weight=weight,
        scales=scales,
        biases=biases,
        group_size=source.group_size,
        bits=source.bits,
        mode=source.mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    layer = model.language_model.model.layers[args.layer]
    mixer = layer.linear_attn
    output_name, output_projection = projection_output(mixer)
    projections = [(name, getattr(mixer, name)) for name in PROJECTION_NAMES]
    projections.append((output_name, output_projection))

    input_width = projections[0][1].weight.shape[1] * 32 // projections[0][1].bits
    inputs = {}
    for name, projection in projections:
        width = projection.weight.shape[1] * 32 // projection.bits
        inputs[name] = mx.random.normal((1, args.tokens, width), dtype=mx.bfloat16)
    mx.eval(*inputs.values())
    kernel = q4_affine_kernel()
    tiled_kernel = q4_affine_simd_tiled_kernel()

    print("MiniMLX independent GDN Q4 prefill projection experiment")
    print(
        f"layer={args.layer}, tokens={args.tokens}, input_width={input_width}, "
        f"warmup={args.warmup}, repeats={args.repeats}"
    )
    print("research_only=true, production_dispatch=false")

    stock_outputs = {}
    custom_outputs = {}
    for name, projection in projections:
        x = inputs[name]
        stock = projection(x)
        custom = custom_qmm(kernel, x, projection)
        mx.eval(stock, custom)
        error = float(mx.max(mx.abs(stock.astype(mx.float32) - custom.astype(mx.float32))))
        stock_outputs[name] = stock
        custom_outputs[name] = custom
        stock_ms = timed(lambda p=projection: p(x), args.warmup, args.repeats)
        custom_ms = timed(
            lambda p=projection: custom_qmm(kernel, x, p), args.warmup, args.repeats
        )
        tiled = custom_simd_tiled_qmm(tiled_kernel, x, projection)
        mx.eval(tiled)
        tiled_error = float(mx.max(mx.abs(stock.astype(mx.float32) - tiled.astype(mx.float32))))
        tiled_ms = timed(
            lambda p=projection: custom_simd_tiled_qmm(tiled_kernel, x, p),
            args.warmup,
            args.repeats,
        )
        print(
            f"{name} shape={tuple(stock.shape)} max_abs_error={error:.8e} "
            f"stock={stock_ms:.3f} ms custom={custom_ms:.3f} ms "
            f"speedup={stock_ms / custom_ms:.3f}x "
            f"tiled_error={tiled_error:.8e} tiled={tiled_ms:.3f} ms "
            f"tiled_speedup={stock_ms / tiled_ms:.3f}x"
        )

    # The four input projections have the same input width, so also measure
    # the projection fusion opportunity independently from the output path.
    input_projections = projections[:4]
    first = input_projections[0][1]
    x = inputs[input_projections[0][0]]
    fused_weight = mx.concatenate([p.weight for _, p in input_projections], axis=0)
    fused_scales = mx.concatenate([p.scales for _, p in input_projections], axis=0)
    fused_biases = mx.concatenate([p.biases for _, p in input_projections], axis=0)
    fused = as_affine_layer(fused_weight, fused_scales, fused_biases, first)
    mx.eval(fused_weight, fused_scales, fused_biases)

    def stock_fused():
        return mx.quantized_matmul(
            x,
            fused_weight,
            scales=fused_scales,
            biases=fused_biases,
            transpose=True,
            group_size=first.group_size,
            bits=first.bits,
            mode=first.mode,
        )

    def custom_fused():
        return custom_qmm(kernel, x, fused)

    stock_fused_out = stock_fused()
    custom_fused_out = custom_fused()
    mx.eval(stock_fused_out, custom_fused_out)
    fused_error = float(
        mx.max(mx.abs(stock_fused_out.astype(mx.float32) - custom_fused_out.astype(mx.float32)))
    )
    stock_fused_ms = timed(stock_fused, args.warmup, args.repeats)
    custom_fused_ms = timed(custom_fused, args.warmup, args.repeats)
    print(
        f"input_fused shape={tuple(stock_fused_out.shape)} "
        f"max_abs_error={fused_error:.8e} stock={stock_fused_ms:.3f} ms "
        f"custom={custom_fused_ms:.3f} ms speedup={stock_fused_ms / custom_fused_ms:.3f}x"
    )


if __name__ == "__main__":
    main()
