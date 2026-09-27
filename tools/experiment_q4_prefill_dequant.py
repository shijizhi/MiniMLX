#!/usr/bin/env python3
"""Compare stock affine-Q4 QMM with prefill-only dequantize + dense matmul."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def timed(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        values.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    mlp = model.language_model.model.layers[0].mlp
    gate = mlp.gate_proj
    hidden = gate.weight.shape[1] * 32 // gate.bits
    x = mx.random.normal((1, args.tokens, hidden), dtype=mx.bfloat16)
    down_x = mx.random.normal(
        (1, args.tokens, mlp.down_proj.weight.shape[1] * 32 // mlp.down_proj.bits),
        dtype=mx.bfloat16,
    )
    mx.eval(x, down_x)

    weights = {}
    for name, layer in (("gate", mlp.gate_proj), ("up", mlp.up_proj), ("down", mlp.down_proj)):
        weights[name] = {
            dtype: mx.dequantize(
                layer.weight,
                layer.scales,
                layer.biases,
                group_size=layer.group_size,
                bits=layer.bits,
                mode=layer.mode,
                dtype=dtype,
            )
            for dtype in (mx.bfloat16, mx.float16)
        }
    mx.eval(*(value for item in weights.values() for value in item.values()))
    gate_up_bf16 = mx.concatenate(
        [weights["gate"][mx.bfloat16], weights["up"][mx.bfloat16]], axis=0
    )
    mx.eval(gate_up_bf16)

    def stock(layer, value):
        return lambda: mx.quantized_matmul(
            value,
            layer.weight,
            scales=layer.scales,
            biases=layer.biases,
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )

    def dense(name, dtype, value):
        weight = weights[name][dtype]
        return lambda: value.astype(dtype) @ weight.T

    def dense_gate_up():
        gate_up = x @ gate_up_bf16.T
        split = mlp.gate_proj.weight.shape[0]
        return gate_up[..., :split], gate_up[..., split:]

    print(
        "MiniMLX Q4 prefill dequant A/B "
        f"(L0, tokens={args.tokens}, warmup={args.warmup}, repeats={args.repeats})"
    )
    print(f"input={x.shape}, gate_weight={gate.weight.shape}, group_size={gate.group_size}")
    for name, layer, value in (
        ("gate", mlp.gate_proj, x),
        ("up", mlp.up_proj, x),
        ("down", mlp.down_proj, down_x),
    ):
        stock_ms = timed(stock(layer, value), args.warmup, args.repeats)
        bf16_ms = timed(dense(name, mx.bfloat16, value), args.warmup, args.repeats)
        f16_ms = timed(dense(name, mx.float16, value), args.warmup, args.repeats)
        print(f"{name}_stock_qmm={stock_ms:.3f} ms")
        print(f"{name}_dequant_bf16_dense={bf16_ms:.3f} ms")
        print(f"{name}_dequant_f16_dense={f16_ms:.3f} ms")

    gate_up = dense_gate_up()
    mx.eval(*gate_up)
    print(f"gate_up_dequant_bf16_dense={timed(dense_gate_up, args.warmup, args.repeats):.3f} ms")

    stock_gate_up = mx.concatenate(
        [
            stock(mlp.gate_proj, x)(),
            stock(mlp.up_proj, x)(),
        ],
        axis=-1,
    )
    dense_gate_up_value = mx.concatenate(gate_up, axis=-1)
    mx.eval(stock_gate_up, dense_gate_up_value)
    print(
        "gate_up_bf16_max_abs_error="
        f"{float(mx.max(mx.abs(stock_gate_up.astype(mx.float32) - dense_gate_up_value.astype(mx.float32)))):.8e}"
    )

    def stock_mlp():
        return mlp.down_proj(nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x))

    def dense_mlp(dtype):
        gate_w = weights["gate"][dtype]
        up_w = weights["up"][dtype]
        down_w = weights["down"][dtype]
        hidden_value = nn.silu(x.astype(dtype) @ gate_w.T) * (x.astype(dtype) @ up_w.T)
        return hidden_value @ down_w.T

    stock_out = stock_mlp()
    dense_out = dense_mlp(mx.bfloat16)
    mx.eval(stock_out, dense_out)
    print(f"mlp_stock_qmm={timed(stock_mlp, args.warmup, args.repeats):.3f} ms")
    print(
        "mlp_dequant_bf16_dense="
        f"{timed(lambda: dense_mlp(mx.bfloat16), args.warmup, args.repeats):.3f} ms"
    )
    print(
        "mlp_bf16_max_abs_error="
        f"{float(mx.max(mx.abs(stock_out.astype(mx.float32) - dense_out.astype(mx.float32)))):.8e}"
    )

    stock_gate = stock(mlp.gate_proj, x)()
    dense_gate = dense("gate", mx.bfloat16, x)()
    mx.eval(stock_gate, dense_gate)
    print(
        "gate_bf16_max_abs_error="
        f"{float(mx.max(mx.abs(stock_gate.astype(mx.float32) - dense_gate.astype(mx.float32)))):.8e}"
    )


if __name__ == "__main__":
    main()
