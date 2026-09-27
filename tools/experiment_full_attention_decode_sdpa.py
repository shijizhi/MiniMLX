#!/usr/bin/env python3
"""Measure the real Qwen3.5 cached-decode SDPA dispatch choices.

This is an isolated kernel experiment.  It intentionally excludes Q/K/V
projection, RoPE, KV-cache append, output projection, and the decoder loop.
The cached one-token path uses mask=None in mlx-lm, so that is the production
shape measured here.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load


def timed(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--context", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    full_layer = next(
        layer for layer in model.language_model.model.layers if hasattr(layer, "self_attn")
    )
    attn = full_layer.self_attn
    contexts = args.context or [1024, 4096, 16384, 32768]
    dtype = mx.bfloat16

    print("MiniMLX full-attention cached-decode SDPA experiment")
    print(
        f"q_heads={attn.num_attention_heads}, kv_heads={attn.num_key_value_heads}, "
        f"head_dim={attn.head_dim}, mask=None, dtype={dtype}, "
        f"warmup={args.warmup}, repeats={args.repeats}"
    )

    for context in contexts:
        if context <= 0:
            raise SystemExit("context must be positive")
        queries = mx.random.normal(
            (1, attn.num_attention_heads, 1, attn.head_dim), dtype=dtype
        )
        keys = mx.random.normal(
            (1, attn.num_key_value_heads, context, attn.head_dim), dtype=dtype
        )
        values = mx.random.normal(
            (1, attn.num_key_value_heads, context, attn.head_dim), dtype=dtype
        )
        mx.eval(queries, keys, values)

        def stock():
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=attn.scale,
                mask=None,
                force_fused=False,
            )

        def fused():
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=attn.scale,
                mask=None,
                force_fused=True,
            )

        reference = stock()
        candidate = fused()
        mx.eval(reference, candidate)
        error = mx.max(mx.abs(reference - candidate)).item()
        stock_ms = timed(stock, args.warmup, args.repeats)
        fused_ms = timed(fused, args.warmup, args.repeats)
        print(
            f"context={context} stock={stock_ms:.3f} ms "
            f"force_fused={fused_ms:.3f} ms speedup={stock_ms / fused_ms:.3f}x "
            f"max_abs_error={error:.6e}"
        )


if __name__ == "__main__":
    main()
