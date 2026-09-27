#!/usr/bin/env python3
"""Stock MLX FA-256 attention baseline for the first custom-kernel study."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx


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
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--force-fused", action="store_true")
    args = parser.parse_args()

    q = mx.random.normal((1, args.q_heads, args.tokens, args.head_dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, args.kv_heads, args.tokens, args.head_dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, args.kv_heads, args.tokens, args.head_dim)).astype(mx.bfloat16)
    mx.eval(q, k, v)

    def attention():
        return mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=args.head_dim**-0.5,
            mask="causal",
            force_fused=args.force_fused,
        )

    output = attention()
    mx.eval(output)
    fused_output = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=args.head_dim**-0.5, mask="causal", force_fused=True
    )
    mx.eval(fused_output)
    fused_error = float(
        mx.max(mx.abs(output.astype(mx.float32) - fused_output.astype(mx.float32)))
    )
    elapsed = timed(attention, args.warmup, args.repeats)
    print("MiniMLX FA-256 stock SDPA baseline")
    print(
        f"tokens={args.tokens}, q_heads={args.q_heads}, kv_heads={args.kv_heads}, "
        f"head_dim={args.head_dim}, dtype={q.dtype}"
    )
    print(f"stock_sdpa={'force_fused' if args.force_fused else 'default'}={elapsed:.3f} ms")
    print(f"default_vs_force_fused_max_abs_error={fused_error:.8e}")
    print(f"output_shape={list(output.shape)}")


if __name__ == "__main__":
    main()
