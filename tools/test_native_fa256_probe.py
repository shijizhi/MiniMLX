#!/usr/bin/env python3
"""Standalone correctness/performance probe for MiniMLX FA-256."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native"))
from minimlx_q4_native import fa256_attention


def timed(fn, warmup: int = 2, repeats: int = 5) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        mx.eval(value)
        samples.append((time.perf_counter() - start) * 1000.0)
    return sorted(samples)[len(samples) // 2]


def main() -> None:
    print("MiniMLX native FA-256 probe")
    print("dtype=bfloat16 q_heads=8 kv_heads=2 head_dim=256 causal=true")
    q0 = mx.zeros((1, 8, 32, 256), dtype=mx.bfloat16)
    k0 = mx.zeros((1, 2, 32, 256), dtype=mx.bfloat16)
    v0 = mx.ones((1, 2, 32, 256), dtype=mx.bfloat16)
    invariant = fa256_attention(q0, k0, v0, 256 ** -0.5)
    mx.eval(invariant)
    invariant_error = mx.max(mx.abs(invariant.astype(mx.float32) - 1.0))
    mx.eval(invariant_error)
    print(f"invariant_tokens=32 max_abs_error={float(invariant_error.item()):.8e}")
    for query_tokens, cache_tokens in ((32, 32), (64, 64), (1024, 1024), (1024, 2048), (1024, 4096)):
        q = mx.random.normal((1, 8, query_tokens, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, cache_tokens, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, cache_tokens, 256)).astype(mx.bfloat16)
        scale = 256 ** -0.5

        stock = lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask="causal"
        )
        fused = lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask="causal", force_fused=True
        )
        native = lambda: fa256_attention(q, k, v, scale)
        reference = stock()
        candidate = native()
        fused_value = fused()
        mx.eval(reference, candidate, fused_value)
        error = mx.max(mx.abs(reference.astype(mx.float32) - candidate.astype(mx.float32)))
        fused_error = mx.max(mx.abs(reference.astype(mx.float32) - fused_value.astype(mx.float32)))
        stock_peak = mx.max(mx.abs(reference.astype(mx.float32)))
        native_peak = mx.max(mx.abs(candidate.astype(mx.float32)))
        mx.eval(error, fused_error, stock_peak, native_peak)
        stock_ms = timed(stock)
        native_ms = timed(native)
        print(
            f"query={query_tokens} cache={cache_tokens} "
            f"max_abs_error={float(error.item()):.8e} "
            f"stock_fused_error={float(fused_error.item()):.8e} "
            f"stock_peak={float(stock_peak.item()):.3f} native_peak={float(native_peak.item()):.3f} "
            f"stock={stock_ms:.3f} ms minimlx_native={native_ms:.3f} ms"
        )


if __name__ == "__main__":
    main()
