#!/usr/bin/env python3
"""Interleaved FA-256 kernel comparison to reduce thermal/order bias."""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

import mlx.core as mx
from minimlx_q4_native import fa256_attention
from mlx_lm.models import base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--kv", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    random.seed(20261013)
    budget = 25_000_000
    for q_len in args.query:
        for kv_len in args.kv:
            q = mx.random.normal((1, 40, q_len, 256)).astype(mx.bfloat16)
            k = mx.random.normal((1, 8, kv_len, 256)).astype(mx.bfloat16)
            v = mx.random.normal((1, 8, kv_len, 256)).astype(mx.bfloat16)
            mx.eval(q, k, v)

            def native():
                return fa256_attention(q, k, v, 256**-0.5, budget)

            def stock():
                return base.scaled_dot_product_attention(
                    q, k, v, None, 256**-0.5, "causal"
                )

            def fused():
                return mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=256**-0.5,
                    mask="causal",
                    force_fused=True,
                )

            funcs = {"native": native, "stock": stock, "fused": fused}
            outputs = {name: fn() for name, fn in funcs.items()}
            mx.eval(*outputs.values())
            errors = {}
            for name, output in outputs.items():
                delta = output.astype(mx.float32) - outputs["stock"].astype(mx.float32)
                errors[name] = (
                    float(mx.max(mx.abs(delta)).item()),
                    float(mx.sqrt(mx.mean(delta * delta)).item()),
                )
            samples: dict[str, list[float]] = {name: [] for name in funcs}
            order = list(funcs)
            for _ in range(args.repeats):
                random.shuffle(order)
                for name in order:
                    start = time.perf_counter()
                    result = funcs[name]()
                    mx.eval(result)
                    samples[name].append((time.perf_counter() - start) * 1000)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            print(
                f"q={q_len} kv={kv_len} work={40*q_len*kv_len:,} "
                + " ".join(f"{name}={medians[name]:.2f}ms" for name in funcs)
                + f" native_vs_stock={(medians['native']/medians['stock']-1)*100:+.1f}%"
                + f" fused_vs_stock={(medians['fused']/medians['stock']-1)*100:+.1f}%"
                + " errors="
                + ",".join(
                    f"{name}:max{errors[name][0]:.4f}/rms{errors[name][1]:.5f}"
                    for name in funcs
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
