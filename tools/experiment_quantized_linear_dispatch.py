#!/usr/bin/env python3
"""Profile Q4 prefill projection dispatch on real Qwen layer shapes.

Compares stock quantized matmul, cached BF16 GEMM, and per-call dequantize+
BF16 GEMM. This is a research tool only; it does not modify model weights or
runtime dispatch.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native"))
try:
    from minimlx_q4_native import affine_qmm as mini_q4_qmm
except ImportError:
    mini_q4_qmm = None


def median_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    layer = model.language_model.model.layers[0]
    owners = (
        ("attn", getattr(layer, "self_attn", None), ("q_proj", "k_proj", "v_proj", "o_proj")),
        ("gdn", getattr(layer, "linear_attn", None), ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")),
        ("mlp", getattr(layer, "mlp", None), ("gate_proj", "up_proj", "down_proj")),
    )
    projections = []
    seen = set()
    for owner_name, owner, names in owners:
        if owner is None:
            continue
        for name in names:
            projection = getattr(owner, name, None)
            if projection is None or id(projection) in seen:
                continue
            if not all(hasattr(projection, key) for key in ("weight", "scales", "bits", "group_size", "mode")):
                continue
            seen.add(id(projection))
            input_dim = int(projection.weight.shape[1] * 32 // projection.bits)
            if input_dim <= 0:
                continue
            projections.append((f"{owner_name}.{name}", projection, input_dim))

    print(
        f"Q4 linear dispatch profile: layer=0, warmup={args.warmup}, "
        f"repeats={args.repeats}, device={mx.default_device()}, "
        f"mini_native_qmm={'available' if mini_q4_qmm else 'unavailable'}"
    )
    for name, projection, input_dim in projections:
        if projection.mode != "affine":
            continue
        dense = mx.dequantize(
            projection.weight,
            projection.scales,
            projection.biases,
            group_size=projection.group_size,
            bits=projection.bits,
            mode=projection.mode,
            dtype=mx.bfloat16,
        )
        mx.eval(dense)
        print(
            f"\n{name}: in={input_dim}, out={projection.weight.shape[0]}, "
            f"bits={projection.bits}, group={projection.group_size}, "
            f"dense_weight={dense.nbytes / (1024**2):.1f} MiB"
        )
        for tokens in args.tokens:
            x = mx.random.normal((1, tokens, input_dim)).astype(mx.bfloat16)
            mx.eval(x)

            def stock():
                return mx.quantized_matmul(
                    x,
                    projection.weight,
                    scales=projection.scales,
                    biases=projection.biases,
                    transpose=True,
                    group_size=projection.group_size,
                    bits=projection.bits,
                    mode=projection.mode,
                )

            def cached_dense():
                return x @ dense.T

            def native_qmm():
                if mini_q4_qmm is None:
                    raise RuntimeError("MiniMLX native Q4 extension is unavailable")
                return mini_q4_qmm(
                    x,
                    projection.weight,
                    projection.scales,
                    projection.biases,
                    8,
                )

            def dequant_each_call():
                weight = mx.dequantize(
                    projection.weight,
                    projection.scales,
                    projection.biases,
                    group_size=projection.group_size,
                    bits=projection.bits,
                    mode=projection.mode,
                    dtype=mx.bfloat16,
                )
                return x @ weight.T

            qmm_ms = median_ms(stock, args.warmup, args.repeats)
            native_eligible = (
                mini_q4_qmm is not None
                and projection.bits == 4
                and projection.group_size == 64
                and projection.weight.shape[0] % 64 == 0
                and input_dim % 64 == 0
            )
            native_ms = (
                median_ms(native_qmm, args.warmup, args.repeats)
                if native_eligible
                else float("nan")
            )
            cached_ms = median_ms(cached_dense, args.warmup, args.repeats)
            streamed_ms = median_ms(dequant_each_call, args.warmup, args.repeats)
            q = stock()
            d = cached_dense()
            mx.eval(q, d)
            max_error = float(mx.max(mx.abs(q.astype(mx.float32) - d.astype(mx.float32))))
            native_error = float("nan")
            if native_eligible:
                n = native_qmm()
                mx.eval(n)
                native_error = float(mx.max(mx.abs(q.astype(mx.float32) - n.astype(mx.float32))))
            print(
                f"  M={tokens:5d} qmm={qmm_ms:8.3f} ms "
                f"mini_native={native_ms:8.3f} ms "
                f"cached_bf16={cached_ms:8.3f} ms "
                f"dequant+bf16={streamed_ms:8.3f} ms "
                f"cached_vs_qmm={(qmm_ms / cached_ms - 1) * 100:+6.1f}% "
                f"max_abs_error={max_error:.3e} native_error={native_error:.3e}"
            )


if __name__ == "__main__":
    main()
