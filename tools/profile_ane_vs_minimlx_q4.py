#!/usr/bin/env python3
"""Compare a fixed-shape ANE probe with MiniMLX's real MLX Q4 projection.

The ANE model is built separately by build_ane_linear_probe.py.  This script
uses the same deterministic FP16 input as ane_linear_probe.swift and measures
the production Q4 projection directly through mlx_lm.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


def deterministic_input(tokens: int, width: int) -> np.ndarray:
    values = (np.arange(tokens * width, dtype=np.int64) % 97 - 48) / 97.0
    return values.astype(np.float16).reshape(tokens, width)


def median_ms(fn, warmup: int, repeats: int) -> float:
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
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--ane-output", type=Path)
    parser.add_argument("--ane-input-dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    model, _ = load(str(args.model_dir))
    projection = model.language_model.model.layers[args.layer].linear_attn.in_proj_z
    input_dim = projection.weight.shape[1] * 32 // projection.bits
    output_dim = projection.weight.shape[0]
    host_input = deterministic_input(args.tokens, input_dim)
    x = mx.array(host_input)
    mx.eval(x)

    def q4_projection():
        return projection(x)

    reference = q4_projection()
    mx.eval(reference)
    elapsed = median_ms(q4_projection, args.warmup, args.repeats)

    print("MiniMLX Q4 vs ANE projection comparison")
    print(f"layer={args.layer}, tokens={args.tokens}, input_dim={input_dim}, output_dim={output_dim}")
    print(f"q4_projection_mlx={elapsed:.3f} ms")

    if args.ane_output is None:
        print("ane_output=not supplied; run the Swift probe with the same shape first")
        return

    ane = np.fromfile(args.ane_output, dtype=np.float32)
    expected = args.tokens * output_dim
    if ane.size != expected:
        raise SystemExit(f"ANE output has {ane.size} values; expected {expected}")
    ane_mx = mx.array(ane.reshape(args.tokens, output_dim))
    mx.eval(ane_mx)
    ref_f32 = reference.astype(mx.float32)
    diff = mx.abs(ref_f32 - ane_mx)
    max_error = float(mx.max(diff))
    mean_error = float(mx.mean(diff))
    ref_norm = float(mx.linalg.norm(ref_f32))
    relative_l2 = float(mx.linalg.norm(ref_f32 - ane_mx)) / max(ref_norm, 1e-12)
    print(f"q4_vs_ane_max_abs_error={max_error:.8e}")
    print(f"q4_vs_ane_mean_abs_error={mean_error:.8e}")
    print(f"q4_vs_ane_relative_l2={relative_l2:.8e}")


if __name__ == "__main__":
    main()
