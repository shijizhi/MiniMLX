"""Compare one ANE multi-output GDN projection with four MLX Q4 calls."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from minimlx_runtime.ane import ANEProjection


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
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--ane-output", type=Path)
    parser.add_argument("--ane-library", type=Path)
    parser.add_argument("--ane-model", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    model, _ = load(str(args.model_dir))
    mixer = model.language_model.model.layers[args.layer].linear_attn
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    projections = [getattr(mixer, name) for name in names]
    input_dim = projections[0].weight.shape[1] * 32 // projections[0].bits
    x = mx.array(deterministic_input(args.tokens, input_dim))
    mx.eval(x)

    def mlk():
        return mx.concatenate([projection(x) for projection in projections], axis=-1)

    reference = mlk()
    mx.eval(reference)
    print("MiniMLX GDN stage projection vs ANE")
    print(f"layer={args.layer}, tokens={args.tokens}, input_dim={input_dim}")
    print(f"mlx_four_q4={median_ms(mlk, args.warmup, args.repeats):.3f} ms")
    if args.ane_output is None:
        print("ane_output=not supplied")
        return
    ane = np.fromfile(args.ane_output, dtype=np.float32)
    expected = args.tokens * reference.shape[-1]
    if ane.size != expected:
        raise SystemExit(f"ANE output has {ane.size} values; expected {expected}")
    ane_mx = mx.array(ane.reshape(args.tokens, reference.shape[-1]))
    mx.eval(ane_mx)
    diff = mx.abs(reference.astype(mx.float32) - ane_mx)
    print(f"max_abs_error={float(mx.max(diff)):.8e}")
    print(f"mean_abs_error={float(mx.mean(diff)):.8e}")

    if args.ane_library is None or args.ane_model is None:
        print("ane_bridge=not supplied")
        return
    bridge = ANEProjection(
        str(args.ane_library),
        str(args.ane_model),
        args.tokens,
        input_dim,
        reference.shape[-1],
    )
    bridge_input = x.reshape(1, args.tokens, input_dim)

    def ane_bridge_call():
        return bridge(bridge_input)

    bridge_ms = median_ms(ane_bridge_call, args.warmup, args.repeats)
    print(f"ane_bridge_total={bridge_ms:.3f} ms")
    print(f"ane_bridge_calls={bridge.stats()[0]}")


if __name__ == "__main__":
    main()
