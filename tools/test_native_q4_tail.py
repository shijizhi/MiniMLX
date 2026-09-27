#!/usr/bin/env python3
"""Check the production Q4 QMM's partial-M tile against stock MLX."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
from minimlx_q4_native import affine_qmm  # noqa: E402


def main() -> None:
    config = json.loads((ROOT / "config" / "minimlx.local.json").read_text())
    model, _ = load(config["model_dir"])
    projection = model.language_model.model.layers[0].linear_attn.in_proj_qkv

    # Production prefill reserves one prompt token for decode. A 2048-token
    # prefill step therefore sends 2047 rows to its final large Q4 projection.
    input_width = projection.weight.shape[1] * 32 // projection.bits
    x = mx.random.normal((1, 2047, input_width), dtype=mx.bfloat16)
    mx.eval(x)
    stock = projection(x)
    native = affine_qmm(
        x, projection.weight, projection.scales, projection.biases, 0
    )
    mx.eval(stock, native)
    error = float(mx.max(mx.abs(stock.astype(mx.float32) - native.astype(mx.float32))))
    print(
        f"Q4 partial-M correctness tokens=2047 shape={native.shape} "
        f"max_abs_error={error:.8e}"
    )
    if error != 0.0:
        raise SystemExit("native Q4 output differs from stock MLX")


if __name__ == "__main__":
    main()
