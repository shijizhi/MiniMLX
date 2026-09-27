#!/usr/bin/env python3
"""Probe MLX primitives relevant to an inference-only W4A8 prefill path."""

from __future__ import annotations

import inspect

import mlx.core as mx
import mlx.nn as nn


def main() -> None:
    print("MiniMLX QxA8 capability probe")
    print(f"mx.quantize={inspect.signature(mx.quantize)}")
    print(f"mx.quantized_matmul={inspect.signature(mx.quantized_matmul)}")
    print(f"mlx.nn.QuantizedLinear={inspect.signature(nn.QuantizedLinear)}")

    x = mx.random.normal((1, 1024, 1024), dtype=mx.bfloat16)
    mx.eval(x)
    for mode, bits, group_size in (
        ("affine", 8, 64),
        ("mxfp8", 8, 32),
        ("nvfp4", 4, 16),
    ):
        try:
            q = mx.quantize(x, group_size=group_size, bits=bits, mode=mode)
            mx.eval(*q)
            print(
                f"quantize mode={mode} bits={bits} group_size={group_size} "
                f"weight_dtype={q[0].dtype} scale_dtype={q[1].dtype} "
                f"weight_shape={q[0].shape}"
            )
        except Exception as exc:  # capability probe: preserve every result
            print(
                f"quantize mode={mode} bits={bits} group_size={group_size} "
                f"unsupported: {type(exc).__name__}: {exc}"
            )

    fast_names = sorted(name for name in dir(mx.fast) if "quant" in name.lower() or "matmul" in name.lower())
    print(f"mx.fast quant/matmul symbols={fast_names}")
    print("No runtime dispatch is changed by this probe.")


if __name__ == "__main__":
    main()
