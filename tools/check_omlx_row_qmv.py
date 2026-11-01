#!/usr/bin/env python3
"""Compare oMLX row-exact kernels with this MLX build's serial projections."""
import argparse

import mlx.core as mx
import mlx.nn as nn
from minimlx_runtime.omlx_kernels import row_exact_qmv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    for k, n, group in ((2560, 1024, 64), (2560, 2560, 64), (2560, 1024, 32), (640, 2560, 32)):
        for rows in (2, 3, 4):
            linear = nn.QuantizedLinear(k, n, bias=False, group_size=group, bits=4)
            weights = (mx.random.normal((n, k)) * .05).astype(mx.bfloat16)
            linear.weight, linear.scales, linear.biases = mx.quantize(weights, group_size=group, bits=4)
            mismatches = 0
            for _ in range(args.samples):
                x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
                actual = row_exact_qmv.quantized_linear(linear, x)
                expected = mx.concatenate([linear(x[:, i:i + 1]) for i in range(rows)], axis=1)
                mx.eval(actual, expected)
                mismatches += not mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()
            print(f"k={k} n={n} group={group} rows={rows} samples={args.samples} mismatches={mismatches}", flush=True)


if __name__ == "__main__":
    main()
