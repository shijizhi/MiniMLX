#!/usr/bin/env python3
"""Check gathered expert projections against MiniMLX's per-expert path."""
import mlx.core as mx


def main():
    mx.random.seed(17)
    for width, output in ((2560, 640), (640, 2560)):
        weight = (mx.random.normal((10, output, width)) * 0.05).astype(mx.bfloat16)
        w, s, b = mx.quantize(weight, group_size=32, bits=4)
        mx.eval(w, s, b)
        for rows in (1, 2, 3, 4):
            failures = 0
            max_difference = 0.0
            for _ in range(8):
                x = mx.random.normal((1, rows, width)).astype(mx.bfloat16)
                ids = mx.broadcast_to(mx.arange(10, dtype=mx.int32), (1, rows, 10))
                actual = mx.gather_qmm(mx.expand_dims(x, (-2, -3)), w, s, b,
                                       rhs_indices=ids, transpose=True, group_size=32, bits=4).squeeze(-2)
                expected = mx.stack([
                    mx.concatenate([mx.quantized_matmul(x[:, i:i + 1], w[j], s[j], b[j],
                                                       transpose=True, group_size=32, bits=4)
                                    for i in range(rows)], axis=1)
                    for j in range(10)
                ], axis=2)
                mx.eval(actual, expected)
                failures += not mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()
                max_difference = max(max_difference, float(mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()))
            print(f"width={width} output={output} rows={rows} mismatches={failures}/8 max_abs={max_difference}", flush=True)


if __name__ == "__main__":
    main()
