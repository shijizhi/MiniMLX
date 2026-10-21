#!/usr/bin/env python3
"""Compare the opt-in fused GDN conv+SiLU kernel with mlx-vlm's native math."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from minimlx_runtime.kernels.gdn_conv_silu import fused_depthwise_conv_silu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    if args.channels < 32 or args.channels % 32 or args.cases < 1:
        parser.error("channels must be >=32 and divisible by 32; cases must be positive")
    dtype = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    mx.random.seed(0)
    conv_layer = nn.Conv1d(
        in_channels=args.channels,
        out_channels=args.channels,
        kernel_size=4,
        groups=args.channels,
        bias=False,
    )
    conv_layer.weight = mx.random.normal((args.channels, 4, 1)).astype(dtype)
    layer = SimpleNamespace(
        conv_kernel_size=4,
        conv1d=conv_layer,
    )
    mx.eval(layer.conv1d.weight)
    vlm_max_abs = lm_max_abs = 0.0
    vlm_conv_max_abs = lm_conv_max_abs = 0.0
    vlm_mismatches = lm_mismatches = 0
    for case in range(args.cases):
        # Deterministic, bounded values exercise both signs without extreme
        # sigmoid saturation hiding rounding differences.
        conv_input = (
            mx.random.normal((1, 4, args.channels)) * (0.2 + case * 0.03)
        ).astype(dtype)
        weight = layer.conv1d.weight[:, :, 0].T.astype(mx.float32)
        conv = mx.sum(conv_input.astype(mx.float32) * weight[None, :, :], axis=1)
        vlm_reference = nn.silu(conv.astype(dtype))
        lm_reference = nn.silu(layer.conv1d(conv_input))
        candidate_result = fused_depthwise_conv_silu(
            layer, conv_input, return_conv=True
        )
        if candidate_result is None:
            raise RuntimeError("candidate returned unsupported for a supported case")
        candidate, candidate_conv, candidate_sigmoid = candidate_result
        lm_conv = layer.conv1d(conv_input)
        # nn.silu is mx.compile-decorated, so compare the fused candidate to
        # the same compiled sigmoid primitive rather than eager mx.sigmoid.
        native_sigmoid = nn.sigmoid(conv.astype(dtype))
        eager_sigmoid = mx.sigmoid(conv.astype(dtype))
        mx.eval(vlm_reference, lm_reference, candidate)
        mx.eval(
            conv, lm_conv, candidate_conv, native_sigmoid, eager_sigmoid,
            candidate_sigmoid,
        )
        vlm_diff = mx.abs(
            vlm_reference.astype(mx.float32) - candidate.astype(mx.float32)
        )
        lm_diff = mx.abs(
            lm_reference.astype(mx.float32) - candidate.astype(mx.float32)
        )
        vlm_case_max = float(mx.max(vlm_diff).item())
        lm_case_max = float(mx.max(lm_diff).item())
        vlm_conv_diff = mx.abs(
            conv.astype(dtype).astype(mx.float32)
            - candidate_conv.astype(mx.float32)
        )
        lm_conv_diff = mx.abs(
            lm_conv.astype(mx.float32) - candidate_conv.astype(mx.float32)
        )
        vlm_conv_case_max = float(mx.max(vlm_conv_diff).item())
        lm_conv_case_max = float(mx.max(lm_conv_diff).item())
        sigmoid_diff = mx.abs(
            native_sigmoid.astype(mx.float32)
            - candidate_sigmoid.astype(mx.float32)
        )
        sigmoid_case_max = float(mx.max(sigmoid_diff).item())
        sigmoid_non_bitwise = int(mx.sum(sigmoid_diff != 0).item())
        if case == 0:
            compiled_values = native_sigmoid.reshape(-1).tolist()
            eager_values = eager_sigmoid.reshape(-1).tolist()
            candidate_values = candidate_sigmoid.reshape(-1).tolist()
            conv_values = conv.astype(dtype).reshape(-1).tolist()
            compiled_eager_mismatches = sum(
                left != right for left, right in zip(compiled_values, eager_values)
            )
            samples = []
            for index, (expected, actual) in enumerate(
                zip(compiled_values, candidate_values)
            ):
                if expected != actual:
                    samples.append(
                        f"i{index}:x={conv_values[index]:.8g},"
                        f"compiled={expected:.8g},eager={eager_values[index]:.8g},"
                        f"candidate={actual:.8g}"
                    )
                    if len(samples) == 6:
                        break
            print(
                f"sigmoid_first_case compiled_vs_eager={compiled_eager_mismatches} "
                f"candidate_mismatch_samples=[{' | '.join(samples)}]",
                flush=True,
            )
        vlm_conv_max_abs = max(vlm_conv_max_abs, vlm_conv_case_max)
        lm_conv_max_abs = max(lm_conv_max_abs, lm_conv_case_max)
        vlm_case_mismatches = int(mx.sum(vlm_diff != 0).item())
        lm_case_mismatches = int(mx.sum(lm_diff != 0).item())
        vlm_max_abs = max(vlm_max_abs, vlm_case_max)
        lm_max_abs = max(lm_max_abs, lm_case_max)
        vlm_mismatches += vlm_case_mismatches
        lm_mismatches += lm_case_mismatches
        print(
            f"case={case + 1} vlm_max_abs={vlm_case_max:.9g} "
            f"vlm_non_bitwise={vlm_case_mismatches} "
            f"lm_max_abs={lm_case_max:.9g} lm_non_bitwise={lm_case_mismatches} "
            f"vlm_conv_max_abs={vlm_conv_case_max:.9g} "
            f"lm_conv_max_abs={lm_conv_case_max:.9g} "
            f"sigmoid_max_abs={sigmoid_case_max:.9g} "
            f"sigmoid_non_bitwise={sigmoid_non_bitwise}",
            flush=True,
        )
    passed = max(vlm_max_abs, lm_max_abs) <= args.atol
    print(
        f"GDN_CONV_SILU_CORRECTNESS dtype={args.dtype} channels={args.channels} "
        f"cases={args.cases} vlm_max_abs={vlm_max_abs:.9g} "
        f"vlm_non_bitwise={vlm_mismatches} lm_max_abs={lm_max_abs:.9g} "
        f"lm_non_bitwise={lm_mismatches} "
        f"vlm_conv_max_abs={vlm_conv_max_abs:.9g} "
        f"lm_conv_max_abs={lm_conv_max_abs:.9g} atol={args.atol:g} "
        f"result={'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
