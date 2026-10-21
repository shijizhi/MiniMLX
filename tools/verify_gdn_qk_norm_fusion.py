#!/usr/bin/env python3
"""Numerically compare fused Q/K RMSNorm+GDN decode with mlx-vlm reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=9217)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_vlm.models.qwen3_5 import gated_delta
    from minimlx_runtime.kernels.gdn_decode import (
        gated_delta_decode_precomputed_gates_fused_qk_norm,
        gated_delta_decode_precomputed_gates,
        gated_delta_qk_rms_norm,
    )

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        print("ERROR: run this verifier on Apple Silicon with Metal available", flush=True)
        return 2

    dtype = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    batch, key_heads, value_heads, key_dim, value_dim = 1, 16, 48, 128, 128
    q_shape = (batch, 1, key_heads, key_dim)
    v_shape = (batch, 1, value_heads, value_dim)
    state_shape = (batch, value_heads, value_dim, key_dim)
    mx.random.seed(args.seed)

    ref_state = mx.random.normal(state_shape, dtype=mx.float32) * 0.02
    fused_state = ref_state
    control_state = ref_state
    max_y_error = max_state_error = 0.0
    max_control_y_error = max_control_state_error = 0.0
    for step in range(args.steps):
        q_raw = mx.random.normal(q_shape, dtype=dtype)
        k_raw = mx.random.normal(q_shape, dtype=dtype)
        v = mx.random.normal(v_shape, dtype=dtype)
        a = mx.random.normal((batch, 1, value_heads), dtype=dtype)
        b = mx.random.normal((batch, 1, value_heads), dtype=dtype)
        A_log = mx.log(mx.random.uniform(shape=(value_heads,), low=0.5, high=16.0))
        dt_bias = mx.random.normal((value_heads,), dtype=dtype)

        q_ref, k_ref = (
            (key_dim ** -0.5) ** 2 * mx.fast.rms_norm(q_raw, None, 1e-6),
            (key_dim ** -0.5) * mx.fast.rms_norm(k_raw, None, 1e-6),
        )
        q_norm_metal, k_norm_metal, q_metal, k_metal = gated_delta_qk_rms_norm(q_raw, k_raw)
        if q_norm_metal is None:
            print("ERROR: Q/K norm diagnostic kernel rejected the test shape", flush=True)
            return 2
        g, beta = gated_delta._compute_g_beta(A_log, a, b, dt_bias)
        ref_y, ref_state = gated_delta.gated_delta_kernel(
            q_ref, k_ref, v, g, beta, ref_state
        )
        control_y, control_state = gated_delta_decode_precomputed_gates(
            q_ref, k_ref, v, g, beta, control_state
        )
        fused = gated_delta_decode_precomputed_gates_fused_qk_norm(
            q_raw, k_raw, v, g, beta, fused_state
        )
        if fused is None:
            print("ERROR: fused Metal kernel rejected the test shape", flush=True)
            return 2
        fused_y, fused_state = fused
        q_norm_ref = mx.fast.rms_norm(q_raw, None, 1e-6)
        k_norm_ref = mx.fast.rms_norm(k_raw, None, 1e-6)
        inv_scale = key_dim ** -0.5
        k_scaled_bf16 = mx.array(inv_scale, dtype=dtype) * k_norm_ref
        k_scaled_f32 = mx.array(inv_scale, dtype=mx.float32) * k_norm_ref
        mx.eval(ref_y, ref_state, control_y, control_state, fused_y, fused_state,
                q_norm_metal, k_norm_metal, q_metal, k_metal, q_norm_ref, k_norm_ref,
                k_scaled_bf16, k_scaled_f32)
        y_error = float(mx.max(mx.abs(ref_y.astype(mx.float32) - fused_y.astype(mx.float32))).item())
        state_error = float(mx.max(mx.abs(ref_state - fused_state)).item())
        control_y_error = float(mx.max(mx.abs(ref_y.astype(mx.float32) - control_y.astype(mx.float32))).item())
        control_state_error = float(mx.max(mx.abs(ref_state - control_state)).item())
        q_norm_error = float(mx.max(mx.abs(q_ref.astype(mx.float32) - q_metal.astype(mx.float32))).item())
        k_norm_error = float(mx.max(mx.abs(k_ref.astype(mx.float32) - k_metal.astype(mx.float32))).item())
        q_rms_error = float(mx.max(mx.abs(q_norm_ref.astype(mx.float32) - q_norm_metal.astype(mx.float32))).item())
        k_rms_error = float(mx.max(mx.abs(k_norm_ref.astype(mx.float32) - k_norm_metal.astype(mx.float32))).item())
        bf16_scale_to_ref = float(mx.max(mx.abs(k_scaled_bf16.astype(mx.float32) - k_ref.astype(mx.float32))).item())
        f32_scale_to_ref = float(mx.max(mx.abs(k_scaled_f32.astype(mx.float32) - k_ref.astype(mx.float32))).item())
        metal_vs_bf16_scale = float(mx.max(mx.abs(k_scaled_bf16.astype(mx.float32) - k_metal.astype(mx.float32))).item())
        metal_vs_f32_scale = float(mx.max(mx.abs(k_scaled_f32.astype(mx.float32) - k_metal.astype(mx.float32))).item())
        max_y_error = max(max_y_error, y_error)
        max_state_error = max(max_state_error, state_error)
        max_control_y_error = max(max_control_y_error, control_y_error)
        max_control_state_error = max(max_control_state_error, control_state_error)
        print(
            f"step={step + 1} fused_output_abs={y_error:.8g} fused_state_abs={state_error:.8g} "
            f"same_norm_output_abs={control_y_error:.8g} "
            f"same_norm_state_abs={control_state_error:.8g} "
            f"q_rms_abs={q_rms_error:.8g} k_rms_abs={k_rms_error:.8g} "
            f"scaled_q_abs={q_norm_error:.8g} scaled_k_abs={k_norm_error:.8g} "
            f"bf16_scale_to_ref={bf16_scale_to_ref:.8g} f32_scale_to_ref={f32_scale_to_ref:.8g} "
            f"metal_vs_bf16_scale={metal_vs_bf16_scale:.8g} "
            f"metal_vs_f32_scale={metal_vs_f32_scale:.8g}",
            flush=True,
        )

    output_ok = bool(mx.allclose(ref_y, fused_y, rtol=1e-3, atol=1e-3).item())
    state_ok = bool(mx.allclose(ref_state, fused_state, rtol=1e-4, atol=1e-5).item())
    control_output_ok = bool(mx.allclose(ref_y, control_y, rtol=1e-3, atol=1e-3).item())
    control_state_ok = bool(mx.allclose(ref_state, control_state, rtol=1e-4, atol=1e-5).item())
    print(
        f"GDN_QK_NORM_FUSED output={'PASS' if output_ok else 'FAIL'} "
        f"state={'PASS' if state_ok else 'FAIL'} max_output_abs={max_y_error:.8g} "
        f"max_state_abs={max_state_error:.8g}",
        flush=True,
    )
    print(
        f"GDN_SAME_NORMALIZED_INPUT_CONTROL output={'PASS' if control_output_ok else 'FAIL'} "
        f"state={'PASS' if control_state_ok else 'FAIL'} "
        f"max_output_abs={max_control_y_error:.8g} "
        f"max_state_abs={max_control_state_error:.8g}",
        flush=True,
    )
    return 0 if output_ok and state_ok and control_output_ok and control_state_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
