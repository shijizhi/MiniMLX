#!/usr/bin/env python3
"""Compare the opt-in fused-gates GDN step with mlx-vlm's reference path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=48)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    if args.value_heads % args.key_heads:
        parser.error("value-heads must be divisible by key-heads")
    if args.key_dim % 32 or args.steps < 1:
        parser.error("key-dim must be divisible by 32 and steps must be positive")

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5 import gated_delta
    from minimlx_runtime.kernels.gdn_decode import (
        gated_delta_decode_fused_gates,
        gated_delta_decode_precomputed_gates,
        gated_delta_compute_fused_gates,
    )

    dtype = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    batch = 1
    shape_q = (batch, 1, args.key_heads, args.key_dim)
    shape_v = (batch, 1, args.value_heads, args.value_dim)
    shape_state = (batch, args.value_heads, args.value_dim, args.key_dim)
    mx.random.seed(9217)

    reference_state = mx.random.normal(shape_state, dtype=mx.float32) * 0.02
    recurrence_state = reference_state
    fused_state = reference_state
    max_output_error = 0.0
    max_state_error = 0.0
    max_recurrence_output_error = 0.0
    max_recurrence_state_error = 0.0
    for step in range(args.steps):
        q = mx.random.normal(shape_q, dtype=dtype)
        k = mx.random.normal(shape_q, dtype=dtype)
        v = mx.random.normal(shape_v, dtype=dtype)
        a = mx.random.normal((batch, 1, args.value_heads), dtype=dtype)
        b = mx.random.normal((batch, 1, args.value_heads), dtype=dtype)
        A_log = mx.log(mx.random.uniform(shape=(args.value_heads,), low=0.5, high=16.0))
        dt_bias = mx.random.normal((args.value_heads,), dtype=dtype)

        g, beta = gated_delta._compute_g_beta(A_log, a, b, dt_bias)
        (fused_g, rounded_g, rne_g, fused_beta, tanh_beta, rne_beta,
         fused_sum, fused_softplus, rounded_softplus) = gated_delta_compute_fused_gates(a, b, A_log, dt_bias)
        reference_sum = a + dt_bias
        reference_softplus = nn.softplus(reference_sum)
        reference_y, reference_state = gated_delta.gated_delta_kernel(
            q, k, v, g, beta, reference_state
        )
        recurrence_y, recurrence_state = gated_delta_decode_precomputed_gates(
            q, k, v, g, beta, recurrence_state
        )
        fused = gated_delta_decode_fused_gates(
            q, k, v, a, b, A_log, dt_bias,
            fused_state, mask=None, use_kernel=True,
        )
        if fused is None:
            raise RuntimeError("candidate rejected this shape; check Metal and dimensions")
        fused_y, fused_state = fused
        mx.eval(reference_y, reference_state, recurrence_y, recurrence_state, fused_y, fused_state, g, beta, fused_g, rounded_g, rne_g, fused_beta, tanh_beta, rne_beta, fused_sum, fused_softplus, rounded_softplus, reference_sum, reference_softplus)
        output_error = float(mx.max(mx.abs(reference_y.astype(mx.float32) - fused_y.astype(mx.float32))).item())
        state_error = float(mx.max(mx.abs(reference_state - fused_state)).item())
        recurrence_output_error = float(mx.max(mx.abs(reference_y.astype(mx.float32) - recurrence_y.astype(mx.float32))).item())
        recurrence_state_error = float(mx.max(mx.abs(reference_state - recurrence_state)).item())
        decay_error = float(mx.max(mx.abs(g.astype(mx.float32) - fused_g)).item())
        beta_error = float(mx.max(mx.abs(beta.astype(mx.float32) - fused_beta.astype(mx.float32))).item())
        beta_tanh_error = float(mx.max(mx.abs(beta.astype(mx.float32) - tanh_beta.astype(mx.float32))).item())
        rounded_decay_error = float(mx.max(mx.abs(g.astype(mx.float32) - rounded_g)).item())
        rne_decay_error = float(mx.max(mx.abs(g.astype(mx.float32) - rne_g)).item())
        beta_rne_error = float(mx.max(mx.abs(beta.astype(mx.float32) - rne_beta.astype(mx.float32))).item())
        decay_diff = mx.abs(g.astype(mx.float32) - fused_g).reshape(-1)
        decay_worst = int(mx.argmax(decay_diff).item())
        decay_ref_at_worst = float(g.reshape(-1)[decay_worst].item())
        decay_fused_at_worst = float(fused_g.reshape(-1)[decay_worst].item())
        sum_error = float(mx.max(mx.abs(reference_sum.astype(mx.float32) - fused_sum.astype(mx.float32))).item())
        softplus_error = float(mx.max(mx.abs(reference_softplus.astype(mx.float32) - fused_softplus)).item())
        max_output_error = max(max_output_error, output_error)
        max_state_error = max(max_state_error, state_error)
        max_recurrence_output_error = max(max_recurrence_output_error, recurrence_output_error)
        max_recurrence_state_error = max(max_recurrence_state_error, recurrence_state_error)
        print(
            f"step={step + 1} raw_gate_output_abs={output_error:.8g} "
            f"raw_gate_state_abs={state_error:.8g} "
            f"same_gate_output_abs={recurrence_output_error:.8g} "
            f"same_gate_state_abs={recurrence_state_error:.8g} "
            f"decay_gate_abs={decay_error:.8g} beta_gate_abs={beta_error:.8g} "
            f"decay_worst[{decay_worst}]={decay_ref_at_worst:.8g}/"
            f"{decay_fused_at_worst:.8g} sum_abs={sum_error:.8g} "
            f"softplus_abs={softplus_error:.8g} rounded_softplus_abs="
            f"{float(mx.max(mx.abs(reference_softplus.astype(mx.float32) - rounded_softplus.astype(mx.float32))).item()):.8g} "
            f"rounded_decay_abs={rounded_decay_error:.8g} beta_tanh_abs={beta_tanh_error:.8g} "
            f"rne_decay_abs={rne_decay_error:.8g} beta_rne_abs={beta_rne_error:.8g}",
            flush=True,
        )

    output_ok = mx.allclose(reference_y, fused_y, rtol=1e-3, atol=1e-3).item()
    state_ok = mx.allclose(reference_state, fused_state, rtol=1e-4, atol=1e-5).item()
    recurrence_output_ok = mx.allclose(reference_y, recurrence_y, rtol=1e-3, atol=1e-3).item()
    recurrence_state_ok = mx.allclose(reference_state, recurrence_state, rtol=1e-4, atol=1e-5).item()
    print(
        f"GDN_RECURRENCE_SAME_GATES output={'PASS' if recurrence_output_ok else 'FAIL'} "
        f"state={'PASS' if recurrence_state_ok else 'FAIL'} "
        f"max_output_abs={max_recurrence_output_error:.8g} "
        f"max_state_abs={max_recurrence_state_error:.8g}",
        flush=True,
    )
    print(
        f"GDN_FUSED_GATES_CORRECTNESS output={'PASS' if output_ok else 'FAIL'} "
        f"state={'PASS' if state_ok else 'FAIL'} "
        f"max_output_abs={max_output_error:.8g} max_state_abs={max_state_error:.8g}",
        flush=True,
    )
    return 0 if output_ok and state_ok and recurrence_output_ok and recurrence_state_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
