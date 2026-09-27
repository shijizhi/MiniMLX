"""Local GDN prefill experiment.

This is intentionally an isolated A/B benchmark. It does not patch the
runtime. The candidate keeps the exact scalar-gated delta recurrence, stages
q/k/v time tiles in threadgroup memory, keeps the recurrent state in registers,
and uses SIMD shuffle reductions for the two Dk contractions.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, gated_delta_ops


def blocked256_simd_kernel(block_t: int):
    """256-thread blocked recurrence with register state and SIMD reductions."""
    source = f"""
        constexpr int BT = {block_t};
        constexpr int DB = 32;
        constexpr int SEG = 8;
        constexpr int FRAG = 16;
        threadgroup InT k_tile[BT * Dk];
        threadgroup InT q_tile[BT * Dk];
        threadgroup InT v_tile[BT * DB];

        uint tid = thread_position_in_threadgroup.x;
        uint block = threadgroup_position_in_grid.x;
        uint hv_idx = threadgroup_position_in_grid.y;
        uint b_idx = threadgroup_position_in_grid.z;
        uint hk_idx = hv_idx / (Hv / Hk);
        uint dv = tid / SEG;
        uint seg = tid % SEG;
        uint d0 = seg * FRAG;
        uint dv0 = block * DB;

        auto k_base = k + ((size_t)b_idx * T * Hk + hk_idx) * Dk;
        auto q_base = q + ((size_t)b_idx * T * Hk + hk_idx) * Dk;
        auto v_base = v + ((size_t)b_idx * T * Hv + hv_idx) * Dv + dv0;
        auto y_base = y + ((size_t)b_idx * T * Hv + hv_idx) * Dv + dv0;
        auto state_in_base = state_in + (((size_t)b_idx * Hv + hv_idx) * Dv + dv0 + dv) * Dk + d0;
        auto state_out_base = state_out + (((size_t)b_idx * Hv + hv_idx) * Dv + dv0 + dv) * Dk + d0;

        float state[FRAG];
        for (int i = 0; i < FRAG; ++i) state[i] = float(state_in_base[i]);

        for (int t0 = 0; t0 < T; t0 += BT) {{
            int count = min(BT, T - t0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int p = tid; p < count * Dk; p += 256) {{
                int r = p / Dk;
                int d = p - r * Dk;
                k_tile[r * Dk + d] = k_base[(size_t)(t0 + r) * Hk * Dk + d];
                q_tile[r * Dk + d] = q_base[(size_t)(t0 + r) * Hk * Dk + d];
            }}
            for (int p = tid; p < count * DB; p += 256) {{
                int r = p / DB;
                int d = p - r * DB;
                v_tile[r * DB + d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int t = 0; t < count; ++t) {{
                float kv = 0.0f;
                for (int i = 0; i < FRAG; ++i) {{
                    int d = d0 + i;
                    state[i] *= float(g[(size_t)(b_idx * T + t0 + t) * Hv + hv_idx]);
                    kv += state[i] * float(k_tile[t * Dk + d]);
                }}
                kv += simd_shuffle_down(kv, 4);
                kv += simd_shuffle_down(kv, 2);
                kv += simd_shuffle_down(kv, 1);
                float kv_sum = simd_shuffle(kv, (tid % 32) / SEG * SEG);

                float beta_t = float(beta[(size_t)(b_idx * T + t0 + t) * Hv + hv_idx]);
                float delta = (float(v_tile[t * DB + dv]) - kv_sum) * beta_t;
                float out = 0.0f;
                for (int i = 0; i < FRAG; ++i) {{
                    int d = d0 + i;
                    state[i] += float(k_tile[t * Dk + d]) * delta;
                    out += state[i] * float(q_tile[t * Dk + d]);
                }}
                out += simd_shuffle_down(out, 4);
                out += simd_shuffle_down(out, 2);
                out += simd_shuffle_down(out, 1);
                if (seg == 0) {{
                    y_base[(size_t)(t0 + t) * Hv * Dv + dv] = InT(out);
                }}
            }}
        }}
        for (int i = 0; i < FRAG; ++i) state_out_base[i] = StT(state[i]);
    """
    return mx.fast.metal_kernel(
        name=f"minimlx_gdn_blocked256_simd_{block_t}",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


def blocked256_update(kernel, q, k, v, g, beta, state):
    b, t, hk, dk = q.shape
    hv, dv = v.shape[2:]
    return kernel(
        inputs=[q, k, v, g, beta, state, t],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)],
        grid=(256 * (dv // 32), hv, b),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, t, hv, dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def timed(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        out = fn()
        mx.eval(*out)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        mx.eval(*out)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.tokens < 1 or args.block not in (8, 16, 32, 48):
        raise SystemExit("tokens must be positive; block must be 8, 16, 32, or 48")

    model, _ = load(args.model_dir)
    layer = model.language_model.model.layers[0]
    if not layer.is_linear:
        raise RuntimeError("L0 is not a linear-attention layer")
    attn = layer.linear_attn
    b, t = 1, args.tokens
    hidden_size = layer.input_layernorm.weight.shape[0]
    hidden = mx.random.normal((b, t, hidden_size), dtype=mx.bfloat16)
    qkv = attn.in_proj_qkv(hidden)
    a = attn.in_proj_a(hidden)
    beta_logits = attn.in_proj_b(hidden)
    conv_state = mx.zeros((b, attn.conv_kernel_size - 1, attn.conv_dim), dtype=hidden.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = nn.silu(attn.conv1d(conv_input))
    q, k, v = mx.split(conv_out, [attn.key_dim, 2 * attn.key_dim], axis=-1)
    q = q.reshape(b, t, attn.num_k_heads, attn.head_k_dim)
    k = k.reshape(b, t, attn.num_k_heads, attn.head_k_dim)
    v = v.reshape(b, t, attn.num_v_heads, attn.head_v_dim)
    inv_scale = attn.head_k_dim ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    beta = mx.sigmoid(beta_logits)
    g = compute_g(attn.A_log, a, attn.dt_bias)
    # Keep the experiment on the same input type as production MLX GDN.
    # The model's quantized projections normally produce bfloat16 here.
    state = mx.zeros((b, attn.num_v_heads, attn.head_v_dim, attn.head_k_dim), dtype=mx.float32)
    mx.eval(q, k, v, beta, g, state)
    print(f"shapes q={q.shape} k={k.shape} v={v.shape} state={state.shape}")

    candidate256 = blocked256_simd_kernel(args.block)
    ref = lambda: gated_delta_kernel(q, k, v, g, beta, state, None)
    ops = lambda: gated_delta_ops(q, k, v, g, beta, state, None)
    opt256 = lambda: blocked256_update(candidate256, q, k, v, g, beta, state)
    ref_out = ref()
    ops_out = ops()
    opt256_out = opt256()
    mx.eval(*ref_out, *ops_out, *opt256_out)
    blocked256_out_nan = int(mx.sum(mx.isnan(opt256_out[0])))
    blocked256_state_nan = int(mx.sum(mx.isnan(opt256_out[1])))
    blocked256_nan = blocked256_out_nan + blocked256_state_nan
    blocked256_out_max = float(mx.max(mx.abs(opt256_out[0])))
    blocked256_state_max = float(mx.max(mx.abs(opt256_out[1])))
    ops_error = max(
        float(mx.max(mx.abs(ref_out[0].astype(mx.float32) - ops_out[0].astype(mx.float32)))),
        float(mx.max(mx.abs(ref_out[1] - ops_out[1]))),
    )
    blocked256_error = max(
        float(mx.max(mx.abs(ref_out[0].astype(mx.float32) - opt256_out[0].astype(mx.float32)))),
        float(mx.max(mx.abs(ref_out[1] - opt256_out[1]))),
    )
    blocked256_output_error = float(mx.max(mx.abs(ref_out[0].astype(mx.float32) - opt256_out[0].astype(mx.float32))))
    blocked256_state_error = float(mx.max(mx.abs(ref_out[1] - opt256_out[1])))
    ref_ms = timed(ref, args.warmup, args.repeats)
    ops_ms = timed(ops, args.warmup, args.repeats)
    opt256_ms = timed(opt256, args.warmup, args.repeats)
    print(f"MiniMLX isolated GDN prefill A/B (tokens={t}, block={args.block})")
    print(f"blocked256_max_abs_error={blocked256_error:.8e}")
    print(f"blocked256_output_error={blocked256_output_error:.8e} blocked256_state_error={blocked256_state_error:.8e}")
    print(f"blocked256_nan_count={blocked256_nan} out_nan={blocked256_out_nan} state_nan={blocked256_state_nan} blocked256_out_max={blocked256_out_max:.6e} blocked256_state_max={blocked256_state_max:.6e}")
    print(f"blocked256_prefix={opt256_out[0][0, 0, 0, :8].tolist()}")
    print(f"stock_prefix={ref_out[0][0, 0, 0, :8].tolist()}")
    print(f"mlx_ops_max_abs_error={ops_error:.8e}")
    print(f"stock_gated_delta_kernel={ref_ms:.3f} ms")
    print(f"mlx_ops_recurrence={ops_ms:.3f} ms")
    print(f"blocked256_simd_kernel={opt256_ms:.3f} ms")
    print(f"blocked256_speedup={ref_ms / opt256_ms:.3f}x")


if __name__ == "__main__":
    main()
