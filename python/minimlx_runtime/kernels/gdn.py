"""Inference-only Gated DeltaNet prefill kernel.

This is deliberately an adapter-level optimization.  Unsupported shapes,
masks, CPU execution, and single-token decode fall back to mlx-lm's kernel.
The recurrence is unchanged; only its Apple-GPU mapping is specialized for
the Qwen3.5 GDN shape used by the first MiniMLX target.
"""

from __future__ import annotations

import os
from typing import Optional

import mlx.core as mx


_kernels: dict[int, object] = {}
_mtp_vlm_gdn_routes = 0
_mtp_vlm_gdn_fallbacks = 0



def _kernel(block_t: int):
    cached = _kernels.get(block_t)
    if cached is not None:
        return cached
    source = f"""
        constexpr int BT = {block_t};
        constexpr int DB = 32;
        constexpr int SEG = 8;
        constexpr int FRAG = 16;
        // Pad each row so the 16-bit key/query row starts remain 16-byte
        // aligned for the float4 fragments used in the recurrent update.
        threadgroup InT k_tile[BT * (Dk + 8)];
        threadgroup InT q_tile[BT * (Dk + 8)];
        threadgroup InT v_tile[BT * (DB + 8)];
        threadgroup float g_tile[BT];
        threadgroup float beta_tile[BT];

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

        float4 state[4];
        auto state_in_vec = (device const float4*)state_in_base;
        for (int i = 0; i < 4; ++i) state[i] = state_in_vec[i];

        for (int t0 = 0; t0 < T; t0 += BT) {{
            int count = min(BT, T - t0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int p = tid; p < count * Dk; p += 256) {{
                int r = p / Dk;
                int d = p - r * Dk;
                k_tile[r * (Dk + 8) + d] = k_base[(size_t)(t0 + r) * Hk * Dk + d];
                q_tile[r * (Dk + 8) + d] = q_base[(size_t)(t0 + r) * Hk * Dk + d];
            }}
            for (int p = tid; p < count * DB; p += 256) {{
                int r = p / DB;
                int d = p - r * DB;
                v_tile[r * (DB + 8) + d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
            }}
            for (int p = tid; p < count; p += 256) {{
                g_tile[p] = g[(size_t)(b_idx * T + t0 + p) * Hv + hv_idx];
                beta_tile[p] = beta[(size_t)(b_idx * T + t0 + p) * Hv + hv_idx];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int t = 0; t < count; ++t) {{
                auto k_vec = (threadgroup vec<InT, 4>*)&k_tile[t * (Dk + 8) + d0];
                auto q_vec = (threadgroup vec<InT, 4>*)&q_tile[t * (Dk + 8) + d0];
                float4 kf[4];
                for (int i = 0; i < 4; ++i) kf[i] = float4(k_vec[i]);

                float4 kv_frag = 0.0f;
                for (int i = 0; i < 4; ++i) {{
                    state[i] *= g_tile[t];
                    kv_frag += state[i] * kf[i];
                }}
                float kv = kv_frag.x + kv_frag.y + kv_frag.z + kv_frag.w;
                kv += simd_shuffle_down(kv, 4);
                kv += simd_shuffle_down(kv, 2);
                kv += simd_shuffle_down(kv, 1);
                float kv_sum = simd_shuffle(kv, (tid % 32) / SEG * SEG);

                float delta = (float(v_tile[t * (DB + 8) + dv]) - kv_sum) * beta_tile[t];
                float4 out_frag = 0.0f;
                for (int i = 0; i < 4; ++i) {{
                    state[i] += kf[i] * delta;
                    out_frag += state[i] * float4(q_vec[i]);
                }}
                float out = out_frag.x + out_frag.y + out_frag.z + out_frag.w;
                out += simd_shuffle_down(out, 4);
                out += simd_shuffle_down(out, 2);
                out += simd_shuffle_down(out, 1);
                if (seg == 0) y_base[(size_t)(t0 + t) * Hv * Dv + dv] = InT(out);
            }}
        }}
        auto state_out_vec = (device float4*)state_out_base;
        for (int i = 0; i < 4; ++i) state_out_vec[i] = state[i];
    """
    compiled = mx.fast.metal_kernel(
        name=f"minimlx_gdn_prefill_simd_{block_t}",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )
    _kernels[block_t] = compiled
    return compiled


def gated_delta_prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
):
    """Run the specialized recurrence or return ``None`` when unsupported."""
    if mask is not None or g.ndim != 3 or q.ndim != 4:
        return None
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if tokens < 16 or key_dim != 128 or value_dim % 32 != 0:
        return None
    if state.dtype != mx.float32:
        return None
    override = os.environ.get("MINIMLX_GDN_BLOCK_T")
    if override in {"16", "32", "48"}:
        block_t = int(override)
    else:
        # Match oMLX's dtype-based selection: BF16 uses the larger time
        # tile, while FP32 stays at 16 to respect threadgroup memory.
        block_t = 16 if q.dtype == mx.float32 else 32
    kernel = _kernel(block_t)
    return kernel(
        inputs=[q, k, v, g, beta, state, tokens],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", key_dim),
            ("Dv", value_dim),
            ("Hk", heads),
            ("Hv", value_heads),
        ],
        grid=(256 * (value_dim // 32), value_heads, batch),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, tokens, value_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def install_mtp_vlm_gdn_prefill() -> bool:
    """Install MiniMLX's blocked GDN prefill kernel on mlx-vlm Qwen3.5.

    The regular runtime patches mlx-lm's vendored gated-delta module, while
    the MTP runtime loads Qwen3.5 through mlx-vlm, which has a separate module
    and function binding. Keep the eligibility contract aligned with OMLX's
    qwen35_gdn_chunked patch; unsupported calls retain mlx-vlm's stock path.
    """
    from mlx_vlm.models.qwen3_5 import gated_delta, language

    if getattr(gated_delta, "_minimlx_mtp_gdn_prefill_installed", False):
        return True

    original = gated_delta.gated_delta_update

    def patched(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
        use_kernel=True, state_steps=None, cache=None, cache_index=1,
    ):
        global _mtp_vlm_gdn_routes, _mtp_vlm_gdn_fallbacks
        if cache is not None:
            if state is not None or state_steps is not None:
                raise ValueError(
                    "The recurrent cache owns GDN state and history retention."
                )
            return cache.update_recurrent(
                cache_index,
                q.shape[1],
                lambda initial, steps: patched(
                    q,
                    k,
                    v,
                    a,
                    b,
                    A_log,
                    dt_bias,
                    state=(
                        initial
                        if initial is None or initial.shape[0] == q.shape[0]
                        else None
                    ),
                    mask=mask,
                    use_kernel=use_kernel,
                    state_steps=steps,
                ),
            )
        eligible = (
            use_kernel
            and state_steps is None
            and mask is None
            and q.ndim == 4
            and q.shape[1] >= 64
            and q.shape[-1] % 16 == 0
            and v.shape[-1] % 32 == 0
            and a.ndim == 3
            and mx.metal.is_available()
        )
        if eligible:
            g, beta = gated_delta._compute_g_beta(A_log, a, b, dt_bias)
            if state is None:
                state = mx.zeros(
                    (q.shape[0], v.shape[-2], v.shape[-1], q.shape[-1]),
                    dtype=mx.float32,
                )
            result = gated_delta_prefill(q, k, v, g, beta, state, mask)
            if result is not None:
                _mtp_vlm_gdn_routes += 1
                if _mtp_vlm_gdn_routes == 1:
                    print(
                        "[mtp-gdn-native] route "
                        f"q={tuple(q.shape)} v={tuple(v.shape)} "
                        f"dtype={q.dtype}"
                    )
                return result
        _mtp_vlm_gdn_fallbacks += 1
        return original(
            q, k, v, a, b, A_log, dt_bias, state, mask,
            use_kernel=use_kernel, state_steps=state_steps,
            cache=cache, cache_index=cache_index,
        )

    gated_delta.gated_delta_update = patched
    language.gated_delta_update = patched
    gated_delta._minimlx_mtp_gdn_prefill_installed = True
    gated_delta._minimlx_mtp_gdn_prefill_original = original
    return True


def mtp_vlm_gdn_prefill_stats() -> tuple[int, int]:
    return _mtp_vlm_gdn_routes, _mtp_vlm_gdn_fallbacks


def reset_mtp_vlm_gdn_prefill_stats() -> None:
    global _mtp_vlm_gdn_routes, _mtp_vlm_gdn_fallbacks
    _mtp_vlm_gdn_routes = 0
    _mtp_vlm_gdn_fallbacks = 0
