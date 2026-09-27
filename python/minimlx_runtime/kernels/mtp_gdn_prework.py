"""Fused Qwen3.5 MTP GDN prework candidate, adapted to mlx-vlm 0.7.x."""

from __future__ import annotations

import os

import mlx.core as mx


# SPDX-License-Identifier: Apache-2.0
# Prework kernel adapted from oMLX's qwen35_gdn_prework.py, itself adapted
# from mlx-serve / mlxfast-challenge. Keep this provenance with the kernel.
_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint logical_head = threadgroup_position_in_grid.z;
    bool is_q = logical_head < uint(HK);
    bool is_k = logical_head >= uint(HK) && logical_head < 2 * uint(HK);
    uint head = is_q ? logical_head : (is_k ? logical_head - uint(HK)
                                             : logical_head - 2 * uint(HK));
    uint base = is_q ? head * uint(DK)
               : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                       : 2 * uint(HK) * uint(DK) + head * uint(DV));
    T activated[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint channel = base + lane * 4 + i;
        float acc = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            uint input_row = row + tap;
            T xv = input_row < uint(NKEEP)
                ? conv_state[input_row * uint(C) + channel]
                : qkv[(input_row - uint(NKEEP)) * uint(C) + channel];
            acc += float(xv) * float(conv_w[channel * 4 + tap]);
        }
        T conv = T(acc);
        T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
        T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
        activated[i] = act;
        float value = float(act);
        sumsq += value * value;
    }
    if (is_q || is_k) {
        sumsq = simd_sum(sumsq);
        float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
        T scale = is_q ? q_scale : k_scale;
        uint out_base = (row * uint(HK) + head) * uint(DK) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            T rms = T(float(activated[i]) * inv);
            T value = scale * rms;
            if (is_q) q_out[out_base + i] = value;
            else k_out[out_base + i] = value;
        }
    } else {
        uint out_base = (row * uint(HV) + head) * uint(DV) + lane * 4;
        for (uint i = 0; i < 4; ++i) v_out[out_base + i] = activated[i];
    }
    if (row + uint(NKEEP) >= uint(S)) {
        uint state_row = row + uint(NKEEP) - uint(S);
        uint raw_base = row * uint(C) + base + lane * 4;
        uint state_base = state_row * uint(C) + base + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            conv_out[state_base + i] = qkv[raw_base + i];
        }
    }
"""

_kernel = None
_routes = 0


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="minimlx_mtp_qwen35_gdn_prework",
            input_names=["qkv", "conv_state", "conv_w", "q_scale", "k_scale"],
            output_names=["q_out", "k_out", "v_out", "conv_out"],
            source=_SOURCE,
        )
    return _kernel


def _fused_prework(layer, mixed_qkv, conv_state, q_scale, k_scale):
    batch, seq, _ = mixed_qkv.shape
    hk, hv = layer.num_k_heads, layer.num_v_heads
    dk, dv = layer.head_k_dim, layer.head_v_dim
    if batch != 1 or not 3 <= seq <= 9 or dk != 128 or dv != 128:
        return None
    if mixed_qkv.dtype != mx.bfloat16 or layer.conv1d.weight.dtype != mx.bfloat16:
        return None
    return _get_kernel()(
        inputs=[mixed_qkv, conv_state, layer.conv1d.weight, q_scale, k_scale],
        template=[
            ("T", mixed_qkv.dtype), ("HK", hk), ("HV", hv),
            ("DK", dk), ("DV", dv), ("NKEEP", 3),
            ("C", layer.conv_dim), ("S", seq),
        ],
        grid=(32, seq, 2 * hk + hv),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, seq, hk, dk),
            (1, seq, hk, dk),
            (1, seq, hv, dv),
            (1, 3, layer.conv_dim),
        ],
        output_dtypes=[mixed_qkv.dtype] * 4,
    )


def install_mtp_gdn_prework() -> bool:
    """Install an opt-in GDN verify prework fusion on mlx-vlm's verifier."""
    if not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
        from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update
    except ImportError:
        return False
    cls = Qwen3_5BatchInvariantForward
    if getattr(cls, "_minimlx_mtp_gdn_prework_installed", False):
        return True
    original = cls._gated_delta

    def patched(self, layer, inputs, mask, cache):
        global _routes
        enabled = os.environ.get("MINIMLX_MTP_GDN_PREWORK", "0") == "1"
        batch, length, _ = inputs.shape
        cached = cache[0] if cache is not None else None
        eligible = (
            enabled and mx.metal.is_available() and batch == 1
            and 3 <= length <= 9 and mask is None
            and inputs.dtype == mx.bfloat16
            and layer.conv_kernel_size == 4
            and layer.head_k_dim == layer.head_v_dim == 128
            and cache is not None and cached is not None
            and hasattr(cache, "update_window")
            and cached.shape == (1, 3, layer.conv_dim)
            and cached.dtype == mx.bfloat16
            and layer.conv1d.weight.dtype == mx.bfloat16
            and getattr(layer.conv1d, "bias", None) is None
            and getattr(cache, "lengths", None) is None
        )
        if not eligible:
            return original(self, layer, inputs, mask, cache)

        helpers = self._helpers()
        mixed_qkv, z, b, a = self._linears(
            (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a),
            inputs,
        )
        z = z.reshape(batch, length, -1, layer.head_v_dim)
        if not hasattr(layer, "_minimlx_mtp_qk_scales"):
            inv = layer.head_k_dim ** -0.5
            layer._minimlx_mtp_qk_scales = (
                mx.array(inv * inv, dtype=mx.bfloat16),
                mx.array(inv, dtype=mx.bfloat16),
            )
        q_scale, k_scale = layer._minimlx_mtp_qk_scales
        q, k, v, next_conv_state = _fused_prework(
            layer, mixed_qkv, cached, q_scale, k_scale
        )

        # Preserve ArraysCache's speculative temporal-window record exactly;
        # only commit it after the recurrent update and output projection graph
        # have been constructed successfully.
        conv_input = mx.concatenate([cached, mixed_qkv], axis=1)
        output, _ = gated_delta_update(
            q, k, v, a, b, layer.A_log, layer.dt_bias,
            mask=mask, use_kernel=not layer.training, cache=cache,
        )
        output = layer.norm(output, z)
        result = self._linear(layer.out_proj, output.reshape(batch, length, -1))
        # Match OMLX's fused cache-state output. Keep the speculative input
        # window as a lazy rollback record, but do not derive/materialize the
        # ordinary next state through ArraysCache.update_window.
        cache.record_speculative_window(
            0, conv_input, layer.conv_kernel_size - 1
        )
        cache[0] = next_conv_state
        if hasattr(cache, "advance"):
            cache.advance(length)
            helpers._qwen3_5_advance_left_padding_info(cache, length)
            helpers._qwen3_5_advance_lengths_info(cache, length)
        _routes += 1
        return result

    cls._gated_delta = patched
    cls._minimlx_mtp_gdn_prework_installed = True
    cls._minimlx_mtp_gdn_prework_original = original
    return True


def mtp_gdn_prework_routes() -> int:
    return _routes


def reset_mtp_gdn_prework_routes() -> None:
    global _routes
    _routes = 0
