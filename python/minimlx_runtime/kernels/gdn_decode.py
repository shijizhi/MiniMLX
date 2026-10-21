"""Opt-in single-token GDN kernel candidate with fused gate preprocessing.

This candidate is shape-gated and keeps the upstream implementation as the
fallback. It is experimental until validated on Apple Silicon hardware.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx


_kernel_cache: dict[tuple[Any, ...], Any] = {}
_route_count = 0
_fallback_count = 0
_qk_norm_route_count = 0
_qk_norm_fallback_count = 0


def _build_kernel(
    key_dim: int,
    value_dim: int,
    input_dtype: Any,
    gate_mode: str = "raw",
    normalize_qk: bool = False,
):
    key = (key_dim, value_dim, input_dtype, gate_mode, normalize_qk)
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    if gate_mode == "raw":
        gate_inputs = ["a", "b", "A_log", "dt_bias"]
        gate_setup = """
        float a_value = float(a[size_t(b_idx) * Hv + hv_idx]);
        float dt_value = float(dt_bias[hv_idx]);
        InT a_sum = InT(a_value + dt_value);
        float x = float(a_sum);
        float softplus_raw = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
        InT softplus_rounded = InT(softplus_raw);
        float decay = exp(-exp(float(A_log[hv_idx])) * float(softplus_rounded));
        float beta_input = float(b[size_t(b_idx) * Hv + hv_idx]);
        InT beta_rounded = InT(1.0f / (1.0f + exp(-beta_input)));
        float beta_value = float(beta_rounded);
        """
    elif gate_mode == "precomputed":
        gate_inputs = ["g", "beta"]
        gate_setup = """
        float decay = float(g[size_t(b_idx) * Hv + hv_idx]);
        float beta_value = float(beta[size_t(b_idx) * Hv + hv_idx]);
        """
    else:
        raise ValueError(f"unknown gate mode: {gate_mode}")

    if normalize_qk:
        inv_scale = key_dim ** -0.5
        q_rescale = inv_scale ** 2
        k_rescale = inv_scale
        norm_setup = f"""
        float q_sq_partial = 0.0f;
        float k_sq_partial = 0.0f;
        for (int i = 0; i < N_PER_T; ++i) {{
            uint s_idx = N_PER_T * dk_idx + i;
            float q_value = float(q_[s_idx]);
            float k_value = float(k_[s_idx]);
            q_sq_partial += q_value * q_value;
            k_sq_partial += k_value * k_value;
        }}
        float q_scale = metal::precise::rsqrt(simd_sum(q_sq_partial) / float(DK) + 1.0e-6f);
        float k_scale = metal::precise::rsqrt(simd_sum(k_sq_partial) / float(DK) + 1.0e-6f);
        InT q_rescale_weight = InT({q_rescale:.17g}f);
        InT k_rescale_weight = InT({k_rescale:.17g}f);
        """
        key_value = "float(InT(float(InT(float(k_[s_idx]) * k_scale)) * float(k_rescale_weight)))"
        query_value = "float(InT(float(InT(float(q_[s_idx]) * q_scale)) * float(q_rescale_weight)))"
    else:
        norm_setup = ""
        key_value = "float(k_[s_idx])"
        query_value = "float(q_[s_idx])"

    source = f"""
        constexpr int DK = {key_dim};
        constexpr int DV = {value_dim};
        constexpr int N_PER_T = DK / 32;

        uint n = thread_position_in_grid.z;
        uint b_idx = n / Hv;
        uint hv_idx = n % Hv;
        uint hk_idx = hv_idx / (Hv / Hk);
        uint dk_idx = thread_position_in_threadgroup.x;
        uint dv_idx = thread_position_in_grid.y;

        auto q_ = q + (size_t(b_idx) * Hk + hk_idx) * DK;
        auto k_ = k + (size_t(b_idx) * Hk + hk_idx) * DK;
        auto v_ = v + (size_t(b_idx) * Hv + hv_idx) * DV;
        auto state_in_ = state_in + (size_t(n) * DV + dv_idx) * DK;
        auto state_out_ = state_out + (size_t(n) * DV + dv_idx) * DK;

        {gate_setup}
        {norm_setup}

        float state[N_PER_T];
        for (int i = 0; i < N_PER_T; ++i) {{
            uint s_idx = N_PER_T * dk_idx + i;
            state[i] = float(state_in_[s_idx]);
        }}

        float kv_partial = 0.0f;
        for (int i = 0; i < N_PER_T; ++i) {{
            uint s_idx = N_PER_T * dk_idx + i;
            state[i] *= decay;
            kv_partial += state[i] * {key_value};
        }}
        float kv = simd_sum(kv_partial);
        float delta = (float(v_[dv_idx]) - kv) * beta_value;

        float out_partial = 0.0f;
        for (int i = 0; i < N_PER_T; ++i) {{
            uint s_idx = N_PER_T * dk_idx + i;
            state[i] += {key_value} * delta;
            out_partial += state[i] * {query_value};
            state_out_[s_idx] = state[i];
        }}
        float out_value = simd_sum(out_partial);
        if (thread_index_in_simdgroup == 0) {{
            y[(size_t(b_idx) * Hv + hv_idx) * DV + dv_idx] = InT(out_value);
        }}
    """
    kernel = mx.fast.metal_kernel(
        name=f"minimlx_gdn_decode_fused_gates_dk{key_dim}_dv{value_dim}",
        input_names=["q", "k", "v", *gate_inputs, "state_in"],
        output_names=["y", "state_out"],
        source=source,
    )
    _kernel_cache[key] = kernel
    return kernel


def gated_delta_decode_precomputed_gates(q, k, v, g, beta, state=None):
    """Run the candidate recurrence with gates computed by the upstream MLX ops."""
    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or q.ndim != 4
        or q.shape[1] != 1
        or q.shape[-1] % 32
        or k.shape != q.shape
        or v.ndim != 4
        or v.shape[0] != q.shape[0]
        or v.shape[1] != 1
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or g.shape != (q.shape[0], 1, v.shape[2])
        or beta.shape != g.shape
    ):
        return None
    batch, _, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if value_heads % key_heads:
        return None
    state_shape = (batch, value_heads, value_dim, key_dim)
    if state is None:
        state = mx.zeros(state_shape, dtype=mx.float32)
    if state.shape != state_shape or state.dtype != mx.float32:
        return None
    kernel = _build_kernel(key_dim, value_dim, q.dtype, "precomputed")
    return kernel(
        inputs=[q, k, v, g, beta, state],
        template=[("InT", q.dtype), ("Dk", key_dim), ("Dv", value_dim),
                  ("Hk", key_heads), ("Hv", value_heads)],
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, 1, value_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def gated_delta_decode_precomputed_gates_fused_qk_norm(
    q, k, v, g, beta, state=None
):
    """Diagnostic candidate: fuse Q/K RMSNorm and scaling into recurrence.

    Matches Qwen3.5's operation order: RMSNorm(eps=1e-6), cast to input dtype,
    then multiply Q by 1/Dk and K by 1/sqrt(Dk), each rounded to input dtype.
    """
    result = _gated_delta_decode_precomputed_gates(
        q, k, v, g, beta, state, normalize_qk=True
    )
    return result


def gated_delta_qk_rms_norm(q, k):
    """Expose the Metal Q/K norm used by the fused candidate for parity checks."""
    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or q.ndim != 4
        or q.shape[1] != 1
        or q.shape[-1] % 32
        or k.shape != q.shape
        or q.dtype != k.dtype
    ):
        return None
    batch, _, key_heads, key_dim = q.shape
    inv_scale = key_dim ** -0.5
    q_rescale = inv_scale ** 2
    k_rescale = inv_scale
    source = f"""
        constexpr int DK = {key_dim};
        constexpr int N_PER_T = DK / 32;
        uint b_idx = thread_position_in_grid.z;
        uint hk_idx = thread_position_in_grid.y;
        uint dk_idx = thread_position_in_threadgroup.x;
        auto q_ = q + (size_t(b_idx) * Hk + hk_idx) * DK;
        auto k_ = k + (size_t(b_idx) * Hk + hk_idx) * DK;
        auto q_norm_out_ = q_norm_out + (size_t(b_idx) * Hk + hk_idx) * DK;
        auto k_norm_out_ = k_norm_out + (size_t(b_idx) * Hk + hk_idx) * DK;
        auto q_out_ = q_out + (size_t(b_idx) * Hk + hk_idx) * DK;
        auto k_out_ = k_out + (size_t(b_idx) * Hk + hk_idx) * DK;
        float q_sq_partial = 0.0f;
        float k_sq_partial = 0.0f;
        for (int i = 0; i < N_PER_T; ++i) {{
            uint idx = N_PER_T * dk_idx + i;
            float qv = float(q_[idx]);
            float kv = float(k_[idx]);
            q_sq_partial += qv * qv;
            k_sq_partial += kv * kv;
        }}
        float q_rms = metal::precise::rsqrt(simd_sum(q_sq_partial) / float(DK) + 1.0e-6f);
        float k_rms = metal::precise::rsqrt(simd_sum(k_sq_partial) / float(DK) + 1.0e-6f);
        InT q_rescale_weight = InT({q_rescale:.17g}f);
        InT k_rescale_weight = InT({k_rescale:.17g}f);
        for (int i = 0; i < N_PER_T; ++i) {{
            uint idx = N_PER_T * dk_idx + i;
            InT q_normalized = InT(float(q_[idx]) * q_rms);
            InT k_normalized = InT(float(k_[idx]) * k_rms);
            q_norm_out_[idx] = q_normalized;
            k_norm_out_[idx] = k_normalized;
            q_out_[idx] = InT(float(q_normalized) * float(q_rescale_weight));
            k_out_[idx] = InT(float(k_normalized) * float(k_rescale_weight));
        }}
    """
    kernel = mx.fast.metal_kernel(
        name=f"minimlx_gdn_qk_rms_norm_dk{key_dim}",
        input_names=["q", "k"],
        output_names=["q_norm_out", "k_norm_out", "q_out", "k_out"],
        source=source,
    )
    return kernel(
        inputs=[q, k],
        template=[("InT", q.dtype), ("Dk", key_dim), ("Hk", key_heads)],
        grid=(32, key_heads, batch),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape, k.shape, q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype, q.dtype, k.dtype],
    )


def _gated_delta_decode_precomputed_gates(q, k, v, g, beta, state, normalize_qk):
    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or q.ndim != 4
        or q.shape[1] != 1
        or q.shape[-1] % 32
        or k.shape != q.shape
        or v.ndim != 4
        or v.shape[0] != q.shape[0]
        or v.shape[1] != 1
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or g.shape != (q.shape[0], 1, v.shape[2])
        or beta.shape != g.shape
    ):
        return None
    batch, _, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if value_heads % key_heads:
        return None
    state_shape = (batch, value_heads, value_dim, key_dim)
    if state is None:
        state = mx.zeros(state_shape, dtype=mx.float32)
    if state.shape != state_shape or state.dtype != mx.float32:
        return None
    kernel = _build_kernel(
        key_dim, value_dim, q.dtype, "precomputed", normalize_qk=normalize_qk
    )
    return kernel(
        inputs=[q, k, v, g, beta, state],
        template=[("InT", q.dtype), ("Dk", key_dim), ("Dv", value_dim),
                  ("Hk", key_heads), ("Hv", value_heads)],
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, 1, value_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def gated_delta_compute_fused_gates(a, b, A_log, dt_bias):
    """Expose the hand-written candidate gate math for parity diagnostics only."""
    batch, _, value_heads = a.shape
    source = """
        uint n = thread_position_in_grid.x;
        uint b_idx = n / Hv;
        uint hv_idx = n % Hv;
        InT a_sum = InT(float(a[n]) + float(dt_bias[hv_idx]));
        float x = float(a_sum);
        float softplus_value = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
        float decay = exp(-exp(float(A_log[hv_idx])) * softplus_value);
        float decay_rounded = exp(-exp(float(A_log[hv_idx])) * float(InT(softplus_value)));
        float beta_value = 1.0f / (1.0f + exp(-float(b[n])));
        float beta_tanh_value = 0.5f * tanh(0.5f * float(b[n])) + 0.5f;
        uint sp_bits = as_type<uint>(softplus_value);
        uint sp_rne_bits = sp_bits + 0x7fff + ((sp_bits >> 16) & 1);
        float softplus_bf16_rne = as_type<float>(sp_rne_bits & 0xffff0000);
        uint beta_bits = as_type<uint>(beta_value);
        uint beta_rne_bits = beta_bits + 0x7fff + ((beta_bits >> 16) & 1);
        float beta_bf16_rne = as_type<float>(beta_rne_bits & 0xffff0000);
        float decay_bf16_rne = exp(-exp(float(A_log[hv_idx])) * softplus_bf16_rne);
        g_out[n] = decay;
        g_rounded_out[n] = decay_rounded;
        g_bf16_rne_out[n] = decay_bf16_rne;
        beta_out[n] = InT(beta_value);
        beta_tanh_out[n] = InT(beta_tanh_value);
        beta_bf16_rne_out[n] = InT(beta_bf16_rne);
        sum_out[n] = a_sum;
        softplus_out[n] = softplus_value;
        softplus_rounded_out[n] = InT(softplus_value);
    """
    kernel = mx.fast.metal_kernel(
        name="minimlx_gdn_gate_math_diagnostic",
        input_names=["a", "b", "A_log", "dt_bias"],
        output_names=["g_out", "g_rounded_out", "g_bf16_rne_out", "beta_out", "beta_tanh_out", "beta_bf16_rne_out", "sum_out", "softplus_out", "softplus_rounded_out"],
        source=source,
    )
    return kernel(
        inputs=[a, b, A_log, dt_bias],
        template=[("InT", a.dtype), ("Hv", value_heads)],
        grid=(batch * value_heads, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[a.shape, a.shape, a.shape, a.shape, a.shape, a.shape, a.shape, a.shape, a.shape],
        output_dtypes=[mx.float32, mx.float32, mx.float32, b.dtype, b.dtype, b.dtype, a.dtype, mx.float32, a.dtype],
    )


def gated_delta_decode_fused_gates(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array | None,
    mask: mx.array | None,
    use_kernel: bool,
):
    """Return a fused single-token result, or ``None`` if shape is unsupported."""
    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or mask is not None
        or q.ndim != 4
        or q.shape[1] != 1
        or k.shape != q.shape
        or v.ndim != 4
        or v.shape[0] != q.shape[0]
        or v.shape[1] != 1
        or a.shape != (q.shape[0], 1, v.shape[2])
        or b.shape != a.shape
        or A_log.shape != (v.shape[2],)
        or dt_bias.shape != (v.shape[2],)
        or q.shape[-1] % 32 != 0
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.dtype != a.dtype
        or q.dtype != b.dtype
        or q.dtype != dt_bias.dtype
    ):
        return None

    batch, _, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if value_heads % key_heads != 0:
        return None
    if state is None:
        state = mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32)
    if (
        state.shape != (batch, value_heads, value_dim, key_dim)
        or state.dtype != mx.float32
    ):
        return None

    kernel = _build_kernel(key_dim, value_dim, q.dtype)
    return kernel(
        inputs=[q, k, v, a, b, A_log, dt_bias, state],
        template=[
            ("InT", q.dtype),
            ("Dk", key_dim),
            ("Dv", value_dim),
            ("Hk", key_heads),
            ("Hv", value_heads),
        ],
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, 1, value_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def install_gdn_decode_custom_recurrence() -> bool:
    """Install opt-in GDN recurrence candidates, preserving upstream gate math."""
    fused_qk_norm = os.environ.get("MINIMLX_GDN_DECODE_FUSED_QK_NORM") == "1"
    if (
        os.environ.get("MINIMLX_GDN_DECODE_CUSTOM_RECURRENCE") != "1"
        and not fused_qk_norm
    ):
        return False

    import importlib

    modules = []
    installed = False
    for module_name, alias_names in (
        ("mlx_vlm.models.qwen3_5.gated_delta", ("language",)),
        ("mlx_lm.models.gated_delta", ("qwen3_5",)),
    ):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
        if getattr(module, "_minimlx_gdn_decode_custom_recurrence_installed", False):
            installed = True
            continue
        modules.append((module, alias_names))

    for module, alias_names in modules:
        original = module.gated_delta_update
        supports_cache = module.__name__.startswith("mlx_vlm.")

        language_module = None
        original_normalize_qk = None
        if fused_qk_norm and supports_cache:
            try:
                language_module = importlib.import_module(
                    "mlx_vlm.models.qwen3_5.language"
                )
                norm_class = language_module.Qwen3_5GatedDeltaNet
                if not getattr(norm_class, "_minimlx_fused_qk_norm_installed", False):
                    original_normalize_qk = norm_class._normalize_qk

                    def bypass_decode_qk_norm(self, q, k, *, _original=original_normalize_qk):
                        if (
                            not self.training
                            and q.ndim == 4
                            and q.shape[1] == 1
                            and q.shape[-1] % 32 == 0
                            and q.dtype == k.dtype
                        ):
                            return q, k
                        return _original(self, q, k)

                    norm_class._normalize_qk = bypass_decode_qk_norm
                    norm_class._minimlx_fused_qk_norm_installed = True
                    norm_class._minimlx_fused_qk_norm_original = original_normalize_qk
            except (ImportError, AttributeError):
                language_module = None
            if language_module is None:
                raise RuntimeError(
                    "Qwen3.5 mlx-vlm layer hooks are required for fused Q/K norm"
                )

        def patched(
            q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
            use_kernel=True, state_steps=None, cache=None, cache_index=1,
            *, _original=original, _supports_cache=supports_cache, _module=module,
        ):
            global _route_count, _fallback_count
            global _qk_norm_route_count, _qk_norm_fallback_count
            raw_qk = (
                fused_qk_norm
                and _supports_cache
                and q.ndim == 4
                and q.shape[1] == 1
            )

            def normalize_for_fallback(q_value, k_value):
                if not raw_qk:
                    return q_value, k_value
                inv_scale = k_value.shape[-1] ** -0.5
                return (
                    (inv_scale**2) * mx.fast.rms_norm(q_value, None, 1e-6),
                    inv_scale * mx.fast.rms_norm(k_value, None, 1e-6),
                )

            if cache is not None:
                if not _supports_cache:
                    raise TypeError("mlx-lm GDN does not expose recurrent cache arguments")
                return _original(
                    q, k, v, a, b, A_log, dt_bias, state, mask,
                    use_kernel=use_kernel,
                    state_steps=state_steps,
                    cache=cache,
                    cache_index=cache_index,
                )
            if state_steps is None and mask is None and use_kernel:
                if hasattr(_module, "_compute_g_beta"):
                    g, beta = _module._compute_g_beta(A_log, a, b, dt_bias)
                else:
                    g = _module.compute_g(A_log, a, dt_bias)
                    beta = mx.sigmoid(b)
                candidate = (
                    gated_delta_decode_precomputed_gates_fused_qk_norm(
                        q, k, v, g, beta, state
                    )
                    if raw_qk
                    else gated_delta_decode_precomputed_gates(q, k, v, g, beta, state)
                )
                if candidate is not None:
                    _route_count += 1
                    if raw_qk:
                        _qk_norm_route_count += 1
                    return candidate
            _fallback_count += 1
            q, k = normalize_for_fallback(q, k)
            if raw_qk:
                _qk_norm_fallback_count += 1
            return _original(
                q, k, v, a, b, A_log, dt_bias, state, mask,
                use_kernel=use_kernel,
                **({"state_steps": state_steps} if state_steps is not None else {}),
            )

        module.gated_delta_update = patched
        module._minimlx_gdn_decode_custom_recurrence_installed = True
        module._minimlx_gdn_decode_custom_recurrence_original = original
        for alias_name in alias_names:
            try:
                alias_module = importlib.import_module(f"{module.__package__}.{alias_name}")
            except (ImportError, ModuleNotFoundError):
                continue
            if getattr(alias_module, "gated_delta_update", None) is original:
                alias_module.gated_delta_update = patched
        installed = True
    return installed


def gdn_decode_custom_recurrence_stats() -> tuple[int, int]:
    return _route_count, _fallback_count


def reset_gdn_decode_custom_recurrence_stats() -> None:
    global _route_count, _fallback_count
    global _qk_norm_route_count, _qk_norm_fallback_count
    _route_count = 0
    _fallback_count = 0
    _qk_norm_route_count = 0
    _qk_norm_fallback_count = 0


def gdn_decode_fused_qk_norm_stats() -> tuple[int, int]:
    return _qk_norm_route_count, _qk_norm_fallback_count
