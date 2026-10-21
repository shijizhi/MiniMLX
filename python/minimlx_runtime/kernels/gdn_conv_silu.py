"""Opt-in single-token GDN depthwise-convolution + SiLU fusion.

The Metal kernel is shared by mlx-lm and mlx-vlm Qwen GDN layers. Cache
ownership and the rest of each backend's forward pass remain upstream-owned.
"""

from __future__ import annotations

import os
import inspect
import sys
from typing import Any

import mlx.core as mx


_kernels: dict[tuple[int, Any], Any] = {}
_installed_layers: list[Any] = []
_eligible_layers = 0
_route_count = 0
_fallback_count = 0


def _build_kernel(channels: int, dtype: Any, debug: bool = False):
    key = (channels, dtype, debug)
    if key in _kernels:
        return _kernels[key]
    conv_debug_write = (
        "conv_output[c] = conv; sigmoid_output[c] = sigmoid;" if debug else ""
    )
    source = rf"""
        // Keep the four multiply operations separate from the reduction, as
        // the upstream VLM path materializes (input * weight) before mx.sum.
        // The four-element reduction uses a balanced tree, not a serial FMA.
        #pragma clang fp contract(off)
        uint c = thread_position_in_grid.x;
        if (c >= C) return;
        float p0 = float(conv_input[0 * C + c]) * float(weight[c * 4 + 0]);
        float p1 = float(conv_input[1 * C + c]) * float(weight[c * 4 + 1]);
        float p2 = float(conv_input[2 * C + c]) * float(weight[c * 4 + 2]);
        float p3 = float(conv_input[3 * C + c]) * float(weight[c * 4 + 3]);
        float acc = (p0 + p1) + (p2 + p3);
        InT conv = InT(acc);
        // Keep sigmoid's arithmetic in the input dtype. MLX's native
        // nn.sigmoid/nn.silu path rounds these intermediates to BF16/FP16;
        // doing the calculation in float changes the result by one BF16 ULP.
        InT sigmoid_base = InT(1) / (
            InT(1) + InT(metal::exp(metal::abs(conv)))
        );
        InT sigmoid = (conv < InT(0))
            ? sigmoid_base
            : InT(1) - sigmoid_base;
        output[c] = conv * sigmoid;
        {conv_debug_write}
    """
    kernel = mx.fast.metal_kernel(
        name=f"minimlx_gdn_decode_conv_silu_c{channels}{'_debug' if debug else ''}",
        input_names=["conv_input", "weight"],
        output_names=(
            ["output", "conv_output", "sigmoid_output"] if debug else ["output"]
        ),
        source=source,
    )
    _kernels[key] = kernel
    return kernel


def fused_depthwise_conv_silu(layer, conv_input, *, return_conv=False):
    """Return fused [1, 1, C] output, or None when the contract is unsupported."""
    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or conv_input.ndim != 3
        or conv_input.shape[0] != 1
        or conv_input.shape[1] != 4
        or conv_input.dtype not in (mx.bfloat16, mx.float16)
        or getattr(layer, "conv_kernel_size", None) != 4
    ):
        return None

    channels = int(conv_input.shape[-1])
    conv = getattr(layer, "conv1d", None)
    raw_weight = getattr(conv, "weight", None)
    if (
        raw_weight is None
        or raw_weight.shape != (channels, 4, 1)
        or raw_weight.dtype != conv_input.dtype
    ):
        return None

    cached = getattr(layer, "_minimlx_gdn_conv_weight", None)
    weight_identity = id(raw_weight)
    if cached is None or cached[0] != weight_identity:
        weight = raw_weight[:, :, 0].astype(mx.float32)
        mx.eval(weight)
        cached = (weight_identity, weight)
        layer._minimlx_gdn_conv_weight = cached

    kernel = _build_kernel(channels, conv_input.dtype, debug=return_conv)
    outputs = kernel(
        inputs=[conv_input, cached[1]],
        template=[("InT", conv_input.dtype), ("C", channels), ("DEBUG", return_conv)],
        grid=(((channels + 31) // 32) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, channels)] * (3 if return_conv else 1),
        output_dtypes=[conv_input.dtype] * (3 if return_conv else 1),
    )
    # mlx.fast.metal_kernel returns a list even for a single declared output.
    return (outputs[0], outputs[1], outputs[2]) if return_conv else outputs[0]


def _eligible(self, inputs, mask):
    return (
        not getattr(self, "training", False)
        and inputs.ndim == 3
        and inputs.shape[0] == 1
        and inputs.shape[1] == 1
        and inputs.dtype in (mx.bfloat16, mx.float16)
        and mask is None
        and getattr(self, "conv_kernel_size", None) == 4
        and getattr(self, "head_k_dim", 0) % 32 == 0
        and getattr(self, "head_v_dim", 0) % 32 == 0
    )


def _make_vlm_call(original, language):
    def patched(self, inputs, mask=None, cache=None):
        global _route_count, _fallback_count
        if not _eligible(self, inputs, mask):
            _fallback_count += 1
            return original(self, inputs, mask, cache)

        batch, seq, _ = inputs.shape
        mixed_qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch, seq, -1, self.head_v_dim
        )
        b, a = self._project_gates(inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != batch:
                conv_state = mx.zeros(
                    (batch, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        try:
            conv_out = fused_depthwise_conv_silu(self, conv_input)
        except Exception:
            # The native path is authoritative; do not let an optional kernel
            # compilation/dispatch failure break a request.
            conv_out = None
        if conv_out is None:
            _fallback_count += 1
            return original(self, inputs, mask, cache)

        if cache is not None:
            cache.update_window(
                0, conv_input, self.conv_kernel_size - 1, lengths=cache.lengths
            )
        q, k, v = [
            t.reshape(batch, seq, heads, dim)
            for t, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        q, k = self._normalize_qk(q, k)
        out, _ = language.gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias,
            mask=mask, use_kernel=True, cache=cache,
        )
        if cache is not None and hasattr(cache, "advance"):
            cache.advance(seq)
            language._qwen3_5_advance_left_padding_info(cache, seq)
            language._qwen3_5_advance_lengths_info(cache, seq)
        out = self.norm(out, z)
        _route_count += 1
        return self.out_proj(out.reshape(batch, seq, -1))

    return patched


def _make_lm_call(original, gated_delta):
    def patched(self, inputs, mask=None, cache=None):
        global _route_count, _fallback_count
        if not _eligible(self, inputs, mask):
            _fallback_count += 1
            return original(self, inputs, mask, cache)

        batch, seq, _ = inputs.shape
        sharding_group = getattr(self, "sharding_group", None)
        if sharding_group is not None:
            from mlx.nn.layers.distributed import sum_gradients

            inputs = sum_gradients(sharding_group)(inputs)
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch, seq, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        try:
            conv_out = fused_depthwise_conv_silu(self, conv_input)
        except Exception:
            conv_out = None
        if conv_out is None:
            _fallback_count += 1
            return original(self, inputs, mask, cache)

        if cache is not None:
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, seq)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -keep:, :])
        q, k, v = [
            t.reshape(batch, seq, heads, dim)
            for t, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        state = cache[1] if cache is not None else None
        out, state = gated_delta.gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias,
            state, mask, use_kernel=True,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(seq)
        out = self.norm(out, z)
        out = self.out_proj(out.reshape(batch, seq, -1))
        if sharding_group is not None:
            out = mx.distributed.all_sum(out, group=sharding_group)
        _route_count += 1
        return out

    return patched


def _walk_objects(root):
    import mlx.nn as nn

    def children(value):
        try:
            values = list(vars(value).values())
        except TypeError:
            values = []
        # Some mlx-vlm wrappers use slots instead of __dict__. Read only
        # declared slots; scanning dir(value) can trigger expensive MLX
        # properties during model loading.
        slot_names = []
        for cls in type(value).__mro__:
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            slot_names.extend(slots)
        for name in slot_names:
            if name.startswith("_"):
                continue
            try:
                child = getattr(value, name)
            except (AttributeError, RuntimeError, TypeError):
                continue
            if child is not value:
                values.append(child)
        # Known MLX-VLM wrapper fields; avoid dir() because many MLX objects
        # expose expensive computed properties there.
        for name in (
            "model", "language_model", "text_model", "transformer",
            "backbone", "vision_tower", "layers", "decoder_layers",
            "blocks", "linear_attn", "self_attn", "encoder", "decoder",
        ):
            if any(id(child) == id(existing) for existing in values):
                continue
            try:
                child = getattr(value, name)
            except (AttributeError, RuntimeError, TypeError):
                continue
            if child is not value:
                values.append(child)
        return values

    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, dict):
            pending.extend(value.values())
            continue
        if isinstance(value, (list, tuple)):
            pending.extend(value)
            continue
        if not isinstance(value, nn.Module):
            # mlx-vlm may return a lightweight wrapper around the actual
            # nn.Module tree. Continue through ordinary Python containers so
            # the structural GDN matcher can reach the language layers.
            pending.extend(children(value))
            # Some vendored MLX model classes can cross module boundaries and
            # fail an isinstance check even though they expose the same GDN
            # interface. Let the installer inspect those named layers too.
            if (
                type(value).__name__.endswith("GatedDeltaNet")
                or (
                    hasattr(value, "conv1d")
                    and hasattr(value, "in_proj_qkv")
                )
            ):
                yield value
            continue
        yield value
        pending.extend(children(value))


def _source_contract_matches(cls, backend: str) -> bool:
    try:
        parameters = tuple(inspect.signature(cls.__call__).parameters)
        source = inspect.getsource(cls.__call__)
    except (OSError, TypeError, ValueError):
        return False
    if parameters[:4] != ("self", "inputs", "mask", "cache"):
        return False
    if backend == "mlx-vlm":
        markers = (
            "conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)",
            "cache.update_window(",
            "nn.silu(self._causal_conv1d_decode(conv_input))",
            "self._normalize_qk(q, k)",
            "gated_delta_update(",
            "cache.advance(S)",
        )
    else:
        markers = (
            "conv_input = mx.concatenate([conv_state, qkv], axis=1)",
            "nn.silu(self.conv1d(conv_input))",
            "gated_delta_update(",
            "cache[1] = state",
            "cache.advance(S)",
        )
    return all(marker in source for marker in markers)


def _structural_qwen_gdn_match(layer) -> bool:
    """Recognize vendored/aliased Qwen GDN classes across MLX loaders."""
    cls = type(layer)
    return cls.__name__.endswith("GatedDeltaNet") and all(
        hasattr(layer, name)
        for name in (
            "conv1d", "conv_kernel_size", "conv_dim", "key_dim", "value_dim",
            "head_k_dim", "head_v_dim", "num_k_heads", "num_v_heads",
            "in_proj_qkv", "in_proj_z", "norm", "out_proj",
        )
    )


def _qwen_gdn_layers(model):
    """Yield Qwen3.5 linear-attention layers through the stable model path."""
    language_model = getattr(model, "language_model", None)
    inner_model = getattr(language_model, "model", language_model)
    layers = getattr(inner_model, "layers", ())
    for decoder_layer in layers:
        layer = getattr(decoder_layer, "linear_attn", None)
        if layer is not None:
            yield layer


def install_gdn_decode_conv_silu(model) -> int:
    """Install narrow per-instance Qwen GDN hooks; return eligible layer count."""
    global _eligible_layers
    if os.environ.get("MINIMLX_GDN_DECODE_FUSED_CONV_SILU") != "1":
        return 0
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return 0

    try:
        import importlib

        vlm_language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    except (ImportError, ModuleNotFoundError):
        vlm_language = None
    try:
        import importlib

        lm_qwen = importlib.import_module("mlx_lm.models.qwen3_5")
        lm_gdn = importlib.import_module("mlx_lm.models.gated_delta")
    except (ImportError, ModuleNotFoundError):
        lm_qwen = lm_gdn = None

    count = 0
    walked = 0
    named_layers: list[str] = []
    candidates = list(_qwen_gdn_layers(model))
    if not candidates:
        candidates = list(_walk_objects(model))
    for layer in candidates:
        walked += 1
        cls = type(layer)
        if cls.__name__.endswith("GatedDeltaNet"):
            missing = [
                name for name in (
                    "conv1d", "conv_kernel_size", "conv_dim", "key_dim", "value_dim",
                    "head_k_dim", "head_v_dim", "num_k_heads", "num_v_heads",
                    "in_proj_qkv", "in_proj_z", "norm", "out_proj",
                ) if not hasattr(layer, name)
            ]
            named_layers.append(
                f"{cls.__module__}.{cls.__name__}"
                + (f" missing={','.join(missing)}" if missing else " ready")
            )
        if (
            getattr(layer, "_minimlx_gdn_conv_silu_original_class", None)
            is not None
        ):
            continue
        required = (
            "conv1d", "conv_kernel_size", "conv_dim", "key_dim", "value_dim",
            "head_k_dim", "head_v_dim", "num_k_heads", "num_v_heads",
            "in_proj_qkv", "in_proj_z", "norm", "out_proj",
        )
        if not all(hasattr(layer, name) for name in required):
            continue

        structural_match = _structural_qwen_gdn_match(layer)
        module_name = cls.__module__
        is_vlm_class = (
            vlm_language is not None
            and (
                module_name.startswith("mlx_vlm.models.qwen3_5")
                or (structural_match and "mlx_vlm" in module_name)
            )
        )
        is_lm_class = (
            lm_qwen is not None
            and (
                module_name == lm_qwen.__name__
                or (structural_match and "mlx_lm" in module_name)
            )
        )

        if is_vlm_class:
            if not (
                _source_contract_matches(cls, "mlx-vlm")
                or structural_match
            ):
                continue
            original = cls.__call__
            patched_call = _make_vlm_call(original, vlm_language)
        elif is_lm_class:
            if not (
                _source_contract_matches(cls, "mlx-lm")
                or structural_match
            ):
                continue
            original = cls.__call__
            patched_call = _make_lm_call(original, lm_gdn)
        else:
            continue

        patched_cls = type(
            f"MiniMLXDecodeConvSiLU_{cls.__name__}",
            (cls,),
            {"__call__": patched_call},
        )
        layer.__class__ = patched_cls
        layer._minimlx_gdn_conv_silu_original_class = cls
        _installed_layers.append(layer)
        count += 1
    if count == 0 and os.environ.get("MINIMLX_GDN_DECODE_FUSED_CONV_SILU") == "1":
        print(
            "GDN conv+SiLU install diagnostics: "
            f"walked={walked} named_gdn={len(named_layers)} "
            f"layers={named_layers[:12]} "
            f"root={type(model).__module__}.{type(model).__name__} "
            "root_attrs=" + str({
                name: type(getattr(model, name)).__name__
                for name in (
                    "model", "language_model", "text_model", "transformer",
                    "backbone", "vision_tower",
                )
                if hasattr(model, name)
            }),
            file=sys.stderr,
            flush=True,
        )
    _eligible_layers += count
    return count


def gdn_decode_conv_silu_stats() -> tuple[int, int, int]:
    return _eligible_layers, _route_count, _fallback_count


def reset_gdn_decode_conv_silu_stats() -> None:
    global _route_count, _fallback_count
    _route_count = _fallback_count = 0


def restore_gdn_decode_conv_silu(model=None) -> None:
    """Restore per-instance classes, useful for tests and backend teardown."""
    global _eligible_layers
    targets = {id(item) for item in _walk_objects(model)} if model is not None else None
    remaining = []
    for layer in _installed_layers:
        if targets is not None and id(layer) not in targets:
            remaining.append(layer)
            continue
        original = getattr(layer, "_minimlx_gdn_conv_silu_original_class", None)
        if original is not None:
            layer.__class__ = original
            del layer._minimlx_gdn_conv_silu_original_class
        if hasattr(layer, "_minimlx_gdn_conv_weight"):
            del layer._minimlx_gdn_conv_weight
    _installed_layers[:] = remaining
    _eligible_layers = len(remaining)
