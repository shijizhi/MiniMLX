"""Opt-in force-fused SDPA dispatch for cached one-token decode."""

from __future__ import annotations

import sys
from typing import Any

import mlx.core as mx


_route_calls = 0
_fallback_calls = 0


def install_full_attention_decode_force_fused() -> bool:
    """Use MLX's fused SDPA for the real cached decode shape.

    The route is intentionally narrow: one query token, an existing KV
    sequence, unquantized cache, no mask/sinks, and matching low-precision
    tensors.  Unsupported calls return to the original implementation.
    """
    from mlx_lm.models import base

    if getattr(base, "_minimlx_fa_decode_force_fused_installed", False):
        return True

    stock = base.scaled_dot_product_attention

    def eligible(queries: Any, keys: Any, values: Any, cache: Any, mask: Any, sinks: Any) -> bool:
        if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
            return False
        if queries.shape[-2] != 1 or keys.shape[-2] <= 1:
            return False
        if queries.shape[-1] != keys.shape[-1] or keys.shape[-1] != values.shape[-1]:
            return False
        if queries.shape[-3] % keys.shape[-3] != 0:
            return False
        if queries.dtype not in (mx.float16, mx.bfloat16):
            return False
        if queries.dtype != keys.dtype or keys.dtype != values.dtype:
            return False
        if cache is not None and hasattr(cache, "bits"):
            return False
        return mask is None and sinks is None

    def patched(queries, keys, values, cache, scale, mask, sinks=None):
        global _route_calls, _fallback_calls
        if eligible(queries, keys, values, cache, mask, sinks):
            _route_calls += 1
            try:
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=scale,
                    mask=None,
                    sinks=None,
                    force_fused=True,
                )
            except Exception:
                _fallback_calls += 1
        return stock(queries, keys, values, cache, scale, mask, sinks)

    base.scaled_dot_product_attention = patched
    base._minimlx_fa_decode_force_fused_installed = True
    base._minimlx_fa_decode_force_fused_original = stock
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("mlx_lm.models."):
            continue
        if getattr(module, "scaled_dot_product_attention", None) is stock:
            module.scaled_dot_product_attention = patched
    return True


def full_attention_decode_force_fused_stats() -> tuple[int, int]:
    return _route_calls, _fallback_calls
