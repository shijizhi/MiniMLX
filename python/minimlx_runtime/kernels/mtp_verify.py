"""Narrow fused block-attention path for Qwen3.5 MTP verification."""

from __future__ import annotations

import os
import mlx.core as mx


_route_calls = 0
_fallback_calls = 0


def install_mtp_verify_fused_attention() -> bool:
    """Install an A/B-switchable block attention path for MTP verification.

    The upstream exact verifier handles a multi-token query by issuing one
    attention call per position and truncating K/V each time. The candidate
    evaluates the same causal block in one fused SDPA call. Eligibility is
    narrow, and the environment switch is read per call so the benchmark can
    compare both paths in one process. The path is enabled by default and can
    be disabled with MINIMLX_MTP_FORCE_FUSED_VERIFY=0.
    """
    from mlx_vlm.models.qwen3_5 import language

    if getattr(language, "_minimlx_mtp_verify_block_attention_installed", False):
        return True

    stock = language._qwen3_5_left_padded_attention

    def patched(queries, keys, values, *, cache, scale: float, mask):
        global _route_calls, _fallback_calls
        eligible = (
            os.environ.get("MINIMLX_MTP_FORCE_FUSED_VERIFY", "1") == "1"
            and queries.ndim == keys.ndim == values.ndim == 4
            and 1 < queries.shape[-2] <= 8
            and keys.shape[-2] > queries.shape[-2]
            and queries.shape[0] == 1
            and queries.shape[-1] == keys.shape[-1] == values.shape[-1] == 256
            and queries.shape[-3] % keys.shape[-3] == 0
            and queries.dtype in (mx.float16, mx.bfloat16)
            and queries.dtype == keys.dtype == values.dtype
            and not (cache is not None and hasattr(cache, "bits"))
            and isinstance(mask, str)
            and mask == "causal"
            and getattr(cache, "left_padding", None) is None
        )
        if eligible:
            _route_calls += 1
            try:
                length = queries.shape[-2]
                # OMLX's VLM MTP patch avoids forcing the fused-attention
                # kernel here.  It groups rows up to MLX's vector-SDPA limit
                # (q_len * GQA <= 32), then relies on bottom-right causal
                # alignment.  Keep this as an explicit source-derived A/B
                # route; the existing force_fused path remains the default
                # until the candidate is verified on the target machine.
                if os.environ.get("MINIMLX_MTP_VERIFY_SDPA", "fused") == "vector":
                    gqa = queries.shape[-3] // keys.shape[-3]
                    row_budget = 32 // gqa
                    if row_budget <= 0:
                        return stock(
                            queries, keys, values, cache=cache, scale=scale, mask=mask
                        )
                    outputs = []
                    start = 0
                    while start < length:
                        end = min(start + row_budget, length)
                        kv_end = keys.shape[-2] - (length - end)
                        outputs.append(
                            mx.fast.scaled_dot_product_attention(
                                queries[..., start:end, :],
                                keys[..., :kv_end, :],
                                values[..., :kv_end, :],
                                scale=scale,
                                mask="causal",
                            )
                        )
                        start = end
                    return outputs[0] if len(outputs) == 1 else mx.concatenate(
                        outputs, axis=-2
                    )

                prefix_length = keys.shape[-2] - length
                q_positions = prefix_length + mx.arange(length)
                k_positions = mx.arange(keys.shape[-2])
                causal_mask = (k_positions[None, :] <= q_positions[:, None])[None, None]
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=scale,
                    mask=causal_mask,
                    sinks=None,
                    force_fused=True,
                )
            except Exception:
                _fallback_calls += 1
        return stock(queries, keys, values, cache=cache, scale=scale, mask=mask)

    language._qwen3_5_left_padded_attention = patched
    language._minimlx_mtp_verify_block_attention_installed = True
    language._minimlx_mtp_verify_block_attention_original = stock
    return True


def mtp_verify_fused_attention_stats() -> tuple[int, int]:
    return _route_calls, _fallback_calls


def reset_mtp_verify_fused_attention_stats() -> None:
    global _route_calls, _fallback_calls
    _route_calls = 0
    _fallback_calls = 0
