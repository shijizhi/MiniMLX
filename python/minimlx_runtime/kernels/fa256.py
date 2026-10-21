"""Shape-gated native fused SDPA for Qwen-style FA-256 prefill."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


_route_calls = 0
_fallback_calls = 0
_first_route_signature: tuple[Any, ...] | None = None
_native_route_signatures: set[tuple[int, int]] = set()

_DEFAULT_NATIVE_DISPATCH_BUDGET = 250_000_000
_NATIVE_DISPATCH_TARGET_SECONDS = 0.010
_NATIVE_DISPATCH_MIN_BUDGET = 20_000_000
# Keep the measured budget in the same range as oMLX's 10 ms dispatch target.
# On this host oMLX calibrated to ~24.5M work; allowing a 2B cap prevented
# chunking entirely for production 2K-token prefill calls at 4K-16K context.
_NATIVE_DISPATCH_MAX_BUDGET = 25_000_000


def _calibrate_native_dispatch_budget(fa256_attention: Any, mx: Any) -> int:
    """Size FA256 chunks from measured throughput, as in oMLX."""
    try:
        heads, kv_heads, q_len, kv_len, head_dim = 16, 2, 1024, 8192, 256
        work = heads * q_len * kv_len
        q = mx.random.normal((1, heads, q_len, head_dim)).astype(mx.bfloat16)
        k = mx.random.normal((1, kv_heads, kv_len, head_dim)).astype(mx.bfloat16)
        v = mx.random.normal((1, kv_heads, kv_len, head_dim)).astype(mx.bfloat16)
        mx.eval(q, k, v)
        best = 0.0
        for i in range(4):
            start = time.perf_counter()
            mx.eval(fa256_attention(q, k, v, head_dim**-0.5, 0))
            elapsed = time.perf_counter() - start
            if i:
                best = elapsed if best == 0.0 else min(best, elapsed)
        if best <= 0.0:
            raise RuntimeError(f"invalid calibration duration: {best}")
        budget = int(work / best * _NATIVE_DISPATCH_TARGET_SECONDS)
        return max(
            _NATIVE_DISPATCH_MIN_BUDGET,
            min(_NATIVE_DISPATCH_MAX_BUDGET, budget),
        )
    except Exception as exc:
        print(
            "[fa256-native] dispatch calibration failed; "
            f"using {_DEFAULT_NATIVE_DISPATCH_BUDGET}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return _DEFAULT_NATIVE_DISPATCH_BUDGET


def install_fa256_force_fused(min_query_tokens: int = 1024) -> bool:
    """Route eligible head-dim-256 causal prefills to MLX's fused SDPA.

    This uses the public MLX operator rather than a copied oMLX kernel. The
    route is opt-in until it has been validated across supported hardware.
    """
    if os.environ.get("MINIMLX_FA256_FORCE_FUSED") != "1":
        return False
    from mlx_lm.models import base

    if getattr(base, "_minimlx_fa256_force_fused_installed", False):
        return True
    stock = base.scaled_dot_product_attention
    minimum = int(os.environ.get("MINIMLX_FA256_MIN_QUERY_TOKENS", min_query_tokens))
    trace = os.environ.get("MINIMLX_FA256_TRACE") == "1"

    def trace_line(message: str) -> None:
        if trace:
            print(f"[fa256] {message}", file=sys.stderr, flush=True)

    def eligible(queries: Any, keys: Any, values: Any, cache: Any, mask: Any, sinks: Any) -> bool:
        if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
            return False
        if queries.shape[-1] != 256 or keys.shape[-1] != 256:
            return False
        if queries.shape[-2] < minimum or queries.shape[-2] > keys.shape[-2]:
            return False
        if queries.shape[-3] % keys.shape[-3] != 0:
            return False
        if queries.dtype not in (mx.float16, mx.bfloat16) or queries.dtype != keys.dtype:
            return False
        if cache is not None and hasattr(cache, "bits"):
            return False
        if sinks is not None:
            return False
        return isinstance(mask, str) and mask == "causal"

    def patched(queries, keys, values, cache, scale, mask, sinks=None):
        global _route_calls, _fallback_calls, _first_route_signature
        if eligible(queries, keys, values, cache, mask, sinks):
            _route_calls += 1
            if _first_route_signature is None:
                _first_route_signature = (
                    tuple(queries.shape),
                    tuple(keys.shape),
                    tuple(values.shape),
                    str(queries.dtype),
                    minimum,
                )
                trace_line(
                    "route hit "
                    f"q={_first_route_signature[0]} "
                    f"k={_first_route_signature[1]} "
                    f"v={_first_route_signature[2]} "
                    f"dtype={_first_route_signature[3]} min_q={minimum}"
                )
            try:
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=scale,
                    mask=mask,
                    sinks=sinks,
                    force_fused=True,
                )
            except Exception:
                # Preserve the stock implementation as the correctness fallback.
                _fallback_calls += 1
                trace_line("force_fused raised; fell back to stock")
                pass
        return stock(queries, keys, values, cache, scale, mask, sinks)

    base.scaled_dot_product_attention = patched
    base._minimlx_fa256_force_fused_installed = True
    base._minimlx_fa256_force_fused_original = stock
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("mlx_lm.models."):
            continue
        if getattr(module, "scaled_dot_product_attention", None) is stock:
            module.scaled_dot_product_attention = patched
    trace_line("installed")
    return True


def install_native_fa256_prefill(min_query_tokens: int = 1024) -> bool:
    """Route supported mlx-lm and mlx-vlm Qwen FA-256 prefills natively."""
    if os.environ.get("MINIMLX_ENABLE_NATIVE_FA256_PREFILL") != "1":
        return False
    bases = []
    try:
        from mlx_lm.models import base as lm_base

        bases.append(lm_base)
    except ImportError:
        pass
    # The MTP path loads Qwen3.5 through mlx-vlm. Its language module imports
    # scaled_dot_product_attention from this separate base module, so patching
    # mlx-lm alone leaves VLM prefill on stock SDPA.
    try:
        from mlx_vlm.models import base as vlm_base

        if vlm_base not in bases:
            bases.append(vlm_base)
    except ImportError:
        pass
    pending_bases = [
        module
        for module in bases
        if not getattr(module, "_minimlx_native_fa256_installed", False)
    ]
    if not pending_bases:
        return bool(bases)
    native_root = Path(__file__).resolve().parents[3] / "native"
    if str(native_root) not in sys.path:
        sys.path.insert(0, str(native_root))
    try:
        from minimlx_q4_native import fa256_attention
    except ImportError:
        return False

    dispatch_budget = _calibrate_native_dispatch_budget(fa256_attention, mx)
    print(
        f"[fa256-native] calibrated dispatch_budget={dispatch_budget}",
        file=sys.stderr,
        flush=True,
    )

    minimum = int(os.environ.get("MINIMLX_NATIVE_FA256_MIN_QUERY_TOKENS", min_query_tokens))

    def eligible(queries: Any, keys: Any, values: Any, cache: Any, mask: Any, sinks: Any) -> bool:
        return (
            cache is not None
            and not hasattr(cache, "bits")
            and sinks is None
            and isinstance(mask, str)
            and mask == "causal"
            and queries.ndim == 4
            and keys.ndim == 4
            and values.ndim == 4
            and queries.dtype == mx.bfloat16
            and keys.dtype == mx.bfloat16
            and values.dtype == mx.bfloat16
            and queries.shape[-1] == 256
            and keys.shape[-1] == 256
            and values.shape[-1] == 256
            and queries.shape[1] % keys.shape[1] == 0
            and queries.shape[2] >= minimum
            and queries.shape[2] <= keys.shape[2]
            # Match OMLX_FA256_MIN_KV_LEN's default. The first 1K-token
            # prompt chunk stays on the stock path; longer-context chunks
            # take the native FA-256 route.
            and keys.shape[2] >= 2048
        )

    installed = False
    for base in pending_bases:
        stock = base.scaled_dot_product_attention

        def patched(queries, keys, values, cache, scale, mask, sinks=None, _stock=stock):
            if os.environ.get("MINIMLX_DISABLE_NATIVE_FA256_RUNTIME") == "1":
                return _stock(queries, keys, values, cache, scale, mask, sinks)
            if eligible(queries, keys, values, cache, mask, sinks):
                try:
                    signature = (int(queries.shape[2]), int(keys.shape[2]))
                    if signature not in _native_route_signatures and len(_native_route_signatures) < 8:
                        _native_route_signatures.add(signature)
                        print(
                            "[fa256-native] route "
                            f"q={tuple(queries.shape)} k={tuple(keys.shape)} "
                            f"dtype={queries.dtype}",
                            file=sys.stderr,
                            flush=True,
                        )
                    return fa256_attention(
                        queries, keys, values, scale, dispatch_budget
                    )
                except (RuntimeError, ValueError):
                    pass
            return _stock(queries, keys, values, cache, scale, mask, sinks)

        base.scaled_dot_product_attention = patched
        base._minimlx_native_fa256_installed = True
        base._minimlx_native_fa256_original = stock
        prefixes = ("mlx_lm.models.", "mlx_vlm.models.")
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith(prefixes):
                continue
            if getattr(module, "scaled_dot_product_attention", None) is stock:
                module.scaled_dot_product_attention = patched
        installed = True
    return installed


def fa256_stats() -> tuple[int, int]:
    """Return (force-fused route calls, force-fused fallback calls)."""
    return _route_calls, _fallback_calls
