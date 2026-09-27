#!/usr/bin/env python3
"""Compare default and force-fused SDPA on real Qwen3.5 prefill shapes."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load


def timed(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def measure(attn, tokens: int, warmup: int, repeats: int):
    dtype = mx.bfloat16
    input_dim = attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits
    x = mx.random.normal((1, tokens, input_dim), dtype=dtype)
    q_proj = attn.q_proj(x)
    queries, _ = mx.split(
        q_proj.reshape(1, tokens, attn.num_attention_heads, -1), 2, axis=-1
    )
    k_proj = attn.k_proj(x)
    v_proj = attn.v_proj(x)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    keys = attn.k_norm(
        k_proj.reshape(1, tokens, attn.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = v_proj.reshape(
        1, tokens, attn.num_key_value_heads, -1
    ).transpose(0, 2, 1, 3)
    queries = attn.rope(queries)
    keys = attn.rope(keys)
    mx.eval(queries, keys, values)

    def stock():
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=attn.scale,
            mask="causal",
            force_fused=False,
        )

    def fused():
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=attn.scale,
            mask="causal",
            force_fused=True,
        )

    result: dict[str, float | str] = {}
    try:
        reference = stock()
        mx.eval(reference)
        result["stock"] = timed(stock, warmup, repeats)
    except Exception as exc:
        result["stock"] = f"ERROR: {exc}"
        reference = None

    try:
        candidate = fused()
        mx.eval(candidate)
        result["fused"] = timed(fused, warmup, repeats)
        if reference is not None:
            result["error"] = float(mx.max(mx.abs(reference - candidate)).item())
        else:
            result["error"] = "n/a"
    except Exception as exc:
        result["fused"] = f"ERROR: {exc}"
        result["error"] = "n/a"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", type=int, action="append", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    layer = model.language_model.model.layers[args.layer]
    if not hasattr(layer, "self_attn"):
        raise SystemExit(f"layer {args.layer} is not full attention")
    attn = layer.self_attn
    print(
        "MiniMLX long-prefill SDPA experiment "
        f"(L{args.layer}, q_heads={attn.num_attention_heads}, "
        f"kv_heads={attn.num_key_value_heads}, head_dim={attn.head_dim}, "
        f"warmup={args.warmup}, repeats={args.repeats})"
    )
    for tokens in args.tokens:
        result = measure(attn, tokens, args.warmup, args.repeats)
        stock = result["stock"]
        fused = result["fused"]
        if isinstance(stock, float) and isinstance(fused, float):
            speedup = stock / fused
            print(
                f"tokens={tokens} stock={stock:.3f} ms "
                f"force_fused={fused:.3f} ms speedup={speedup:.3f}x "
                f"max_abs_error={result['error']:.6e}"
            )
        else:
            print(
                f"tokens={tokens} stock={stock} force_fused={fused} "
                f"max_abs_error={result['error']}"
            )


if __name__ == "__main__":
    main()
