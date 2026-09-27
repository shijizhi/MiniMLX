#!/usr/bin/env python3
"""Profile one Qwen3.5 full-attention prefill layer by real execution phase.

This is intentionally an isolated diagnostic.  It does not patch the runtime
and does not compare whole-model throughput.  The phases mirror
mlx_lm.models.qwen3_next.Qwen3NextAttention.__call__ for a causal prefill.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache


def elapsed(value: mx.array) -> float:
    start = time.perf_counter()
    mx.eval(value)
    return (time.perf_counter() - start) * 1000.0


def measure_once(attn, x: mx.array, force_fused: bool = False):
    b, length, _ = x.shape
    cache = KVCache()
    phases: dict[str, float] = {}
    start_total = time.perf_counter()

    q_proj = attn.q_proj(x)
    phases["q_projection"] = elapsed(q_proj)
    queries, gate = mx.split(
        q_proj.reshape(b, length, attn.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(b, length, -1)
    phases["q_split_gate_reshape"] = elapsed((queries, gate))

    k_proj = attn.k_proj(x)
    phases["k_projection"] = elapsed(k_proj)
    v_proj = attn.v_proj(x)
    phases["v_projection"] = elapsed(v_proj)

    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    keys = attn.k_norm(
        k_proj.reshape(b, length, attn.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = v_proj.reshape(b, length, attn.num_key_value_heads, -1).transpose(
        0, 2, 1, 3
    )
    phases["qk_norm_and_layout"] = elapsed((queries, keys, values))

    queries = attn.rope(queries)
    keys = attn.rope(keys)
    phases["rope"] = elapsed((queries, keys))

    keys, values = cache.update_and_fetch(keys, values)
    phases["kv_cache_update"] = elapsed((keys, values))

    output = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=attn.scale,
        mask="causal",
        force_fused=force_fused,
    )
    phases["sdpa"] = elapsed(output)

    gated = output.transpose(0, 2, 1, 3).reshape(b, length, -1)
    gated = gated * mx.sigmoid(gate)
    phases["gate_and_output_layout"] = elapsed(gated)

    result = attn.o_proj(gated)
    phases["o_projection"] = elapsed(result)
    phases["total"] = (time.perf_counter() - start_total) * 1000.0
    return phases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--force-fused",
        action="store_true",
        help="use MLX's native fused SDPA instead of the default dispatch",
    )
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    layer = model.language_model.model.layers[args.layer]
    if not hasattr(layer, "self_attn"):
        raise SystemExit(f"layer {args.layer} is not full attention")
    attn = layer.self_attn

    hidden_size = attn.q_proj.weight.shape[-1]
    x = mx.random.normal((1, args.tokens, hidden_size), dtype=mx.bfloat16)
    mx.eval(x)
    print(
        "MiniMLX full-attention prefill phase profile "
        f"(L{args.layer}, tokens={args.tokens}, warmup={args.warmup}, "
        f"repeats={args.repeats}, median, "
        f"sdpa={'force_fused' if args.force_fused else 'default'})"
    )
    print(
        f"shape: q_heads={attn.num_attention_heads}, "
        f"kv_heads={attn.num_key_value_heads}, head_dim={attn.head_dim}, "
        f"dtype={x.dtype}"
    )

    for _ in range(args.warmup):
        measure_once(attn, x, args.force_fused)

    samples = [measure_once(attn, x, args.force_fused) for _ in range(args.repeats)]
    names = list(samples[0])
    for name in names:
        print(f"  {name}={statistics.median(s[name] for s in samples):.3f} ms")


if __name__ == "__main__":
    main()
