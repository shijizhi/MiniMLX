#!/usr/bin/env python3
"""Profile one real Qwen3.5 full-attention cached-decode layer by stage."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache


def elapsed(value) -> float:
    start = time.perf_counter()
    mx.eval(value)
    return (time.perf_counter() - start) * 1000.0


def measure_once(attn, context: int):
    dtype = mx.bfloat16
    # QuantizedLinear stores packed uint32 columns.  Recover the logical
    # input width instead of using the packed weight's last dimension.
    input_dim = attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits
    x = mx.random.normal((1, 1, input_dim), dtype=dtype)
    old_keys = mx.random.normal(
        (1, attn.num_key_value_heads, context, attn.head_dim), dtype=dtype
    )
    old_values = mx.random.normal(
        (1, attn.num_key_value_heads, context, attn.head_dim), dtype=dtype
    )
    mx.eval(x, old_keys, old_values)
    cache = KVCache()
    cache.update_and_fetch(old_keys, old_values)
    mx.eval(cache.state)

    phases: dict[str, float] = {}
    q_proj = attn.q_proj(x)
    phases["q_projection"] = elapsed(q_proj)
    queries, gate = mx.split(
        q_proj.reshape(1, 1, attn.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(1, 1, -1)
    phases["q_split_gate_reshape"] = elapsed((queries, gate))

    k_proj = attn.k_proj(x)
    phases["k_projection"] = elapsed(k_proj)
    v_proj = attn.v_proj(x)
    phases["v_projection"] = elapsed(v_proj)

    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    new_keys = attn.k_norm(
        k_proj.reshape(1, 1, attn.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    new_values = v_proj.reshape(
        1, 1, attn.num_key_value_heads, -1
    ).transpose(0, 2, 1, 3)
    phases["qk_norm_and_layout"] = elapsed((queries, new_keys, new_values))

    queries = attn.rope(queries, offset=context)
    new_keys = attn.rope(new_keys, offset=context)
    phases["rope"] = elapsed((queries, new_keys))

    keys, values = cache.update_and_fetch(new_keys, new_values)
    phases["kv_cache_expand"] = elapsed((keys, values))

    output = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=attn.scale,
        mask=None,
    )
    phases["sdpa_after_expand"] = elapsed(output)

    second_keys = mx.random.normal(new_keys.shape, dtype=dtype)
    second_values = mx.random.normal(new_values.shape, dtype=dtype)
    keys, values = cache.update_and_fetch(second_keys, second_values)
    phases["kv_cache_steady"] = elapsed((keys, values))

    output = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=attn.scale,
        mask=None,
    )
    phases["sdpa_steady"] = elapsed(output)

    gated = output.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    gated = gated * mx.sigmoid(gate)
    phases["gate_output_layout"] = elapsed(gated)
    result = attn.o_proj(gated)
    phases["o_projection"] = elapsed(result)
    return phases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--context", type=int, action="append")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    layer = model.language_model.model.layers[args.layer]
    if not hasattr(layer, "self_attn"):
        raise SystemExit(f"layer {args.layer} is not full attention")
    attn = layer.self_attn
    contexts = args.context or [1024, 4096, 16384, 32768]

    print(
        "MiniMLX full-attention cached-decode stage profile "
        f"(L{args.layer}, q_len=1, warmup={args.warmup}, repeats={args.repeats})"
    )
    print(
        f"q_heads={attn.num_attention_heads}, kv_heads={attn.num_key_value_heads}, "
        f"head_dim={attn.head_dim}, dtype=mlx.core.bfloat16"
    )
    for context in contexts:
        samples = [
            measure_once(attn, context) for _ in range(args.warmup + args.repeats)
        ]
        measured = samples[args.warmup :]
        print(f"context={context}")
        for name in measured[0]:
            print(f"  {name}={statistics.median(p[name] for p in measured):.3f} ms")


if __name__ == "__main__":
    main()
