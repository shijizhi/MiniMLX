#!/usr/bin/env python3
"""Probe fixed-shape mx.compile for one-token cached Decode.

This is deliberately isolated: it does not patch mlx-lm or MiniMLX runtime.
"""

from __future__ import annotations

import argparse
import time
import traceback
from functools import partial

import mlx.core as mx
from mlx_lm import load
from mlx.utils import tree_flatten


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def build_cache(model, tokens):
    cache = model.make_cache()
    model(tokens, cache=cache)
    mx.eval([c.state for c in cache])
    reserve_decode_capacity(cache)
    return cache


def reserve_decode_capacity(cache) -> None:
    """Reserve one KV-cache update without changing the logical offset.

    Qwen3.5 mixes ArraysCache (linear-attention state) and KVCache
    (full-attention).  KVCache.state is a shortened view, while the decode
    implementation reads/writes KVCache.keys and .values directly.  A first
    decode token can also grow the backing buffer.  A compiled fixed-shape
    probe must capture the backing arrays and avoid that Python-side growth.
    """
    for item in cache:
        keys = getattr(item, "keys", None)
        values = getattr(item, "values", None)
        if keys is None or values is None:
            continue
        required = item.offset + 1
        if required <= keys.shape[2]:
            continue
        step = getattr(item, "step", 256)
        new_capacity = ((required + step - 1) // step) * step
        key_padding = mx.zeros(
            keys.shape[:2] + (new_capacity - keys.shape[2],) + keys.shape[3:],
            dtype=keys.dtype,
        )
        value_padding = mx.zeros(
            values.shape[:2] + (new_capacity - values.shape[2],) + values.shape[3:],
            dtype=values.dtype,
        )
        item.keys = mx.concatenate([keys, key_padding], axis=2)
        item.values = mx.concatenate([values, value_padding], axis=2)


def cache_capture_state(cache):
    """Return the actual mutable arrays used by each cache implementation."""
    captured = []
    for item in cache:
        keys = getattr(item, "keys", None)
        values = getattr(item, "values", None)
        if keys is not None and values is not None:
            # KVCache.state is a view; capture the backing buffers instead.
            captured.append({"keys": keys, "values": values})
        else:
            # ArraysCache.state is its mutable list and must remain a list so
            # assignments such as cache[1] = new_state are reflected.
            captured.append(item.state)
    return captured


def try_compile_probe(label, function, *, inputs=None, outputs=None, call_args=()):
    """Compile and execute one probe, returning its result or None."""
    try:
        compiled = mx.compile(function, inputs=inputs, outputs=outputs)
        result = compiled(*call_args)
        if outputs is not None:
            mx.eval(result, outputs)
        else:
            mx.eval(result)
        print(f"compile_probe_{label}=PASS")
        return result
    except Exception as exc:
        print(f"compile_probe_{label}=FAIL ({type(exc).__name__}: {exc})")
        traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    prompt = mx.full((1, args.context), 9419, dtype=mx.uint32)
    token = mx.array([[9419]], dtype=mx.uint32)

    stock_cache = build_cache(model, prompt)
    compiled_cache = build_cache(model, prompt)

    def stock_step():
        logits = model(token, cache=stock_cache)
        mx.eval(logits, [c.state for c in stock_cache])
        return logits

    # ``mx.compile`` treats arrays reachable through the closure as constants
    # unless they are explicitly captured.  A decode step reads the model
    # parameters as well as mutating the KV/state cache, so both trees must be
    # listed as captured inputs and the cache as captured outputs.  The token
    # remains a normal function argument so this is a fixed-shape, reusable
    # one-token graph rather than a graph tied to the first token value.
    cache_state = cache_capture_state(compiled_cache)
    # Use the flattened array leaves here rather than passing Module objects
    # themselves.  This avoids relying on a particular mlx.nn.Module/dict
    # traversal implementation when compile checks captured inputs.
    # MLX's documented Module pattern captures the complete state tree.  Keep
    # this as the original Module/dict tree for the probe; flattening it into
    # a new list can lose the exact tree identity used by the model closure.
    model_inputs = model.state
    # First isolate model/weight capture from cache capture.  The full probe
    # below is meaningful only if this minimal model-only graph compiles.
    model_probe = try_compile_probe(
        "model_only",
        lambda input_token: model(input_token),
        inputs=model_inputs,
        call_args=(token,),
    )
    if model_probe is None:
        print("compile_supported=False (model-only probe failed)")
        return

    def apply_cache_state(runtime_cache_state):
        for item, state in zip(compiled_cache, runtime_cache_state):
            if isinstance(state, dict):
                # Full-attention KVCache: keep the backing buffers explicit.
                item.keys = state["keys"]
                item.values = state["values"]
            else:
                # Linear-attention ArraysCache: preserve its mutable list.
                item.state = state

    @mx.compile
    def compiled_step(input_token, runtime_cache_state):
        apply_cache_state(runtime_cache_state)
        logits = model(input_token, cache=compiled_cache)
        return logits, runtime_cache_state

    try:
        stock_logits = stock_step()
        compiled_logits, compiled_state = compiled_step(token, cache_state)
        mx.eval(compiled_logits, compiled_state)
    except Exception as exc:
        print("MiniMLX decode compile experiment")
        print(f"compile_supported=False ({type(exc).__name__}: {exc})")
        print(
            "compile_probe_state_counts="
            f"model_arrays:{len(tree_flatten(model.state))},"
            f"cache_arrays:{len(tree_flatten(cache_state))}"
        )
        traceback.print_exc()
        return

    error = mx.max(mx.abs(stock_logits - compiled_logits))
    top1_match = mx.argmax(stock_logits[:, -1, :], axis=-1) == mx.argmax(
        compiled_logits[:, -1, :], axis=-1
    )
    mx.eval(error, top1_match)

    for _ in range(args.warmup):
        stock_step()
        compiled_logits, compiled_state = compiled_step(token, cache_state)
        mx.eval(compiled_logits, compiled_state)

    stock_samples = []
    compiled_samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        stock_step()
        stock_samples.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        compiled_logits, compiled_state = compiled_step(token, cache_state)
        mx.eval(compiled_logits, compiled_state)
        compiled_samples.append((time.perf_counter() - start) * 1000.0)

    stock_ms = median(stock_samples)
    compiled_ms = median(compiled_samples)
    print("MiniMLX decode compile experiment")
    print(f"context={args.context}, warmup={args.warmup}, repeats={args.repeats}")
    print("compile_supported=True")
    print(f"stock_step={stock_ms:.3f} ms")
    print(f"compiled_step={compiled_ms:.3f} ms")
    print(f"speedup={stock_ms / compiled_ms:.2f}x")
    print(f"max_abs_error={float(error):.8e}")
    print(f"top1_match={bool(mx.all(top1_match).item())}")


if __name__ == "__main__":
    main()
