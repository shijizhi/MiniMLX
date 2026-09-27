#!/usr/bin/env python3
"""Measure full-attention KVCache growth policies on the real model path."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache


def run_case(model, context: int, generated: int, prefill_step: int, cache_step: int):
    old_step = KVCache.step
    KVCache.step = cache_step
    try:
        prompt = (mx.arange(context, dtype=mx.uint32) % 1000)[None]
        cache = model.make_cache()
        for start in range(0, context, prefill_step):
            logits = model(prompt[:, start : start + prefill_step], cache=cache)
            mx.eval(logits, [item.state for item in cache])

        token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        start = time.perf_counter()
        for _ in range(generated):
            logits = model(token, cache=cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
            mx.eval(token, [item.state for item in cache])
        elapsed = (time.perf_counter() - start) * 1000.0 / generated
        mx.eval(logits)
        return elapsed, int(mx.argmax(logits[:, -1, :], axis=-1).item())
    finally:
        KVCache.step = old_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--context", type=int, action="append", required=True)
    parser.add_argument("--cache-step", type=int, action="append", required=True)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    print(
        "MiniMLX full-attention KVCache growth experiment "
        f"(prefill_step={args.prefill_step}, decode_tokens={args.decode_tokens}, "
        f"warmup={args.warmup}, repeats={args.repeats})"
    )
    for context in args.context:
        for cache_step in args.cache_step:
            for _ in range(args.warmup):
                run_case(model, context, args.decode_tokens, args.prefill_step, cache_step)
            samples = [
                run_case(model, context, args.decode_tokens, args.prefill_step, cache_step)
                for _ in range(args.repeats)
            ]
            elapsed = statistics.median(item[0] for item in samples)
            tokens = [item[1] for item in samples]
            print(
                f"context={context} cache_step={cache_step} "
                f"decode_step={elapsed:.3f} ms "
                f"tok/s={1000.0 / elapsed:.2f} "
                f"last_token={tokens[0]} stable={len(set(tokens)) == 1}"
            )


if __name__ == "__main__":
    main()
