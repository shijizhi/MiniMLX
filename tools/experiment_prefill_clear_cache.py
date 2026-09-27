#!/usr/bin/env python3
"""Measure the memory/performance trade-off of prefill mx.clear_cache()."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate


def run_once(model, tokenizer, tokens, step: int, generated: int, clear_cache: bool):
    original = mx.clear_cache
    if not clear_cache:
        mx.clear_cache = lambda: None
    try:
        prompt = mx.array(tokens, dtype=mx.uint32)
        start = time.perf_counter()
        responses = stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=generated,
            prefill_step_size=step,
        )
        output = list(responses)
        mx.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        last_token = int(output[-1].token) if output else -1
        return elapsed, last_token, mx.get_peak_memory()
    finally:
        mx.clear_cache = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", type=int, action="append", required=True)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--generated", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer = load(args.model_dir)
    print(
        "MiniMLX prefill clear-cache experiment "
        f"(step={args.step}, generated={args.generated}, "
        f"warmup={args.warmup}, repeats={args.repeats})"
    )
    for tokens_count in args.tokens:
        tokens = [index % 1000 for index in range(tokens_count)]
        for clear_cache in (True, False):
            for _ in range(args.warmup):
                run_once(model, tokenizer, tokens, args.step, args.generated, clear_cache)
            samples = [
                run_once(model, tokenizer, tokens, args.step, args.generated, clear_cache)
                for _ in range(args.repeats)
            ]
            elapsed = statistics.median(item[0] for item in samples)
            peak = statistics.median(item[2] for item in samples)
            last_tokens = [item[1] for item in samples]
            label = "clear_cache" if clear_cache else "keep_allocator_cache"
            print(
                f"tokens={tokens_count} mode={label} total={elapsed:.1f} ms "
                f"peak_memory={peak / (1024**3):.3f} GiB "
                f"last_token={last_tokens[0]} stable={len(set(last_tokens)) == 1}"
            )


if __name__ == "__main__":
    main()
