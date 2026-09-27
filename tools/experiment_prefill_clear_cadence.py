#!/usr/bin/env python3
"""Measure prefill graph/allocator cleanup cadence without changing model math.

MLX-LM evaluates the KV cache after every prefill chunk and then calls
``mx.clear_cache()``.  This experiment keeps the cache evaluation unchanged
and only delays the allocator/graph-cache clear by N chunk boundaries.
It is diagnostic only; it does not modify the runtime default.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate


def run_once(model, tokenizer, tokens, step: int, generated: int, clear_every: int):
    original_clear_cache = mx.clear_cache
    calls = 0
    clears = 0

    def scheduled_clear_cache() -> None:
        nonlocal calls, clears
        calls += 1
        if clear_every > 0 and calls % clear_every == 0:
            clears += 1
            original_clear_cache()

    mx.clear_cache = scheduled_clear_cache
    try:
        prompt = mx.array(tokens, dtype=mx.uint32)
        start = time.perf_counter()
        responses = list(
            stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=generated,
                prefill_step_size=step,
            )
        )
        mx.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        last_token = int(responses[-1].token) if responses else -1
        return elapsed, last_token, calls, clears, mx.get_peak_memory()
    finally:
        mx.clear_cache = original_clear_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", type=int, action="append", required=True)
    parser.add_argument("--clear-every", type=int, action="append", required=True)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--generated", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if any(value <= 0 for value in args.clear_every):
        raise SystemExit("--clear-every values must be positive")

    model, tokenizer = load(args.model_dir)
    print(
        "MiniMLX prefill clear-cadence experiment "
        f"(step={args.step}, generated={args.generated}, "
        f"warmup={args.warmup}, repeats={args.repeats})"
    )
    for token_count in args.tokens:
        tokens = [index % 1000 for index in range(token_count)]
        reference_token = None
        for cadence in args.clear_every:
            for _ in range(args.warmup):
                run_once(model, tokenizer, tokens, args.step, args.generated, cadence)
            samples = [
                run_once(model, tokenizer, tokens, args.step, args.generated, cadence)
                for _ in range(args.repeats)
            ]
            elapsed = statistics.median(item[0] for item in samples)
            last_tokens = [item[1] for item in samples]
            callback_calls = statistics.median(item[2] for item in samples)
            clear_calls = statistics.median(item[3] for item in samples)
            peak = statistics.median(item[4] for item in samples)
            if reference_token is None:
                reference_token = last_tokens[0]
            print(
                f"tokens={token_count} clear_every={cadence} "
                f"total={elapsed:.1f} ms peak_memory={peak / (1024**3):.3f} GiB "
                f"callback_calls={callback_calls} actual_clears={clear_calls} "
                f"last_token={last_tokens[0]} "
                f"stable={len(set(last_tokens)) == 1} "
                f"matches_reference={last_tokens[0] == reference_token}"
            )


if __name__ == "__main__":
    main()
