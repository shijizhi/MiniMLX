#!/usr/bin/env python3
"""Separate MLX-LM generation orchestration from the model prefill graph."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate


def stock_once(model, tokenizer, prompt_ids, step: int) -> tuple[float, int]:
    start = time.perf_counter()
    responses = list(
        stream_generate(
            model, tokenizer, prompt_ids, max_tokens=1, prefill_step_size=step
        )
    )
    mx.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed, int(responses[-1].token) if responses else -1


def direct_once(model, prompt_ids, step: int) -> tuple[float, int]:
    cache = model.make_cache()
    prompt = prompt_ids[None]
    start = time.perf_counter()
    processed = 0
    while prompt.shape[1] - processed > 1:
        remaining = prompt.shape[1] - processed - 1
        count = min(step, remaining)
        model(prompt[:, processed : processed + count], cache=cache)
        mx.eval([c.state for c in cache])
        processed += count
        mx.clear_cache()
    logits = model(prompt[:, processed:], cache=cache)[:, -1, :]
    mx.eval(logits)
    token = int(mx.argmax(logits, axis=-1).item())
    mx.synchronize()
    return (time.perf_counter() - start) * 1000.0, token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", type=int, action="append", required=True)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer = load(args.model_dir)
    print(
        "MiniMLX prefill orchestration experiment "
        f"(step={args.step}, warmup={args.warmup}, repeats={args.repeats})"
    )
    for token_count in args.tokens:
        token_ids = [index % 1000 for index in range(token_count)]
        prompt_ids = mx.array(token_ids, dtype=mx.uint32)
        for _ in range(args.warmup):
            stock_once(model, tokenizer, prompt_ids, args.step)
            direct_once(model, prompt_ids, args.step)
        stock = [
            stock_once(model, tokenizer, prompt_ids, args.step)
            for _ in range(args.repeats)
        ]
        direct = [
            direct_once(model, prompt_ids, args.step) for _ in range(args.repeats)
        ]
        stock_ms = statistics.median(item[0] for item in stock)
        direct_ms = statistics.median(item[0] for item in direct)
        stock_tokens = [item[1] for item in stock]
        direct_tokens = [item[1] for item in direct]
        print(
            f"tokens={token_count} stock={stock_ms:.1f} ms "
            f"direct_prefill={direct_ms:.1f} ms "
            f"orchestration_delta={stock_ms - direct_ms:+.1f} ms "
            f"stock_token={stock_tokens[0]} direct_token={direct_tokens[0]} "
            f"correct={stock_tokens == direct_tokens}"
        )


if __name__ == "__main__":
    main()
