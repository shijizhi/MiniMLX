#!/usr/bin/env python3
"""Compare token-id prefill with one-shot input-embedding prefill."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate


def run_once(model, tokenizer, prompt_ids, step: int, use_embeddings: bool):
    embeddings = None
    start = time.perf_counter()
    if use_embeddings:
        embed_tokens = model.language_model.model.embed_tokens
        embeddings = embed_tokens(prompt_ids[None])[0]
        mx.eval(embeddings)
    responses = list(
        stream_generate(
            model,
            tokenizer,
            prompt_ids,
            max_tokens=1,
            prefill_step_size=step,
            input_embeddings=embeddings,
        )
    )
    mx.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    token = int(responses[-1].token) if responses else -1
    return elapsed, token


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
        "MiniMLX prefill input-embedding experiment "
        f"(step={args.step}, warmup={args.warmup}, repeats={args.repeats})"
    )
    for token_count in args.tokens:
        prompt_ids = mx.array(
            [index % 1000 for index in range(token_count)], dtype=mx.uint32
        )
        for _ in range(args.warmup):
            run_once(model, tokenizer, prompt_ids, args.step, False)
            run_once(model, tokenizer, prompt_ids, args.step, True)
        stock = [
            run_once(model, tokenizer, prompt_ids, args.step, False)
            for _ in range(args.repeats)
        ]
        embedded = [
            run_once(model, tokenizer, prompt_ids, args.step, True)
            for _ in range(args.repeats)
        ]
        stock_ms = statistics.median(item[0] for item in stock)
        embedded_ms = statistics.median(item[0] for item in embedded)
        stock_tokens = [item[1] for item in stock]
        embedded_tokens = [item[1] for item in embedded]
        print(
            f"tokens={token_count} stock={stock_ms:.1f} ms "
            f"embedded={embedded_ms:.1f} ms "
            f"saving={stock_ms - embedded_ms:+.1f} ms "
            f"stock_token={stock_tokens[0]} embedded_token={embedded_tokens[0]} "
            f"correct={stock_tokens == embedded_tokens}"
        )


if __name__ == "__main__":
    main()
