#!/usr/bin/env python3
"""Measure safe greedy decode without the unnecessary logsumexp.

For greedy decoding, argmax(logits) and argmax(logits - logsumexp(logits))
are identical.  This experiment measures only that inference optimization and
does not alter MiniMLX or mlx-lm defaults.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def prefill(model, prompt, step):
    cache = model.make_cache()
    for start in range(0, prompt.shape[1], step):
        logits = model(prompt[:, start : start + step], cache=cache)
        mx.eval(logits, [c.state for c in cache])
    return cache


def decode_stock(model, cache, token, steps):
    sampled = None
    for _ in range(steps):
        logits = model(token, cache=cache)[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampled = mx.argmax(logprobs, axis=-1)
        mx.eval(sampled, logprobs, [c.state for c in cache])
    return sampled


def decode_direct(model, cache, token, steps):
    sampled = None
    for _ in range(steps):
        logits = model(token, cache=cache)[:, -1, :]
        sampled = mx.argmax(logits, axis=-1)
        mx.eval(sampled, logits, [c.state for c in cache])
    return sampled


def timed(model, prompt, token, prefill_step, steps, direct):
    cache = prefill(model, prompt, prefill_step)
    start = time.perf_counter()
    if direct:
        sampled = decode_direct(model, cache, token, steps)
    else:
        sampled = decode_stock(model, cache, token, steps)
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed / steps, sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    prompt = mx.full((1, args.context), 9419, dtype=mx.uint32)
    token = mx.array([[9419]], dtype=mx.uint32)

    stock_token = decode_stock(
        model,
        prefill(model, prompt, args.prefill_step),
        token,
        args.decode_tokens,
    )
    direct_token = decode_direct(
        model,
        prefill(model, prompt, args.prefill_step),
        token,
        args.decode_tokens,
    )
    mx.eval(stock_token, direct_token)
    token_match = bool(mx.all(stock_token == direct_token).item())

    for _ in range(args.warmup):
        timed(
            model,
            prompt,
            token,
            args.prefill_step,
            args.decode_tokens,
            direct=False,
        )
        timed(
            model,
            prompt,
            token,
            args.prefill_step,
            args.decode_tokens,
            direct=True,
        )

    stock_samples = []
    direct_samples = []
    for _ in range(args.repeats):
        stock_ms, _ = timed(
            model,
            prompt,
            token,
            args.prefill_step,
            args.decode_tokens,
            direct=False,
        )
        direct_ms, _ = timed(
            model,
            prompt,
            token,
            args.prefill_step,
            args.decode_tokens,
            direct=True,
        )
        stock_samples.append(stock_ms)
        direct_samples.append(direct_ms)

    stock_ms = median(stock_samples)
    direct_ms = median(direct_samples)
    print("MiniMLX greedy decode skip-logsumexp experiment")
    print(
        f"context={args.context}, prefill_step={args.prefill_step}, "
        f"decode_tokens={args.decode_tokens}, warmup={args.warmup}, "
        f"repeats={args.repeats}"
    )
    print(f"token_sequence_match={token_match}")
    print(f"stock_greedy_step={stock_ms:.3f} ms")
    print(f"direct_argmax_step={direct_ms:.3f} ms")
    print(f"speedup={stock_ms / direct_ms:.2f}x")


if __name__ == "__main__":
    main()
