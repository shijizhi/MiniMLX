#!/usr/bin/env python3
"""Measure skipping intermediate prefill LM-head projections."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def resolve_text_model(model):
    """Return the text-model wrapper for both mlx-lm model layouts."""
    return getattr(model, "language_model", model)


def resolve_backbone(model):
    """Return the transformer backbone without applying the LM head."""
    text_model = resolve_text_model(model)
    backbone = getattr(text_model, "model", None)
    if backbone is None:
        raise RuntimeError("loaded model does not expose a transformer backbone")
    return backbone


def last_logits(model, hidden):
    text_model = resolve_text_model(model)
    hidden = hidden[:, -1:, :]
    if hasattr(text_model, "lm_head"):
        return text_model.lm_head(hidden)
    return text_model.model.embed_tokens.as_linear(hidden)


def stock_prefill(model, tokens, step):
    cache = model.make_cache()
    logits = None
    for start in range(0, tokens.shape[1], step):
        logits = model(tokens[:, start : start + step], cache=cache)
        mx.eval(logits, [c.state for c in cache])
    return logits[:, -1:, :]


def optimized_prefill(model, tokens, step):
    cache = model.make_cache()
    backbone = resolve_backbone(model)
    hidden = None
    last_start = max(0, tokens.shape[1] - step)
    for start in range(0, tokens.shape[1], step):
        hidden = backbone(tokens[:, start : start + step], cache=cache)
        if start < last_start:
            mx.eval(hidden, [c.state for c in cache])
        else:
            logits = last_logits(model, hidden)
            mx.eval(logits, [c.state for c in cache])
            return logits
    raise RuntimeError("prefill produced no final chunk")


def measure(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--token", type=int, default=9419)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    tokens = mx.full((1, args.tokens), args.token, dtype=mx.uint32)
    stock_logits = stock_prefill(model, tokens, args.step)
    optimized_logits = optimized_prefill(model, tokens, args.step)
    error = mx.max(mx.abs(stock_logits - optimized_logits))
    stock_last = stock_logits[:, -1, :]
    optimized_last = optimized_logits[:, -1, :]
    stock_top1 = mx.argmax(stock_last, axis=-1)
    optimized_top1 = mx.argmax(optimized_last, axis=-1)
    stock_top5 = mx.argsort(stock_last, axis=-1)[:, -5:]
    optimized_top5 = mx.argsort(optimized_last, axis=-1)[:, -5:]
    mx.eval(error, stock_top1, optimized_top1, stock_top5, optimized_top5)
    top1_match = bool(mx.all(stock_top1 == optimized_top1).item())
    top5_overlap = bool(
        mx.all(
            mx.sum(
                stock_top5[:, :, None] == optimized_top5[:, None, :],
                axis=(1, 2),
            )
            >= 5
        ).item()
    )

    stock_ms = measure(lambda: stock_prefill(model, tokens, args.step), args.warmup, args.repeats)
    optimized_ms = measure(
        lambda: optimized_prefill(model, tokens, args.step), args.warmup, args.repeats
    )
    print("MiniMLX prefill LM-head suppression experiment")
    print(f"tokens={args.tokens}, step={args.step}, warmup={args.warmup}, repeats={args.repeats}")
    print(f"stock_full_logits={stock_ms:.3f} ms")
    print(f"optimized_last_logits={optimized_ms:.3f} ms")
    print(f"speedup={stock_ms / optimized_ms:.2f}x")
    print(f"max_abs_error_last_logits={float(error):.8e}")
    print(f"top1_match={top1_match}")
    print(f"top5_overlap={top5_overlap}")


if __name__ == "__main__":
    main()
