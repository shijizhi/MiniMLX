#!/usr/bin/env python3
"""Measure MLX-LM quantized KV cache on the mixed Qwen3.5 attention stack.

This is an isolated experiment.  It does not change MiniMLX runtime defaults.
Only KVCache entries are quantized; Qwen3.5 ArraysCache recurrent states stay
in their original representation.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def prefill(model, prompt, step: int):
    cache = model.make_cache()
    logits = None
    for start in range(0, prompt.shape[1], step):
        logits = model(prompt[:, start : start + step], cache=cache)
        mx.eval(logits, [c.state for c in cache])
    return cache, logits


def quantize_kv_cache(cache, bits: int, group_size: int):
    quantized = []
    converted = 0
    for item in cache:
        if hasattr(item, "to_quantized"):
            quantized.append(item.to_quantized(group_size=group_size, bits=bits))
            converted += 1
        else:
            quantized.append(item)
    mx.eval([c.state for c in quantized])
    return quantized, converted


def cache_bytes(cache) -> int:
    return sum(int(c.nbytes) for c in cache)


def decode(model, cache, token, steps: int):
    logits = None
    for _ in range(steps):
        logits = model(token, cache=cache)
        mx.eval(logits, [c.state for c in cache])
    return logits


def timed_decode(model, prompt, prefill_step, token, steps, bits, group_size):
    cache, _ = prefill(model, prompt, prefill_step)
    if bits is not None:
        cache, _ = quantize_kv_cache(cache, bits, group_size)
    start = time.perf_counter()
    decode(model, cache, token, steps)
    return (time.perf_counter() - start) * 1000.0 / steps, cache_bytes(cache)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=8, choices=(4, 8))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    prompt = mx.full((1, args.context), 9419, dtype=mx.uint32)
    token = mx.array([[9419]], dtype=mx.uint32)

    stock_cache, stock_prompt_logits = prefill(model, prompt, args.prefill_step)
    quant_cache, _ = prefill(model, prompt, args.prefill_step)
    quant_cache, converted = quantize_kv_cache(
        quant_cache, args.bits, args.group_size
    )
    stock_logits = decode(model, stock_cache, token, args.decode_tokens)
    quant_logits = decode(model, quant_cache, token, args.decode_tokens)
    error = mx.max(mx.abs(stock_logits - quant_logits))
    top1 = mx.argmax(stock_logits[:, -1, :], axis=-1) == mx.argmax(
        quant_logits[:, -1, :], axis=-1
    )
    mx.eval(error, top1)

    for _ in range(args.warmup):
        timed_decode(
            model,
            prompt,
            args.prefill_step,
            token,
            args.decode_tokens,
            None,
            args.group_size,
        )
        timed_decode(
            model,
            prompt,
            args.prefill_step,
            token,
            args.decode_tokens,
            args.bits,
            args.group_size,
        )

    stock_samples = []
    quant_samples = []
    stock_bytes = quant_bytes = 0
    for _ in range(args.repeats):
        stock_ms, stock_bytes = timed_decode(
            model,
            prompt,
            args.prefill_step,
            token,
            args.decode_tokens,
            None,
            args.group_size,
        )
        quant_ms, quant_bytes = timed_decode(
            model,
            prompt,
            args.prefill_step,
            token,
            args.decode_tokens,
            args.bits,
            args.group_size,
        )
        stock_samples.append(stock_ms)
        quant_samples.append(quant_ms)

    stock_ms = median(stock_samples)
    quant_ms = median(quant_samples)
    print("MiniMLX KV cache quantization experiment")
    print(
        f"context={args.context}, prefill_step={args.prefill_step}, "
        f"decode_tokens={args.decode_tokens}, "
        f"bits={args.bits}, group_size={args.group_size}, "
        f"warmup={args.warmup}, repeats={args.repeats}"
    )
    print(f"quantized_kv_layers={converted}")
    print(f"stock_cache_bytes={stock_bytes}")
    print(f"quantized_cache_bytes={quant_bytes}")
    print(f"stock_step={stock_ms:.3f} ms")
    print(f"quantized_step={quant_ms:.3f} ms")
    print(f"speedup={stock_ms / quant_ms:.2f}x")
    print(f"max_abs_error={float(error):.8e}")
    print(f"top1_match={bool(mx.all(top1).item())}")


if __name__ == "__main__":
    main()
