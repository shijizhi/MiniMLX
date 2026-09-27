#!/usr/bin/env python3
"""Measure semantically equivalent prefix-cache reuse without changing production."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load


def clone_cache(model, source):
    """Clone cache state without relying on Python deepcopy of MLX objects."""
    target = model.make_cache()
    for dst, src in zip(target, source):
        state = src.state
        if isinstance(state, (list, tuple)):
            state = type(state)(
                None if value is None else mx.array(value) for value in state
            )
        elif state is not None:
            state = mx.array(state)
        dst.state = state
        for name in ("lengths", "left_padding"):
            if hasattr(src, name) and getattr(src, name) is not None:
                setattr(dst, name, mx.array(getattr(src, name)))
    return target


def evaluate(model, tokens, cache):
    logits = model(tokens[None], cache=cache)
    mx.eval(logits, [c.state for c in cache])
    return logits


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--token", type=int, default=9419)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    prefix = mx.full((args.prefix_tokens,), args.token, dtype=mx.uint32)
    suffix = mx.full((args.suffix_tokens,), args.token, dtype=mx.uint32)

    prefix_cache = model.make_cache()
    evaluate(model, prefix, prefix_cache)
    cached = clone_cache(model, prefix_cache)
    mx.eval([c.state for c in cached])

    # Use an independent split reference.  A one-shot prefix+suffix forward
    # creates a second large attention cache and can exceed the Metal buffer
    # limit at long context lengths.
    reference_cache = model.make_cache()
    evaluate(model, prefix, reference_cache)
    reference_logits = evaluate(model, suffix, reference_cache)
    direct_suffix_logits = evaluate(model, suffix, prefix_cache)
    cloned_suffix_logits = evaluate(model, suffix, clone_cache(model, cached))
    direct_error = mx.max(
        mx.abs(reference_logits[:, -1, :] - direct_suffix_logits[:, -1, :])
    )
    cloned_error = mx.max(
        mx.abs(reference_logits[:, -1, :] - cloned_suffix_logits[:, -1, :])
    )
    reference_last = reference_logits[:, -1, :]
    direct_last = direct_suffix_logits[:, -1, :]
    cloned_last = cloned_suffix_logits[:, -1, :]
    reference_top1 = mx.argmax(reference_last, axis=-1)
    direct_top1 = mx.argmax(direct_last, axis=-1)
    cloned_top1 = mx.argmax(cloned_last, axis=-1)
    reference_top5 = mx.argsort(reference_last, axis=-1)[..., -5:]
    direct_top5 = mx.argsort(direct_last, axis=-1)[..., -5:]
    cloned_top5 = mx.argsort(cloned_last, axis=-1)[..., -5:]
    top1_direct = mx.all(reference_top1 == direct_top1)
    top1_cloned = mx.all(reference_top1 == cloned_top1)
    top5_direct = mx.all(
        mx.sum(reference_top5[..., None] == direct_top5[..., None, :], axis=(-2, -1)) > 0
    )
    top5_cloned = mx.all(
        mx.sum(reference_top5[..., None] == cloned_top5[..., None, :], axis=(-2, -1)) > 0
    )
    mx.eval(direct_error, cloned_error, top1_direct, top1_cloned, top5_direct, top5_cloned)
    print(f"direct_suffix_error={float(direct_error):.8e}")
    print(f"cloned_suffix_error={float(cloned_error):.8e}")
    print(f"top1_match_direct={bool(top1_direct)}")
    print(f"top1_match_cloned={bool(top1_cloned)}")
    print(f"top5_overlap_direct={bool(top5_direct)}")
    print(f"top5_overlap_cloned={bool(top5_cloned)}")
    semantic_ok = bool(top1_cloned) and bool(top5_cloned)
    print(f"semantic_correctness={'PASS' if semantic_ok else 'FAIL'}")

    cold_times: list[float] = []
    warm_times: list[float] = []
    cold_logits = None
    warm_logits = None
    for _ in range(args.warmup):
        cold_cache = model.make_cache()
        evaluate(model, prefix, cold_cache)
        evaluate(model, suffix, cold_cache)
        evaluate(model, suffix, clone_cache(model, cached))

    for _ in range(args.repeats):
        start = time.perf_counter()
        cold_cache = model.make_cache()
        evaluate(model, prefix, cold_cache)
        cold_logits = evaluate(model, suffix, cold_cache)
        cold_times.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        warm_logits = evaluate(model, suffix, clone_cache(model, cached))
        warm_times.append((time.perf_counter() - start) * 1000.0)

    error = mx.max(mx.abs(cold_logits[:, -1, :] - warm_logits[:, -1, :]))
    mx.eval(error)
    cold_ms = median(cold_times)
    warm_ms = median(warm_times)
    print("MiniMLX prefix-cache experiment (split reference)")
    print(f"prefix={args.prefix_tokens}, suffix={args.suffix_tokens}, warmup={args.warmup}, repeats={args.repeats}")
    print(f"cold_request_prefix_plus_suffix={cold_ms:.3f} ms")
    print(f"warm_cached_suffix={warm_ms:.3f} ms")
    print(f"speedup={cold_ms / warm_ms:.2f}x")
    print(f"max_abs_error_last_logits={float(error):.8e}")
    if not semantic_ok:
        raise SystemExit("prefix cache semantic correctness check failed")


if __name__ == "__main__":
    main()
