#!/usr/bin/env python3
"""Paired end-to-end prefill chunk A/B for a standard MLX-LM Llama model."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minimlx_runtime.runtime import Runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompt_file")
    parser.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--steps", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_tokens < 1 or args.max_tokens < 2:
        parser.error("repeats/warmup must be positive and max-tokens must be at least 2")
    if len(set(args.steps)) != len(args.steps) or any(step < 1 for step in args.steps):
        parser.error("steps must be unique positive integers")

    prompt = Path(args.prompt_file).read_text()
    runtime = Runtime()
    runtime.load(args.model_dir)
    try:
        print(
            "Llama prefill-step paired A/B "
            f"(contexts={args.contexts}, steps={args.steps}, repeats={args.repeats}, "
            f"warmup_tokens={args.warmup_tokens}, max_tokens={args.max_tokens})",
            flush=True,
        )
        for context in args.contexts:
            for step in args.steps:
                result = runtime.generate(
                    prompt,
                    max_tokens=args.warmup_tokens,
                    context_tokens=context,
                    prefill_step_size=step,
                )
                if len(result.prompt_tokens) != context:
                    raise RuntimeError(
                        f"expected {context} prompt tokens, got {len(result.prompt_tokens)}"
                    )
                print(f"warmup context={context} step={step} complete", flush=True)

            samples = {step: [] for step in args.steps}
            generated = {step: [] for step in args.steps}
            for repeat in range(args.repeats):
                # Reverse the order every other pair to balance warm-up and
                # thermal/frequency drift without changing the prompt or model.
                order = list(args.steps)
                if repeat % 2:
                    order.reverse()
                for step in order:
                    started = time.perf_counter()
                    result = runtime.generate(
                        prompt,
                        max_tokens=args.max_tokens,
                        context_tokens=context,
                        prefill_step_size=step,
                    )
                    wall_ms = (time.perf_counter() - started) * 1000.0
                    samples[step].append(
                        (result.prefill_ms, result.decode_ms, wall_ms)
                    )
                    generated[step].append(tuple(result.generated_tokens))
                    print(
                        f"run context={context} step={step} repeat={repeat + 1} "
                        f"TTFT={result.prefill_ms:.1f}ms "
                        f"decode={result.decode_ms:.1f}ms E2E={wall_ms:.1f}ms",
                        flush=True,
                    )

            medians = {
                step: [
                    statistics.median(row[index] for row in samples[step])
                    for index in range(3)
                ]
                for step in args.steps
            }
            reference = medians[args.steps[0]]
            for step in args.steps:
                values = medians[step]
                print(
                    f"median context={context} step={step} "
                    f"TTFT={values[0]:.1f}ms decode={values[1]:.1f}ms "
                    f"E2E={values[2]:.1f}ms "
                    f"TTFT_vs_step_{args.steps[0]}={(values[0] / reference[0] - 1) * 100:+.2f}% "
                    f"E2E_vs_step_{args.steps[0]}={(values[2] / reference[2] - 1) * 100:+.2f}% "
                    f"stable_tokens={len(set(generated[step])) == 1} "
                    f"matches_reference={generated[step] == generated[args.steps[0]]}",
                    flush=True,
                )
    finally:
        runtime.unload()


if __name__ == "__main__":
    main()
