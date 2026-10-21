#!/usr/bin/env python3
"""Compare cold VLM requests with mlx-vlm APC prefix reuse.

This is an opt-in experiment, not a production default. The same loaded model,
prompt, and greedy decode are used for both arms; only the upstream APC manager
is toggled. It is intended for text prompts through a VLM checkpoint such as
Qwen3.8-27B.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path


def run(runtime, request) -> tuple[float, object]:
    started = time.perf_counter()
    result = runtime.infer(request)
    return (time.perf_counter() - started) * 1000.0, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompt_file")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt-chars", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--memory-gb", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.prompt_chars < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("max-tokens/prompt-chars/repeats must be positive and warmup non-negative")
    if args.memory_gb <= 0:
        parser.error("memory-gb must be positive for the memory-only experiment")

    os.environ["MINIMLX_ENABLE_VLM_APC"] = "1"
    os.environ["MINIMLX_VLM_APC_MEMORY_GB"] = str(args.memory_gb)

    from minimlx_runtime.runtime import Runtime
    from minimlx_runtime.types import (
        ContentPart,
        InferenceRequest,
        InputMessage,
        ModelTask,
        Modality,
    )

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")[: args.prompt_chars]
    request = InferenceRequest(
        task=ModelTask.GENERATION,
        messages=(
            InputMessage(
                "user", (ContentPart(Modality.TEXT, text=prompt),)
            ),
        ),
        options={"max_tokens": args.max_tokens, "temperature": 0.0},
    )

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    runtime = Runtime()
    progress(f"Prompt limited to {len(prompt):,} characters")
    progress(f"Loading model: {args.model_dir}")
    runtime.load(args.model_dir)
    progress("Model loaded; starting uncached baseline")
    adapter = runtime.adapter
    manager = getattr(adapter, "_apc_manager", None)
    if manager is None:
        raise RuntimeError("the selected model did not load through the shared VLM APC route")

    try:
        # Compare equal-sized workloads: N+1 uncached requests versus one APC
        # population request plus N reused-prefix requests.
        adapter._apc_manager = None
        for _ in range(args.warmup):
            run(runtime, request)
        cold_ms, cold_result = [], None
        for index in range(args.repeats + 1):
            progress(f"Running uncached workload request {index + 1}/{args.repeats + 1}")
            elapsed, cold_result = run(runtime, request)
            cold_ms.append(elapsed)

        # APC's first request is part of the real workload and must count
        # toward end-to-end cost (it populates the cache but gets no hit).
        adapter._apc_manager = manager
        progress("Populating APC with the prompt (included in workload time)")
        seed_ms, seed_result = run(runtime, request)
        for _ in range(args.warmup):
            run(runtime, request)
        cached_ms, cached_result = [], None
        for index in range(args.repeats):
            progress(f"Running APC-reused workload request {index + 1}/{args.repeats}")
            elapsed, cached_result = run(runtime, request)
            cached_ms.append(elapsed)

        assert cold_result is not None and cached_result is not None
        cold_tokens = cold_result.payload.generated_tokens
        cached_tokens = cached_result.payload.generated_tokens
        stats = manager.stats_snapshot()
        correctness = cold_tokens == cached_tokens
        lookup_hits = int(stats.get("lookups_hit", stats.get("hits", 0)))
        matched_tokens = int(stats.get("matched_tokens", 0))
        hit_ok = lookup_hits > 0 or matched_tokens > 0
        baseline_workload_ms = sum(cold_ms)
        apc_workload_ms = seed_ms + sum(cached_ms)
        print("MiniMLX shared-VLM APC experiment")
        print(f"prompt_tokens={len(cold_result.payload.prompt_tokens)}")
        print(f"max_tokens={args.max_tokens}; cache_reuses={args.repeats}; warmup={args.warmup}")
        print(f"apc_memory_limit_gb={args.memory_gb}; disk_persistence=false")
        print(f"uncached_request_ms_median={statistics.median(cold_ms):.3f}")
        print(f"apc_population_request_ms={seed_ms:.3f}")
        print(f"cached_total_ms_median={statistics.median(cached_ms):.3f}")
        print(f"total_speedup={statistics.median(cold_ms) / statistics.median(cached_ms):.3f}x")
        print(f"uncached_workload_total_ms={baseline_workload_ms:.3f}")
        print(f"apc_workload_total_ms={apc_workload_ms:.3f}")
        print(f"overall_workload_speedup={baseline_workload_ms / apc_workload_ms:.3f}x")
        print(f"overall_workload_time_saved_pct={(1 - apc_workload_ms / baseline_workload_ms) * 100:.1f}%")
        print(f"cold_prefill_ms={cold_result.metrics.get('prefill_ms', 0.0):.3f}")
        print(f"cached_prefill_ms={cached_result.metrics.get('prefill_ms', 0.0):.3f}")
        print(f"apc_lookup_hits={lookup_hits}")
        print(f"apc_matched_tokens={matched_tokens}")
        print(f"apc_resident_bytes={stats.get('resident_bytes', 0)}")
        print(f"cold_generated_tokens={cold_tokens}")
        print(f"cached_generated_tokens={cached_tokens}")
        print(f"apc_population_metrics={seed_result.metrics}")
        print(f"token_correctness={'PASS' if correctness else 'FAIL'}")
        print(f"cache_hit={'PASS' if hit_ok else 'FAIL'}")
        return 0 if correctness and hit_ok else 1
    finally:
        progress("Unloading model")
        runtime.unload()


if __name__ == "__main__":
    raise SystemExit(main())
