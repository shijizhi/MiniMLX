#!/usr/bin/env python3
"""Compare Qwen4 batch MTP against greedy autoregressive output."""
from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Original model or its .minimlx-ssd directory")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--step", type=int, default=512)
    parser.add_argument("--serial-verify", action="store_true")
    parser.add_argument("--mtp-first", action="store_true", help="measure MTP before autoregressive decoding")
    parser.add_argument("--repeat-baseline", action="store_true", help="compare two ordinary decodes before MTP")
    parser.add_argument("--corpus", choices=("code_python", "code_mixed", "novel_en", "novel_ja", "novel_ko"), default="code_python")
    parser.add_argument("--corpus-offset", type=int, default=0, help="character offset into the selected corpus")
    args = parser.parse_args()
    if min(args.context, args.tokens, args.step) <= 0:
        parser.error("context, tokens and step must be positive")

    os.environ["MINIMLX_QWEN4_MTP_BATCH_VERIFY"] = "0" if args.serial_verify else "1"
    os.environ["MINIMLX_FORCE_TEXT_RUNTIME"] = "1"
    os.environ["MINIMLX_RAW_COMPLETION"] = "1"
    os.environ["MINIMLX_IGNORE_EOS"] = "1"
    os.environ.pop("MINIMLX_DISABLE_EMBEDDED_MTP", None)
    from minimlx_runtime.runtime import Runtime

    runtime = Runtime()
    path = Path(args.model).expanduser()
    paged = path / ".minimlx-ssd"
    if (paged / ".minimlx-ssd-manifest.json").is_file():
        path = paged
    print(f"Loading model: {path}", flush=True)
    runtime.load(str(path))
    adapter = runtime.adapter
    expected_verifier = "serial_exact" if args.serial_verify else "batch_exact"
    if not getattr(adapter, "_mtp_enabled", False) or getattr(adapter, "_mtp_verifier", None) != expected_verifier:
        raise RuntimeError(f"The model did not enable {expected_verifier} embedded Qwen4 MTP")
    corpus = (Path(__file__).resolve().parents[1] / "Resources/BenchmarkCorpus" / f"{args.corpus}.txt").read_text()
    if not 0 <= args.corpus_offset < len(corpus):
        parser.error("corpus offset must be within the corpus")
    corpus = corpus[args.corpus_offset:]
    prompt = (corpus * (args.context * 8 // max(len(corpus), 1) + 2))

    def cache_snapshot():
        store = getattr(adapter.model, "moe_offload_store", None)
        stats = store.stats() if store is not None else {"hits": 0, "misses": 0, "evictions": 0}
        slots = [getattr(module, "cache", None) for _, module in adapter.model.named_modules()]
        slots = [cache for cache in slots if getattr(cache, "moe_offload_cache", False)]
        return {**stats, "slot_hits": sum(c.hits for c in slots), "slot_misses": sum(c.misses for c in slots),
                "slot_bytes_read": sum(c.fetched_bytes for c in slots)}

    def run(enabled: bool, tokens: int):
        adapter._mtp_enabled = enabled
        store = getattr(adapter.model, "moe_offload_store", None)
        before = cache_snapshot() if store is not None else None
        first_token = None
        dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
        stream = dispatch.stream_generate

        def trace_stream(*args, **kwargs):
            nonlocal first_token
            for item in stream(*args, **kwargs):
                if first_token is None and store is not None:
                    first_token = cache_snapshot()
                yield item

        start = time.perf_counter()
        with patch.object(dispatch, "stream_generate", trace_stream):
            result = runtime.generate(prompt, context_tokens=args.context, max_tokens=tokens, prefill_step_size=args.step)
        elapsed = time.perf_counter() - start
        after = cache_snapshot() if store is not None else None
        fields = ("hits", "misses", "evictions", "slot_hits", "slot_misses", "slot_bytes_read")
        stats = None if before is None else {
            "total": {key: after[key] - before[key] for key in fields},
            "after_first_token": None if first_token is None else {key: after[key] - first_token[key] for key in fields},
        }
        return result, elapsed, stats

    # Compile both routes before the comparison, using the same prompt.
    print("Warming up autoregressive decoding...", flush=True)
    run(False, 8)
    print("Warming up batch MTP...", flush=True)
    run(True, 8)
    from mlx_vlm.speculative.common import speculative_stats_snapshot, speculative_stats_since
    def measure_mtp():
        snapshot = speculative_stats_snapshot(adapter._mtp_draft_model)
        print("Measuring batch MTP...", flush=True)
        result = run(True, args.tokens)
        return (*result, speculative_stats_since(adapter._mtp_draft_model, snapshot))

    def measure_baseline():
        print("Measuring autoregressive decoding...", flush=True)
        return run(False, args.tokens)

    if args.mtp_first:
        batch, batch_wall, batch_cache, (rounds, accepted, drafted) = measure_mtp()
        baseline, baseline_wall, baseline_cache = measure_baseline()
    else:
        baseline, baseline_wall, baseline_cache = measure_baseline()
        if args.repeat_baseline:
            repeated, _, _ = measure_baseline()
            first = next((i for i, (a, b) in enumerate(zip(baseline.generated_tokens, repeated.generated_tokens)) if a != b), None)
            print(f"autoregressive_repeat_match={baseline.generated_tokens == repeated.generated_tokens} first_mismatch={first}", flush=True)
        batch, batch_wall, batch_cache, (rounds, accepted, drafted) = measure_mtp()
    reference = baseline.generated_tokens
    actual = batch.generated_tokens
    mismatch = next((i for i, (a, b) in enumerate(zip(reference, actual)) if a != b), None)
    if mismatch is None and len(reference) != len(actual):
        mismatch = min(len(reference), len(actual))
    print(f"corpus={args.corpus} offset={args.corpus_offset} context={args.context} output_tokens={args.tokens} expert_cache_gb={os.environ.get('MINIMLX_EXPERT_CACHE_GB') or '12 (default)'}")
    print(f"autoregressive: wall={baseline_wall:.2f}s decode={baseline.decode_ms / 1000:.2f}s tokens={len(reference)} tps={len(reference) * 1000 / baseline.decode_ms if baseline.decode_ms else 0:.2f}")
    print(f"batch MTP: wall={batch_wall:.2f}s decode={batch.decode_ms / 1000:.2f}s tokens={len(actual)} tps={len(actual) * 1000 / batch.decode_ms if batch.decode_ms else 0:.2f}")
    print(f"expert cache (prefill and decode): autoregressive={baseline_cache} batch_mtp={batch_cache}")
    print(f"MTP rounds={rounds} drafted={drafted} accepted={accepted}")
    import mlx.core as mx
    print(f"mlx_peak_gib={mx.get_peak_memory() / (1024 ** 3):.2f}")
    verifier = getattr(adapter.model, "_minimlx_qwen4_batch_verifier", None)
    if verifier is not None:
        print(f"projection_mode={verifier.projection_mode}")
    print(f"greedy_tokens_match={mismatch is None} first_mismatch={mismatch}")
    if mismatch is not None:
        print(f"reference={reference[max(0, mismatch - 3):mismatch + 4]}")
        print(f"batch={actual[max(0, mismatch - 3):mismatch + 4]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
