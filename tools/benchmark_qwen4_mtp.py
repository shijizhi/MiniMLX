#!/usr/bin/env python3
"""Measure Qwen4 MTP across code prompts, without ordinary decoding."""
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--draft-total", type=int, help="verify window size (draft tokens plus one)")
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 124017, 134312])
    args = parser.parse_args()
    if min(args.context, args.tokens, args.step, args.repeat) <= 0:
        parser.error("context, tokens, step and repeat must be positive")
    os.environ["MINIMLX_FORCE_TEXT_RUNTIME"] = "1"
    os.environ["MINIMLX_RAW_COMPLETION"] = "1"
    os.environ["MINIMLX_IGNORE_EOS"] = "1"
    os.environ.pop("MINIMLX_DISABLE_EMBEDDED_MTP", None)
    from minimlx_runtime.runtime import Runtime
    from mlx_vlm.speculative.common import speculative_stats_snapshot, speculative_stats_since
    import mlx.core as mx

    model = Path(args.model).expanduser()
    if (model / ".minimlx-ssd/.minimlx-ssd-manifest.json").is_file():
        model = model / ".minimlx-ssd"
    runtime = Runtime()
    print(f"Loading {model}", flush=True)
    runtime.load(str(model))
    adapter = runtime.adapter
    if not getattr(adapter, "_mtp_enabled", False):
        raise RuntimeError("Embedded MTP is not enabled")
    if args.draft_total is not None:
        if args.draft_total < 2:
            parser.error("draft-total must be at least 2")
        adapter._mtp_draft_model.default_runtime_block_size = args.draft_total
    corpus = (Path(__file__).resolve().parents[1] / "Resources/BenchmarkCorpus/code_python.txt").read_text()
    if any(offset < 0 or offset >= len(corpus) for offset in args.offsets):
        parser.error("every offset must be inside the code corpus")

    def prompt_at(offset):
        part = corpus[offset:]
        return part * (args.context * 8 // len(part) + 2)

    print("Warming up MTP...", flush=True)
    runtime.generate(prompt_at(args.offsets[0]), context_tokens=args.context,
                     max_tokens=8, prefill_step_size=args.step)
    rates = []
    for offset in args.offsets:
        previous = None
        for repetition in range(args.repeat):
            snapshot = speculative_stats_snapshot(adapter._mtp_draft_model)
            start = time.perf_counter()
            result = runtime.generate(prompt_at(offset), context_tokens=args.context,
                                      max_tokens=args.tokens, prefill_step_size=args.step)
            elapsed = time.perf_counter() - start
            rounds, accepted, drafted = speculative_stats_since(adapter._mtp_draft_model, snapshot)
            count = len(result.generated_tokens)
            rate = count * 1000 / result.decode_ms if result.decode_ms else 0.0
            match = previous is None or previous == result.generated_tokens
            rates.append(rate)
            print(f"offset={offset} run={repetition + 1} tokens={count} tps={rate:.2f} "
                  f"decode_s={result.decode_ms / 1000:.2f} wall_s={elapsed:.2f} "
                  f"rounds={rounds} accepted={accepted}/{drafted} repeated_tokens_match={match}", flush=True)
            if not match:
                raise RuntimeError("Repeated greedy MTP output differs")
            previous = result.generated_tokens
    print(f"summary runs={len(rates)} min={min(rates):.2f} median={statistics.median(rates):.2f} "
          f"max={max(rates):.2f} mlx_peak_gib={mx.get_peak_memory() / 1024**3:.2f}", flush=True)


if __name__ == "__main__":
    main()
