#!/usr/bin/env python3
"""Run a single explicitly selected MiniMLX benchmark suite."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minimlx_runtime.benchmark import BenchmarkPolicy, measure, policy_metadata, summarize  # noqa: E402
from minimlx_runtime.runtime import Runtime  # noqa: E402


def revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", choices=("e2e",), required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = BenchmarkPolicy(args.warmup, args.repeats, "median")
    runtime = Runtime()
    runtime.load(args.model)

    decode_samples: list[float] = []
    prefill_samples: list[float] = []
    calls = 0
    def run_once():
        nonlocal calls
        calls += 1
        value = runtime.generate(args.prompt, max_tokens=args.max_tokens)
        if calls > policy.warmup:
            decode_samples.append(value.decode_ms)
            prefill_samples.append(value.prefill_ms)
        return value

    result, wall_samples = measure(policy, run_once)
    record = {
        "suite": args.suite,
        "model": str(Path(args.model).expanduser().resolve()),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "policy": policy_metadata(policy),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "revision": revision(),
        },
        "result": {
            "prefill_ms": summarize(prefill_samples),
            "decode_ms": summarize(decode_samples),
            "wall_ms": summarize(wall_samples),
            "decode_tokens_per_second": args.max_tokens * 1000.0 / summarize(decode_samples),
            "generated_tokens": len(result.generated_tokens),
        },
    }
    encoded = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
