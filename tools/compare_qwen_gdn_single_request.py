#!/usr/bin/env python3
"""Compare stock, custom-recurrence, and fused-Q/K-norm Qwen GDN decode."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tools" / "benchmark_qwen_gdn_single_request.py"
RESULT_RE = re.compile(
    r"BASELINE context=(?P<context>\d+).*?output=(?P<output>\d+) "
    r"prefill_ms=(?P<prefill>[\d.]+) decode_ms=(?P<decode>[\d.]+) "
    r"decode_tps=(?P<tps>[\d.]+).*?token_sha256=(?P<digest>[0-9a-f]+)"
)


def run_benchmark(label: str, strategy: str, command: list[str], base_env: dict[str, str]):
    env = base_env.copy()
    env.pop("MINIMLX_GDN_DECODE_CUSTOM_RECURRENCE", None)
    env.pop("MINIMLX_GDN_DECODE_FUSED_QK_NORM", None)
    if strategy in {"recurrence", "fused_qk_norm"}:
        env["MINIMLX_GDN_DECODE_CUSTOM_RECURRENCE"] = "1"
    if strategy == "fused_qk_norm":
        env["MINIMLX_GDN_DECODE_FUSED_QK_NORM"] = "1"
    print(f"\n=== {label}: starting in a clean process ===", flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)
    return process.wait(), "".join(output_lines)


def parse_results(output: str) -> dict[int, dict[str, str]]:
    return {
        int(match.group("context")): match.groupdict()
        for match in RESULT_RE.finditer(output)
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run stock, recurrence-only, and fused-Q/K-norm MiniMLX benchmarks in clean processes."
    )
    parser.add_argument("model_dir")
    parser.add_argument("prompt_file")
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-context", type=int, default=4096)
    parser.add_argument("--profile-tokens", type=int, default=8)
    parser.add_argument(
        "--allow-eos",
        action="store_true",
        help="allow early EOS; by default both runs generate exactly --max-tokens",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 2 or args.step < 1:
        parser.error("repeats and step must be positive; max-tokens must be >= 2")
    if args.profile_context not in args.contexts:
        parser.error("profile-context must be included in --contexts")

    command = [
        sys.executable,
        str(BENCHMARK),
        args.model_dir,
        args.prompt_file,
        "--contexts",
        *(str(value) for value in args.contexts),
        "--step",
        str(args.step),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--profile-context",
        str(args.profile_context),
        "--profile-tokens",
        str(args.profile_tokens),
    ]
    if not args.allow_eos:
        command.append("--ignore-eos")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    baseline_code, baseline_output = run_benchmark("BASELINE", "baseline", command, env)
    if baseline_code != 0:
        print("\nBaseline failed; stopping before the experimental run.", flush=True)
        return baseline_code

    custom_code, custom_output = run_benchmark("CUSTOM RECURRENCE", "recurrence", command, env)
    if custom_code != 0:
        print("Custom recurrence run failed; stopping before Q/K norm fusion.", flush=True)
        return custom_code
    fused_code, fused_output = run_benchmark("FUSED Q/K NORM", "fused_qk_norm", command, env)

    baseline_results = parse_results(baseline_output)
    custom_results = parse_results(custom_output)
    fused_results = parse_results(fused_output)
    print("\n=== A/B SUMMARY ===")
    if not baseline_results or not custom_results or not fused_results:
        print("Could not parse benchmark summaries; inspect the logs above.")
    else:
        for context in args.contexts:
            base = baseline_results.get(context)
            candidate = custom_results.get(context)
            fused = fused_results.get(context)
            if base is None or candidate is None or fused is None:
                print(f"context={context}: missing a result in one run")
                continue
            base_tps = float(base["tps"])
            candidate_tps = float(candidate["tps"])
            fused_tps = float(fused["tps"])
            recurrence_delta = (candidate_tps / base_tps - 1.0) * 100.0
            fused_delta = (fused_tps / candidate_tps - 1.0) * 100.0
            print(
                f"context={context} outputs={base['output']}/{candidate['output']}/{fused['output']} "
                f"decode_tps baseline={base_tps:.2f} recurrence={candidate_tps:.2f} "
                f"fused_qk_norm={fused_tps:.2f} recurrence_delta={recurrence_delta:+.2f}% "
                f"norm_fusion_delta={fused_delta:+.2f}% "
                f"hash_equal={base['digest'] == candidate['digest'] == fused['digest']}"
            )

    return fused_code


if __name__ == "__main__":
    raise SystemExit(main())
