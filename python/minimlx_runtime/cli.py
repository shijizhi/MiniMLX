from __future__ import annotations

import argparse
import json

from .api import Worker, serve_unix
from .runtime import Runtime


def _runtime(_args) -> Runtime:
    return Runtime()


def main() -> None:
    parser = argparse.ArgumentParser(prog="minimlx-runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("model_dir")
    generate.add_argument("prompt")
    generate.add_argument("--max-tokens", type=int, default=16)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("model_dir")
    benchmark.add_argument("--prompt", default="Hello")
    benchmark.add_argument("--max-tokens", type=int, default=16)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--repeats", type=int, default=5)
    worker = subparsers.add_parser("worker")
    worker.add_argument("model_dir")
    worker.add_argument("--socket", help="Unix domain socket path")
    args = parser.parse_args()
    runtime = _runtime(args)
    runtime.load(args.model_dir)
    if args.command == "generate":
        result = runtime.generate(args.prompt, max_tokens=args.max_tokens)
        print(json.dumps({
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "prefill_ms": result.prefill_ms,
            "decode_ms": result.decode_ms,
            "decode_tokens_per_second": result.decode_tokens_per_second,
        }, ensure_ascii=False, indent=2))
    elif args.command == "benchmark":
        for _ in range(args.warmup):
            runtime.generate(args.prompt, max_tokens=args.max_tokens)
        samples = []
        prefill_samples = []
        for _ in range(args.repeats):
            result = runtime.generate(args.prompt, max_tokens=args.max_tokens)
            samples.append(result.decode_ms)
            prefill_samples.append(result.prefill_ms)
        samples.sort()
        prefill_samples.sort()
        decode_ms = samples[len(samples) // 2]
        prefill_ms = prefill_samples[len(prefill_samples) // 2]
        print(
            "MiniMLX runtime benchmark "
            f"(warmup={args.warmup}, repeats={args.repeats}, statistic=median)"
        )
        print(
            f"prefill={prefill_ms:.1f} ms, "
            f"decode={decode_ms:.1f} ms / {args.max_tokens} tokens, "
            f"decode_tps={args.max_tokens * 1000.0 / decode_ms:.2f} tok/s"
        )
    else:
        instance = Worker(runtime)
        if args.socket:
            serve_unix(instance, args.socket)
        else:
            instance.run_stdio()


if __name__ == "__main__":
    main()
