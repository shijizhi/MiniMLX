#!/usr/bin/env python3
"""Exercise the production Runtime with an extended prompt prefix."""

from __future__ import annotations

import argparse
import time

from minimlx_runtime.runtime import Runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--session-id", default="experiment-session")
    args = parser.parse_args()

    runtime = Runtime()
    runtime.load(args.model_dir)

    start = time.perf_counter()
    first = runtime.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        context_tokens=args.prefix_tokens,
        session_id=args.session_id,
    )
    first_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    second = runtime.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        context_tokens=args.prefix_tokens + args.suffix_tokens,
        session_id=args.session_id,
    )
    second_ms = (time.perf_counter() - start) * 1000.0

    print("MiniMLX production prefix-cache extension experiment")
    print(
        f"prefix={args.prefix_tokens}, suffix={args.suffix_tokens}, "
        f"max_tokens={args.max_tokens}"
    )
    print(f"first_request_total={first_ms:.3f} ms")
    print(f"second_request_total={second_ms:.3f} ms")
    print(f"second_request_prefill={second.prefill_ms:.3f} ms")
    print(f"speedup_total={first_ms / second_ms:.2f}x")
    print(f"first_prompt_tokens={len(first.prompt_tokens)}")
    print(f"second_prompt_tokens={len(second.prompt_tokens)}")


if __name__ == "__main__":
    main()
