#!/usr/bin/env python3
"""End-to-end A/B of MiniMLX native Q4 dispatch vs stock MLX QMM."""

from __future__ import annotations

import argparse
import random
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
    parser.add_argument(
        "--mode", choices=("mini-native", "mlx-qmm", "paired-q4"), required=True
    )
    parser.add_argument("--gdn-kernel", choices=("mini", "mlx"), default="mini")
    parser.add_argument(
        "--fa256",
        choices=("native", "mlx", "mlx-fused", "paired"),
        default="mlx",
        help="FA-256 path: Mini native kernel, stock MLX dispatch, or MLX forced fused SDPA",
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--steps", nargs="+", type=int, default=[1024])
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument(
        "--dense-prefill",
        action="store_true",
        help="enable model-wide cached BF16 Q4 MLP weights (uses substantially more memory)",
    )
    args = parser.parse_args()

    import os

    os.environ["MINIMLX_ENABLE_Q4_NATIVE_PREFILL"] = (
        "1" if args.mode in ("mini-native", "paired-q4") else "0"
    )
    os.environ["MINIMLX_DISABLE_Q4_NATIVE_RUNTIME"] = "0"
    # Keep the comparison to the generic MLX QMM versus native QMM. The
    # BF16-resident route is measured separately because it has a large
    # model-wide memory cost.
    os.environ["MINIMLX_DISABLE_Q4_PREFILL_DENSE"] = (
        "0" if args.dense_prefill else "1"
    )
    os.environ["MINIMLX_DISABLE_CUSTOM_GDN"] = (
        "0" if args.gdn_kernel == "mini" else "1"
    )
    os.environ["MINIMLX_ENABLE_NATIVE_FA256_PREFILL"] = (
        "1" if args.fa256 in ("native", "paired") else "0"
    )
    os.environ["MINIMLX_DISABLE_NATIVE_FA256_RUNTIME"] = "0"
    os.environ["MINIMLX_FULL_ATTENTION_PREFILL_FORCE_FUSED"] = (
        "1" if args.fa256 == "mlx-fused" else "0"
    )
    prompt = Path(args.prompt_file).read_text()
    runtime = Runtime()
    runtime.load(args.model_dir)
    if args.mode in ("mini-native", "paired-q4"):
        # The VLM backend shares MLX quantized projections but does not use
        # Qwen35Adapter's mlx-lm installer, so install the opt-in dispatcher
        # directly on the loaded language model for a real kernel A/B.
        from minimlx_runtime.kernels.q4_native import install_q4_native_prefill

        installed = install_q4_native_prefill(runtime.adapter.model, min_tokens=1024)
        print(f"[q4-native] eligible projections installed={installed}", flush=True)
    if args.fa256 in ("native", "paired"):
        # The Qwen3.8 checkpoint is detected as a VLM and loaded through the
        # shared mlx-vlm backend. Qwen35Adapter installs the native FA256 hook
        # on its mlx-lm load path, so explicitly install it here after mlx-vlm
        # has imported its model base; this is an A/B harness correction only.
        from minimlx_runtime.kernels.fa256 import install_native_fa256_prefill

        install_native_fa256_prefill()
    print(
        f"Q4 dispatch end-to-end A/B mode={args.mode} warmup={args.warmup} "
        f"repeats={args.repeats} steps={args.steps} gdn={args.gdn_kernel} "
        f"fa256={args.fa256}"
    )
    try:
        for step in args.steps:
            for context in args.contexts:
                if args.fa256 == "paired" or args.mode == "paired-q4":
                    random.seed(20261013 + context + step)
                    q4_pair = args.mode == "paired-q4"
                    timings = {"native": [], "stock": []}
                    tokens = {"native": [], "stock": []}
                    prompt_counts = []
                    for mode in ("native", "stock"):
                        if q4_pair:
                            os.environ["MINIMLX_DISABLE_Q4_NATIVE_RUNTIME"] = (
                                "0" if mode == "native" else "1"
                            )
                        else:
                            os.environ["MINIMLX_DISABLE_NATIVE_FA256_RUNTIME"] = (
                                "0" if mode == "native" else "1"
                            )
                        runtime.generate(
                            prompt,
                            max_tokens=args.warmup,
                            context_tokens=context,
                            prefill_step_size=step,
                        )
                    order = ["native", "stock"]
                    for _ in range(args.repeats):
                        random.shuffle(order)
                        for mode in order:
                            if q4_pair:
                                os.environ["MINIMLX_DISABLE_Q4_NATIVE_RUNTIME"] = (
                                    "0" if mode == "native" else "1"
                                )
                            else:
                                os.environ["MINIMLX_DISABLE_NATIVE_FA256_RUNTIME"] = (
                                    "0" if mode == "native" else "1"
                                )
                            start = time.perf_counter()
                            result = runtime.generate(
                                prompt,
                                max_tokens=args.max_tokens,
                                context_tokens=context,
                                prefill_step_size=step,
                            )
                            timings[mode].append(result.prefill_ms)
                            tokens[mode].append(tuple(result.generated_tokens))
                            prompt_counts.append(len(result.prompt_tokens))
                    if set(prompt_counts) != {context}:
                        raise RuntimeError(
                            f"expected exactly {context} prompt tokens, got {sorted(set(prompt_counts))}"
                        )
                    for mode in ("native", "stock"):
                        prefill_ms = statistics.median(timings[mode])
                        print(
                            f"step={step} context={context} path={mode} "
                            f"prefill={prefill_ms:.1f} ms "
                            f"pp_tps={context * 1000 / prefill_ms:.1f} "
                            f"same_tokens={len(set(tokens[mode])) == 1} "
                            f"tokens={tokens[mode][0]}",
                            flush=True,
                        )
                    print(
                        f"step={step} context={context} cross_path_tokens_match="
                        f"{tokens['native'] == tokens['stock']}",
                        flush=True,
                    )
                    continue
                # Warm each exact prompt/chunk shape before collecting timings.
                runtime.generate(
                    prompt,
                    max_tokens=args.warmup,
                    context_tokens=context,
                    prefill_step_size=step,
                )
                samples = []
                generated = []
                prompt_counts = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    result = runtime.generate(
                        prompt,
                        max_tokens=args.max_tokens,
                        context_tokens=context,
                        prefill_step_size=step,
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    samples.append((result.prefill_ms, elapsed_ms))
                    generated.append(tuple(result.generated_tokens))
                    prompt_counts.append(len(result.prompt_tokens))
                if set(prompt_counts) != {context}:
                    raise RuntimeError(
                        f"expected exactly {context} prompt tokens, got {sorted(set(prompt_counts))}"
                    )
                prefill_ms = statistics.median(item[0] for item in samples)
                elapsed_ms = statistics.median(item[1] for item in samples)
                print(
                    f"step={step} context={context} prefill={prefill_ms:.1f} ms "
                    f"pp_tps={context * 1000 / prefill_ms:.1f} "
                    f"e2e={elapsed_ms:.1f} ms generated={generated[0]} "
                    f"same_tokens={len(set(generated)) == 1}"
                )
    finally:
        runtime.unload()


if __name__ == "__main__":
    main()
