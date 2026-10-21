#!/usr/bin/env python3
"""Measure ordinary, uncached Qwen GDN single-request inference.

The end-to-end pass reports prefill/decode/wall time without per-kernel
synchronization. A separate diagnostic pass counts one-token GDN calls but
does not time them. MTP and VLM APC are explicitly disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompt_file")
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "force generation to max-tokens for controlled MiniMLX A/B; only "
            "use when the comparison system can apply the same stop policy"
        ),
    )
    parser.add_argument(
        "--profile-context", type=int, default=4096,
        help="context for the separate GDN call-count diagnostic",
    )
    parser.add_argument("--profile-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 2 or args.profile_tokens < 1:
        parser.error("repeats must be positive; max-tokens must be >=2; profile-tokens positive")
    if args.step < 1 or any(context < 1 for context in args.contexts):
        parser.error("step and contexts must be positive")
    if args.profile_context not in args.contexts:
        parser.error("profile-context must be one of --contexts")

    # Make this a plain autoregressive, no-prefix-reuse test even if the shell
    # already has optional acceleration variables set.
    os.environ.pop("MINIMLX_MTP_DRAFT_MODEL", None)
    os.environ["MINIMLX_ENABLE_VLM_APC"] = "0"

    from minimlx_runtime.runtime import Runtime

    generate_module = None
    original_generate = None
    mlx_lm_module = None
    original_mlx_lm_stream_generate = None
    if args.ignore_eos:
        import importlib
        import mlx_lm as mlx_lm_module
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        generate_module = importlib.import_module("mlx_vlm.generate")
        original_generate = generate_module.generate
        original_mlx_lm_stream_generate = mlx_lm_module.stream_generate

        def generate_without_eos(*positional, **kwargs):
            # A never-stop criterion gives exactly max_tokens. This is useful
            # for within-MiniMLX kernel A/B, but only cross-compare if the
            # other engine can use an equivalent stop policy.
            kwargs["stopping_criteria"] = lambda _token: False
            return original_generate(*positional, **kwargs)

        generate_module.generate = generate_without_eos

        def stream_generate_without_eos(model, tokenizer, *positional, **kwargs):
            if not isinstance(tokenizer, TokenizerWrapper):
                tokenizer = TokenizerWrapper(tokenizer)
            tokenizer.eos_token_ids = ()
            return original_mlx_lm_stream_generate(
                model, tokenizer, *positional, **kwargs
            )

        mlx_lm_module.stream_generate = stream_generate_without_eos

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    runtime = Runtime()
    custom_recurrence = os.environ.get("MINIMLX_GDN_DECODE_CUSTOM_RECURRENCE") == "1"
    fused_qk_norm = os.environ.get("MINIMLX_GDN_DECODE_FUSED_QK_NORM") == "1"
    fused_conv_silu = os.environ.get("MINIMLX_GDN_DECODE_FUSED_CONV_SILU") == "1"
    benchmark_label = "CANDIDATE" if fused_conv_silu else "BASELINE"
    print(
        "Loading model (MTP=off, APC=off, "
        f"GDN_custom_recurrence={'on' if custom_recurrence else 'off'}, "
        f"GDN_fused_qk_norm={'on' if fused_qk_norm else 'off'}, "
        f"GDN_fused_conv_silu={'on' if fused_conv_silu else 'off'})",
        flush=True,
    )
    runtime.load(args.model_dir)
    adapter = runtime.adapter
    if getattr(adapter, "_mtp_enabled", False):
        raise RuntimeError("MTP unexpectedly enabled; refusing to benchmark")
    if getattr(adapter, "_apc_manager", None) is not None:
        raise RuntimeError("APC unexpectedly enabled; refusing to benchmark")

    try:
        for context in args.contexts:
            # Compile/exercise this request shape before timed runs.
            for warmup_index in range(args.warmup):
                result = runtime.generate(
                    prompt,
                    max_tokens=min(8, args.max_tokens),
                    context_tokens=context,
                    prefill_step_size=args.step,
                    session_id=f"single-bench-warmup-{context}-{warmup_index}",
                )
                if len(result.prompt_tokens) != context:
                    raise RuntimeError(
                        f"context={context}: tokenizer produced {len(result.prompt_tokens)} tokens"
                    )

            prefill_ms: list[float] = []
            decode_ms: list[float] = []
            wall_ms: list[float] = []
            token_runs: list[tuple[int, ...]] = []
            for index in range(args.repeats):
                started = time.perf_counter()
                result = runtime.generate(
                    prompt,
                    max_tokens=args.max_tokens,
                    context_tokens=context,
                    prefill_step_size=args.step,
                    session_id=f"single-bench-{context}-{index}",
                )
                wall_ms.append((time.perf_counter() - started) * 1000.0)
                prefill_ms.append(result.prefill_ms)
                decode_ms.append(result.decode_ms)
                token_runs.append(tuple(result.generated_tokens))
                if len(result.prompt_tokens) != context:
                    raise RuntimeError(
                        f"context={context}: tokenizer produced {len(result.prompt_tokens)} tokens"
                    )
            prefill = statistics.median(prefill_ms)
            decode = statistics.median(decode_ms)
            wall = statistics.median(wall_ms)
            output_count = len(token_runs[0])
            token_digest = hashlib.sha256(
                json.dumps(token_runs[0], separators=(",", ":")).encode("ascii")
            ).hexdigest()
            print(
                f"{benchmark_label} context={context} step={args.step} output={output_count} "
                f"prefill_ms={prefill:.1f} decode_ms={decode:.1f} "
                f"decode_tps={output_count * 1000 / decode:.2f} wall_ms={wall:.1f} "
                f"same_tokens={len(set(token_runs)) == 1} "
                f"token_sha256={token_digest} tokens={list(token_runs[0])}",
                flush=True,
            )

        # Diagnostic pass: count one-token calls only. Timing individual MLX
        # calls would force lazy dependencies to materialize and distort them.
        from mlx_vlm.models.qwen3_5 import gated_delta, language

        original = gated_delta.gated_delta_update
        call_count = 0

        def timed_gdn(q, k, v, a, b, A_log, dt_bias, *positional, **kwargs):
            nonlocal call_count
            # The outer cache wrapper delegates to an inner call with
            # cache=None; count that inner recurrence once, not both layers.
            cache = kwargs.get("cache")
            if cache is None and q.ndim == 4 and q.shape[1] == 1:
                result = original(q, k, v, a, b, A_log, dt_bias, *positional, **kwargs)
                call_count += 1
                return result
            return original(q, k, v, a, b, A_log, dt_bias, *positional, **kwargs)

        gated_delta.gated_delta_update = timed_gdn
        language.gated_delta_update = timed_gdn
        try:
            profiled = runtime.generate(
                prompt,
                max_tokens=args.profile_tokens,
                context_tokens=args.profile_context,
                prefill_step_size=args.step,
                session_id=f"single-bench-gdn-profile-{args.profile_context}",
            )
        finally:
            gated_delta.gated_delta_update = original
            language.gated_delta_update = original

        if call_count == 0:
            raise RuntimeError(
                "No one-token GDN calls were observed; inspect the loaded backend/version."
            )
        print(
            f"GDN_CALL_COUNT context={args.profile_context} "
            f"generated={len(profiled.generated_tokens)} calls={call_count} "
            "NOTE=count_only; no per-call timing",
            flush=True,
        )
        if custom_recurrence:
            from minimlx_runtime.kernels.gdn_decode import gdn_decode_custom_recurrence_stats

            routed, fallback = gdn_decode_custom_recurrence_stats()
            print(
                f"GDN_CUSTOM_RECURRENCE routes={routed} fallbacks={fallback}",
                flush=True,
            )
        if fused_qk_norm:
            from minimlx_runtime.kernels.gdn_decode import gdn_decode_fused_qk_norm_stats

            norm_routes, norm_fallbacks = gdn_decode_fused_qk_norm_stats()
            print(
                f"GDN_FUSED_QK_NORM routes={norm_routes} fallbacks={norm_fallbacks}",
                flush=True,
            )
        if fused_conv_silu:
            from minimlx_runtime.kernels.gdn_conv_silu import (
                gdn_decode_conv_silu_stats,
            )

            eligible, routes, fallbacks = gdn_decode_conv_silu_stats()
            print(
                f"GDN_FUSED_CONV_SILU eligible_layers={eligible} "
                f"routes={routes} fallbacks={fallbacks}",
                flush=True,
            )
    finally:
        if generate_module is not None and original_generate is not None:
            generate_module.generate = original_generate
        if mlx_lm_module is not None and original_mlx_lm_stream_generate is not None:
            mlx_lm_module.stream_generate = original_mlx_lm_stream_generate
        print("Unloading model", flush=True)
        runtime.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
