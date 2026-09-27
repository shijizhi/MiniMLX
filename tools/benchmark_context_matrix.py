#!/usr/bin/env python3
"""Run the final MiniMLX context-length benchmark in one MLX process."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from minimlx_runtime.runtime import Runtime


def apply_runtime_config(config: dict[str, object]) -> None:
    """Apply the same kernel switches that the Rust supervisor passes to its worker."""
    settings = {
        "MINIMLX_DISABLE_Q4_PREFILL_DENSE": config.get(
            "disable_q4_prefill_dense", False
        ),
        "MINIMLX_DISABLE_Q4_MLP_FUSION": config.get(
            "disable_q4_mlp_fusion", False
        ),
        "MINIMLX_ENABLE_Q4_NATIVE_PREFILL": config.get(
            "enable_q4_native_prefill", False
        ),
        "MINIMLX_ENABLE_NATIVE_FA256_PREFILL": config.get(
            "enable_native_fa256_prefill", False
        ),
        "MINIMLX_GREEDY_SKIP_LOGSUMEXP": config.get(
            "greedy_skip_logsumexp", False
        ),
        "MINIMLX_FULL_ATTENTION_DECODE_FORCE_FUSED": config.get(
            "full_attention_decode_force_fused", False
        ),
        "MINIMLX_FULL_ATTENTION_PREFILL_FORCE_FUSED": config.get(
            "full_attention_prefill_force_fused", False
        ),
        "MINIMLX_PREFILL_KEEP_ALLOCATOR_CACHE": config.get(
            "prefill_keep_allocator_cache", False
        ),
    }
    for name, enabled in settings.items():
        os.environ[name] = "1" if enabled else "0"
    os.environ["MINIMLX_FULL_ATTENTION_KV_CACHE_STEP"] = str(
        config.get("full_attention_kv_cache_step", 256)
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config" / "minimlx.local.json"
    config = json.loads(config_path.read_text())
    apply_runtime_config(config)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default=config["model_dir"])
    parser.add_argument(
        "prompt_file",
        nargs="?",
        default=str(root / config["prompt_file"]),
    )
    parser.add_argument("--warmup", type=int, default=config.get("warmup", 32))
    parser.add_argument("--max-tokens", type=int, default=config.get("max_tokens", 128))
    parser.add_argument(
        "--step",
        type=int,
        default=config.get("benchmark_prefill_step_size", 1024),
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 4096, 8192, 16384])
    parser.add_argument(
        "--mtp-verify-ab",
        action="store_true",
        help=(
            "Compare stock per-position, force-fused, and OMLX-style vector-SDPA "
            "MTP attention in one process; checks generated-token equality."
        ),
    )
    parser.add_argument(
        "--mtp-argmax-ab",
        action="store_true",
        help=(
            "Compare positionwise and batched Qwen3.5 MTP target argmax with "
            "fused attention fixed; reports speed and token equality."
        ),
    )
    parser.add_argument(
        "--mtp-forward-ab",
        action="store_true",
        help=(
            "Compare the exact Qwen3.5 speculative verifier with a standard "
            "batched target forward; checks generated-token equality."
        ),
    )
    parser.add_argument(
        "--mtp-gdn-ab",
        action="store_true",
        help="Compare stock and OMLX-derived fused GDN verify prework; checks token equality.",
    )
    args = parser.parse_args()

    experiments = sum(
        bool(value)
        for value in (args.mtp_verify_ab, args.mtp_argmax_ab, args.mtp_forward_ab, args.mtp_gdn_ab)
    )
    if experiments > 1:
        parser.error("MTP A/B flags are separate experiments")
    if args.mtp_argmax_ab:
        if not os.environ.get("MINIMLX_MTP_DRAFT_MODEL"):
            parser.error("--mtp-argmax-ab requires MINIMLX_MTP_DRAFT_MODEL")
        os.environ["MINIMLX_MTP_VERIFY_ARGMAX"] = "positionwise"
        os.environ["MINIMLX_MTP_FORCE_FUSED_VERIFY"] = "1"

    if args.mtp_forward_ab and not os.environ.get("MINIMLX_MTP_DRAFT_MODEL"):
        parser.error("--mtp-forward-ab requires MINIMLX_MTP_DRAFT_MODEL")

    if args.mtp_gdn_ab and not os.environ.get("MINIMLX_MTP_DRAFT_MODEL"):
        parser.error("--mtp-gdn-ab requires MINIMLX_MTP_DRAFT_MODEL")

    if args.mtp_verify_ab:
        if not os.environ.get("MINIMLX_MTP_DRAFT_MODEL"):
            parser.error("--mtp-verify-ab requires MINIMLX_MTP_DRAFT_MODEL")
        os.environ["MINIMLX_MTP_VERIFY_AB"] = "1"
        os.environ["MINIMLX_MTP_FORCE_FUSED_VERIFY"] = "0"

    prompt = Path(args.prompt_file).read_text()
    runtime = Runtime()
    runtime.load(args.model_dir)

    def timed_generate(*args, **kwargs):
        start = time.perf_counter()
        result = runtime.generate(*args, **kwargs)
        return result, (time.perf_counter() - start) * 1000.0

    print(
        "MiniMLX context matrix "
        f"(warmup={args.warmup} tokens at context=2048, measured_runs=1, step={args.step})"
    )
    print(
        "runtime kernels "
        f"(q4_native={config.get('enable_q4_native_prefill', False)}, "
        f"fa256_native={config.get('enable_native_fa256_prefill', False)}, "
        f"q4_dense_disabled={config.get('disable_q4_prefill_dense', False)}, "
        f"q4_mlp_fusion_disabled={config.get('disable_q4_mlp_fusion', False)})"
    )
    adapter = runtime.adapter
    mtp_enabled = bool(getattr(adapter, "_mtp_enabled", False))
    mtp_draft_path = os.environ.get("MINIMLX_MTP_DRAFT_MODEL")
    if mtp_draft_path and not mtp_enabled:
        raise RuntimeError(
            "MINIMLX_MTP_DRAFT_MODEL was set, but the loaded adapter did not "
            "enable MTP; refusing to report an autoregressive run as an MTP benchmark."
        )
    print(
        "decode backend "
        f"({'Qwen3.5 MTP' if mtp_enabled else 'autoregressive'}"
        + (
            f", draft={Path(mtp_draft_path).name}, "
            f"block_size={getattr(getattr(adapter._mtp_draft_model, 'config', None), 'block_size', 'default')}"
            if mtp_enabled and mtp_draft_path
            else ""
        )
        + ")"
    )
    # Match oMLX Quick warm-up: compile/exercise one 2K-context request before
    # timing the matrix, rather than warming every measured context separately.
    runtime.generate(
        prompt,
        max_tokens=args.warmup,
        context_tokens=2048,
        prefill_step_size=args.step,
    )
    if args.mtp_verify_ab:
        # Compile both source-derived candidates before collecting measurements.
        os.environ["MINIMLX_MTP_FORCE_FUSED_VERIFY"] = "1"
        os.environ["MINIMLX_MTP_VERIFY_SDPA"] = "fused"
        runtime.generate(
            prompt,
            max_tokens=args.warmup,
            context_tokens=2048,
            prefill_step_size=args.step,
        )
        os.environ["MINIMLX_MTP_VERIFY_SDPA"] = "vector"
        runtime.generate(
            prompt,
            max_tokens=args.warmup,
            context_tokens=2048,
            prefill_step_size=args.step,
        )
        os.environ["MINIMLX_MTP_VERIFY_SDPA"] = "fused"
        os.environ["MINIMLX_MTP_FORCE_FUSED_VERIFY"] = "0"
    if args.mtp_argmax_ab:
        # Compile each candidate target-head path before measurements; keep
        # positionwise as the correctness reference for the A/B comparison.
        for mode in ("batched", "singleton_batch"):
            os.environ["MINIMLX_MTP_VERIFY_ARGMAX"] = mode
            runtime.generate(
                prompt,
                max_tokens=args.warmup,
                context_tokens=2048,
                prefill_step_size=args.step,
            )
        os.environ["MINIMLX_MTP_VERIFY_ARGMAX"] = "positionwise"

    if args.mtp_forward_ab:
        # Warm the candidate with the ordinary model forward before timing.
        os.environ["MINIMLX_MTP_VERIFY_FORWARD"] = "standard"
        runtime.generate(
            prompt,
            max_tokens=args.warmup,
            context_tokens=2048,
            prefill_step_size=args.step,
        )
        os.environ["MINIMLX_MTP_VERIFY_FORWARD"] = "exact"

    if args.mtp_gdn_ab:
        # Compile the optional fused GDN arm before timing, with the same
        # fused-attention and verifier settings used in both measured variants.
        os.environ["MINIMLX_MTP_GDN_PREWORK"] = "1"
        runtime.generate(prompt, max_tokens=args.warmup, context_tokens=2048,
                         prefill_step_size=args.step)
        os.environ["MINIMLX_MTP_GDN_PREWORK"] = "0"

    for context_index, context in enumerate(args.contexts):
        if args.mtp_gdn_ab:
            from minimlx_runtime.kernels.mtp_gdn_prework import (
                mtp_gdn_prework_routes, reset_mtp_gdn_prework_routes,
            )
            variants = {}
            for mode in ("stock", "fused"):
                os.environ["MINIMLX_MTP_GDN_PREWORK"] = "1" if mode == "fused" else "0"
                reset_mtp_gdn_prework_routes()
                result, e2e_ms = timed_generate(
                    prompt, max_tokens=args.max_tokens, context_tokens=context,
                    prefill_step_size=args.step,
                )
                trace = getattr(adapter, "_mtp_last_round_trace", ())
                rounds = len(trace)
                drafted = sum(int(row.get("drafted", 0)) for row in trace)
                accepted = sum(int(row.get("accepted", 0)) for row in trace)
                variants[mode] = (result, e2e_ms, rounds, drafted, accepted)
                decode_ms = result.decode_ms
                tpot_ms = decode_ms / (args.max_tokens - 1) if args.max_tokens > 1 else 0.0
                print(
                    f"context={context} gdn={mode} TTFT={result.prefill_ms:.1f} ms "
                    f"TPOT={tpot_ms:.2f} ms TG_TPS={args.max_tokens * 1000.0 / decode_ms:.2f} "
                    f"E2E={e2e_ms:.1f} ms mtp_rounds={rounds} drafted={drafted} "
                    f"accepted={accepted} gdn_routes={mtp_gdn_prework_routes()}"
                )
            os.environ["MINIMLX_MTP_GDN_PREWORK"] = "0"
            left = variants["stock"][0].generated_tokens
            right = variants["fused"][0].generated_tokens
            mismatch = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                            None if len(left) == len(right) else min(len(left), len(right)))
            print(f"  fused_matches_stock={mismatch is None} first_mismatch={mismatch} context={context}")
            continue

        if args.mtp_forward_ab:
            variants = {}
            for mode in ("exact", "standard"):
                os.environ["MINIMLX_MTP_VERIFY_FORWARD"] = mode
                result, e2e_ms = timed_generate(
                    prompt,
                    max_tokens=args.max_tokens,
                    context_tokens=context,
                    prefill_step_size=args.step,
                )
                trace = getattr(adapter, "_mtp_last_round_trace", ())
                rounds = len(trace)
                drafted = sum(int(row.get("drafted", 0)) for row in trace)
                accepted = sum(int(row.get("accepted", 0)) for row in trace)
                variants[mode] = (result, e2e_ms, rounds, drafted, accepted)
                decode_ms = result.decode_ms
                tpot_ms = decode_ms / (args.max_tokens - 1) if args.max_tokens > 1 else 0.0
                print(
                    f"context={context} verifier={mode} "
                    f"TTFT={result.prefill_ms:.1f} ms "
                    f"TPOT={tpot_ms:.2f} ms "
                    f"TG_TPS={args.max_tokens * 1000.0 / decode_ms:.2f} "
                    f"E2E={e2e_ms:.1f} ms "
                    f"mtp_rounds={rounds} drafted={drafted} accepted={accepted}"
                )
            os.environ["MINIMLX_MTP_VERIFY_FORWARD"] = "exact"
            exact_tokens = variants["exact"][0].generated_tokens
            standard_tokens = variants["standard"][0].generated_tokens
            mismatch = next(
                (
                    i
                    for i, (left, right) in enumerate(zip(exact_tokens, standard_tokens))
                    if left != right
                ),
                None
                if len(exact_tokens) == len(standard_tokens)
                else min(len(exact_tokens), len(standard_tokens)),
            )
            print(
                f"  standard_matches_exact={mismatch is None} "
                f"first_mismatch={mismatch} context={context}"
            )
            continue

        if args.mtp_argmax_ab:
            from minimlx_runtime.kernels.mtp_verify import (
                mtp_verify_fused_attention_stats,
                reset_mtp_verify_fused_attention_stats,
            )

            def run_argmax_variant(mode: str):
                os.environ["MINIMLX_MTP_VERIFY_ARGMAX"] = mode
                reset_mtp_verify_fused_attention_stats()
                result, e2e_ms = timed_generate(
                    prompt,
                    max_tokens=args.max_tokens,
                    context_tokens=context,
                    prefill_step_size=args.step,
                )
                trace = getattr(adapter, "_mtp_last_round_trace", ())
                rounds = len(trace)
                drafted = sum(int(row.get("drafted", 0)) for row in trace)
                accepted = sum(int(row.get("accepted", 0)) for row in trace)
                routes, fallbacks = mtp_verify_fused_attention_stats()
                return result, e2e_ms, rounds, drafted, accepted, routes, fallbacks

            modes = (
                ("positionwise", "batched", "singleton_batch")
                if context_index % 2 == 0
                else ("singleton_batch", "batched", "positionwise")
            )
            measured = {mode: run_argmax_variant(mode) for mode in modes}
            for mode in ("positionwise", "batched", "singleton_batch"):
                result, e2e_ms, rounds, drafted, accepted, routes, fallbacks = measured[mode]
                decode_ms = result.decode_ms
                tpot_ms = decode_ms / (args.max_tokens - 1) if args.max_tokens > 1 else 0.0
                print(
                    f"context={context} argmax={mode} "
                    f"TTFT={result.prefill_ms:.1f} ms "
                    f"TPOT={tpot_ms:.2f} ms "
                    f"TG_TPS={args.max_tokens * 1000.0 / decode_ms:.2f} "
                    f"E2E={e2e_ms:.1f} ms "
                    f"mtp_rounds={rounds} drafted={drafted} accepted={accepted} "
                    f"fused_routes={routes} fused_fallbacks={fallbacks}"
                )
            reference = measured["positionwise"][0].generated_tokens
            for mode in ("batched", "singleton_batch"):
                candidate = measured[mode][0].generated_tokens
                mismatch = next(
                    (
                        i
                        for i, (left, right) in enumerate(zip(reference, candidate))
                        if left != right
                    ),
                    None
                    if len(reference) == len(candidate)
                    else min(len(reference), len(candidate)),
                )
                print(
                    f"  {mode}_matches_positionwise={mismatch is None} "
                    f"first_mismatch={mismatch} context={context}"
                )
            # Compare both speculative variants to a same-target greedy AR
            # control over the exact same prompt token IDs.
            import mlx.core as mx

            lm = adapter.model.language_model
            lm._rope_deltas = None
            lm._position_ids = None
            cache = lm.make_cache()
            prompt_tokens = measured["positionwise"][0].prompt_tokens
            hidden = None
            for offset in range(0, len(prompt_tokens), args.step):
                current = mx.array(
                    [prompt_tokens[offset : offset + args.step]], dtype=mx.uint32
                )
                output = lm(
                    current,
                    cache=cache,
                    return_hidden=True,
                    skip_logits=True,
                )
                hidden = output.hidden_states[-1]
                mx.eval([entry.state for entry in cache if entry is not None])
            if hidden is None:
                raise RuntimeError("cannot compare MTP with AR for an empty prompt")
            eos_ids = set(getattr(adapter.tokenizer, "eos_token_ids", ()) or ())
            target_ar_tokens = []
            token_array = lm.speculative_argmax_from_hidden(hidden[:, -1:, :])
            if token_array is None:
                token_array = mx.argmax(
                    lm.speculative_logits_from_hidden(hidden[:, -1:, :]), axis=-1
                )
            mx.eval(token_array)
            target_ar_tokens.append(int(token_array.reshape(-1).item()))
            for _ in range(max(0, args.max_tokens - 1)):
                if target_ar_tokens[-1] in eos_ids:
                    break
                current = mx.array([[target_ar_tokens[-1]]], dtype=mx.uint32)
                output = lm(current, cache=cache)
                token_array = mx.argmax(output.logits[:, -1, :], axis=-1)
                mx.eval(token_array)
                target_ar_tokens.append(int(token_array.reshape(-1).item()))
            for mode in ("positionwise", "batched", "singleton_batch"):
                generated = measured[mode][0].generated_tokens
                mismatch = next(
                    (
                        i
                        for i, (actual, expected) in enumerate(
                            zip(generated, target_ar_tokens)
                        )
                        if actual != expected
                    ),
                    None
                    if len(generated) == len(target_ar_tokens)
                    else min(len(generated), len(target_ar_tokens)),
                )
                print(
                    f"  {mode}_matches_target_ar={mismatch is None} "
                    f"first_mismatch={mismatch} "
                    f"mtp_tokens={len(generated)} target_ar_tokens={len(target_ar_tokens)} "
                    f"context={context}"
                )
                if mismatch is not None:
                    start = max(0, mismatch - 4)
                    end = mismatch + 5
                    print(
                        f"    mismatch_window[{start}:{end}] "
                        f"mtp={generated[start:end]} "
                        f"target_ar={target_ar_tokens[start:end]}"
                    )
            continue

        if args.mtp_verify_ab:
            from minimlx_runtime.kernels.mtp_verify import (
                mtp_verify_fused_attention_stats,
                reset_mtp_verify_fused_attention_stats,
            )

            def run_variant(mode: str):
                os.environ["MINIMLX_MTP_FORCE_FUSED_VERIFY"] = (
                    "0" if mode == "stock" else "1"
                )
                os.environ["MINIMLX_MTP_VERIFY_SDPA"] = (
                    "vector" if mode == "omlx_vector" else "fused"
                )
                reset_mtp_verify_fused_attention_stats()
                result, e2e_ms = timed_generate(
                    prompt,
                    max_tokens=args.max_tokens,
                    context_tokens=context,
                    prefill_step_size=args.step,
                )
                trace = getattr(adapter, "_mtp_last_round_trace", ())
                rounds = len(trace)
                drafted = sum(int(row.get("drafted", 0)) for row in trace)
                accepted = sum(int(row.get("accepted", 0)) for row in trace)
                route_calls, fallbacks = mtp_verify_fused_attention_stats()
                return result, e2e_ms, rounds, drafted, accepted, route_calls, fallbacks

            # Rotate run order across contexts to reduce systematic
            # warm-up/thermal ordering bias.
            modes = ("stock", "fused", "omlx_vector")
            shift = context_index % len(modes)
            ordered_modes = modes[shift:] + modes[:shift]
            measured = {mode: run_variant(mode) for mode in ordered_modes}
            stock = measured["stock"]
            for label in modes:
                result, e2e_ms, rounds, drafted, accepted, route_calls, fallbacks = measured[label]
                if len(result.prompt_tokens) != context or len(result.generated_tokens) != args.max_tokens:
                    raise RuntimeError(
                        f"shape mismatch at context={context}: "
                        f"prompt={len(result.prompt_tokens)} generated={len(result.generated_tokens)}"
                )
                decode_ms = result.decode_ms
                tpot_ms = decode_ms / (args.max_tokens - 1) if args.max_tokens > 1 else 0.0
                print(
                    f"context={context} variant={label} "
                    f"TTFT={result.prefill_ms:.1f} ms "
                    f"TPOT={tpot_ms:.2f} ms "
                    f"TG_TPS={args.max_tokens * 1000.0 / decode_ms:.2f} "
                    f"E2E={e2e_ms:.1f} ms "
                    f"mtp_rounds={rounds} drafted={drafted} accepted={accepted} "
                    f"acceptance={100.0 * accepted / drafted if drafted else 0.0:.1f}% "
                    f"verify_routes={route_calls} verify_fallbacks={fallbacks}"
                )
            for mode in ("fused", "omlx_vector"):
                candidate = measured[mode]
                if candidate[5] == 0 or candidate[6] > 0:
                    raise RuntimeError(
                        f"{mode} verifier route was not cleanly exercised at context={context}: "
                        f"routes={candidate[5]} fallbacks={candidate[6]}"
                    )
                if stock[0].generated_tokens != candidate[0].generated_tokens:
                    mismatch = next(
                        (
                            i
                            for i, (left, right) in enumerate(
                                zip(stock[0].generated_tokens, candidate[0].generated_tokens)
                            )
                            if left != right
                        ),
                        min(len(stock[0].generated_tokens), len(candidate[0].generated_tokens)),
                    )
                    raise RuntimeError(
                        f"{mode} MTP verifier changed generated tokens at context={context}, "
                        f"index={mismatch}; candidate rejected."
                    )
            print(f"  mtp_verify_token_match=True context={context}")
            continue

        if mtp_enabled:
            from minimlx_runtime.kernels.mtp_verify import (
                mtp_verify_fused_attention_stats,
                reset_mtp_verify_fused_attention_stats,
            )

            reset_mtp_verify_fused_attention_stats()
        gdn_routes = None
        if mtp_enabled:
            from minimlx_runtime.kernels.mtp_gdn_prework import (
                mtp_gdn_prework_routes,
                reset_mtp_gdn_prework_routes,
            )
            from minimlx_runtime.kernels.gdn import (
                mtp_vlm_gdn_prefill_stats,
                reset_mtp_vlm_gdn_prefill_stats,
            )

            reset_mtp_gdn_prework_routes()
            reset_mtp_vlm_gdn_prefill_stats()
        result, e2e_ms = timed_generate(
            prompt,
            max_tokens=args.max_tokens,
            context_tokens=context,
            prefill_step_size=args.step,
        )
        if mtp_enabled:
            gdn_routes = mtp_gdn_prework_routes()
            gdn_prefill_routes, gdn_prefill_fallbacks = mtp_vlm_gdn_prefill_stats()
        if len(result.prompt_tokens) != context or len(result.generated_tokens) != args.max_tokens:
            raise RuntimeError(
                f"shape mismatch at context={context}: "
                f"prompt={len(result.prompt_tokens)} generated={len(result.generated_tokens)}"
            )
        prefill_ms = result.prefill_ms
        decode_ms = result.decode_ms
        tpot_ms = decode_ms / (args.max_tokens - 1) if args.max_tokens > 1 else 0.0
        print(
            f"context={context} TTFT={prefill_ms:.1f} ms "
            f"TPOT={tpot_ms:.2f} ms "
            f"TG_TPS={args.max_tokens * 1000.0 / decode_ms:.2f} "
            f"E2E={e2e_ms:.1f} ms"
        )
        if mtp_enabled:
            trace = getattr(adapter, "_mtp_last_round_trace", ())
            rounds = len(trace)
            drafted = sum(int(row.get("drafted", 0)) for row in trace)
            accepted = sum(int(row.get("accepted", 0)) for row in trace)
            acceptance_pct = 100.0 * accepted / drafted if drafted else 0.0
            route_calls, fallbacks = mtp_verify_fused_attention_stats()
            print(
                f"  mtp rounds={rounds} drafted={drafted} accepted={accepted} "
                f"acceptance={acceptance_pct:.1f}% "
                f"attention_routes={route_calls} attention_fallbacks={fallbacks} "
                f"gdn_routes={gdn_routes} "
                f"gdn_prefill_routes={gdn_prefill_routes} "
                f"gdn_prefill_fallbacks={gdn_prefill_fallbacks}"
            )


if __name__ == "__main__":
    main()
