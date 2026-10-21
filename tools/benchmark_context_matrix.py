#!/usr/bin/env python3
"""Run the final MiniMLX context-length benchmark in one MLX process."""

from __future__ import annotations

import argparse
import json
import os
import statistics
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


def make_omlx_aligned_prompt(tokenizer, max_context: int) -> str:
    """Build the deterministic Python continuation prompt used by oMLX's paired benchmark."""
    seed = (
        "def process_batch(items):\n"
        "    results = []\n"
        "    for item in items:\n"
        "        value = transform(item)\n"
        "        results.append(value)\n"
        "    return results\n\n"
    )
    source = seed * max(1000, max_context // 8)
    token_ids = list(tokenizer.encode(source, add_special_tokens=False))
    if len(token_ids) < max_context:
        raise RuntimeError(
            f"aligned prompt seed is too short: {len(token_ids)} < {max_context}"
        )
    prompt = tokenizer.decode(token_ids[:max_context])
    aligned_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if len(aligned_ids) != max_context:
        raise RuntimeError(
            "oMLX aligned prompt failed round-trip: "
            f"expected {max_context}, got {len(aligned_ids)}"
        )
    return prompt


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
        "--ignore-eos",
        action="store_true",
        help="force the decode to max_tokens; use for apples-to-apples throughput comparison",
    )
    parser.add_argument(
        "--omlx-aligned-prompt",
        action="store_true",
        help="use oMLX/PyMiniMLX's deterministic token-aligned Python prompt",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="force Qwen3.x text requests through mlx-lm instead of the MLLM/VLM lane",
    )
    parser.add_argument(
        "--mtp-draft-model",
        type=str,
        help=(
            "paired Qwen3.5 MTP draft checkpoint; enables the isolated "
            "MLX-VLM MTP backend"
        ),
    )
    parser.add_argument(
        "--dflash-draft-model",
        type=str,
        help="paired DFlash draft checkpoint; enables block-diffusion speculative decoding",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        choices=(4, 8),
        help="opt-in KV cache quantization for long-context decode",
    )
    parser.add_argument(
        "--kv-start",
        type=int,
        default=512,
        help="token offset at which opt-in KV cache quantization begins",
    )
    parser.add_argument("--repeats", type=int, default=1, help="measured runs per context; report the median")
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
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.ignore_eos:
        os.environ["MINIMLX_IGNORE_EOS"] = "1"
    if args.omlx_aligned_prompt:
        os.environ["MINIMLX_RAW_COMPLETION"] = "1"
    if args.text_only:
        os.environ["MINIMLX_FORCE_TEXT_RUNTIME"] = "1"
    if args.mtp_draft_model:
        os.environ["MINIMLX_MTP_DRAFT_MODEL"] = args.mtp_draft_model
        # MTP uses the coherent MLX-VLM target/cache/verifier stack. Do not let
        # the ordinary text-only override select the mlx-lm backend first.
        os.environ.pop("MINIMLX_FORCE_TEXT_RUNTIME", None)
    if args.dflash_draft_model:
        if args.mtp_draft_model:
            parser.error("choose either --mtp-draft-model or --dflash-draft-model")
        os.environ["MINIMLX_DFLASH_DRAFT_MODEL"] = args.dflash_draft_model
        os.environ["MINIMLX_FORCE_TEXT_RUNTIME"] = "1"
    if args.kv_bits is not None:
        os.environ["MINIMLX_KV_CACHE_BITS"] = str(args.kv_bits)
        os.environ["MINIMLX_QUANTIZED_KV_START"] = str(args.kv_start)
    os.environ.setdefault("MINIMLX_PREFILL_SKIP_LOGITS", "1")

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

    runtime = Runtime()
    runtime.load(args.model_dir)
    if args.omlx_aligned_prompt:
        prompt = make_omlx_aligned_prompt(
            adapter_tokenizer := runtime.adapter.tokenizer,
            max(args.contexts),
        )
        print(
            f"prompt protocol (oMLX-aligned Python raw completion, "
            f"max_context_tokens={max(args.contexts)})"
        )
    else:
        prompt = Path(args.prompt_file).read_text()
        print("prompt protocol (corpus file)")

    def timed_generate(*args, **kwargs):
        start = time.perf_counter()
        result = runtime.generate(*args, **kwargs)
        return result, (time.perf_counter() - start) * 1000.0

    print(
        "MiniMLX context matrix "
        f"(warmup={args.warmup} tokens at context=2048, measured_runs={args.repeats}, step={args.step})"
    )
    print(
        "runtime kernels "
        f"(q4_native={config.get('enable_q4_native_prefill', False)}, "
        f"fa256_native={config.get('enable_native_fa256_prefill', False)}, "
        f"q4_dense_disabled={config.get('disable_q4_prefill_dense', False) or os.environ.get('MINIMLX_ENABLE_Q4_NATIVE_PREFILL') == '1'}, "
        f"q4_mlp_fusion_disabled={config.get('disable_q4_mlp_fusion', False)})"
    )
    adapter = runtime.adapter
    mtp_enabled = bool(getattr(adapter, "_mtp_enabled", False))
    dflash_enabled = bool(getattr(adapter, "_dflash_enabled", False))
    mtp_draft_path = os.environ.get("MINIMLX_MTP_DRAFT_MODEL")
    if mtp_draft_path and not mtp_enabled:
        raise RuntimeError(
            "MINIMLX_MTP_DRAFT_MODEL was set, but the loaded adapter did not "
            "enable MTP; refusing to report an autoregressive run as an MTP benchmark."
        )
    dflash_draft_path = os.environ.get("MINIMLX_DFLASH_DRAFT_MODEL")
    if dflash_draft_path and not dflash_enabled:
        raise RuntimeError(
            "MINIMLX_DFLASH_DRAFT_MODEL was set, but the loaded adapter did not "
            "enable DFlash; refusing to report an autoregressive run as a DFlash benchmark."
        )
    print(
        "decode backend "
        f"({'DFlash' if dflash_enabled else 'Qwen3.5 MTP' if mtp_enabled else 'autoregressive'}"
        + (
            f", draft={Path(dflash_draft_path).name}, "
            f"block_size={min(getattr(getattr(adapter._dflash_draft_model, 'config', None), 'block_size', 5), int(os.environ.get('MINIMLX_DFLASH_BLOCK_SIZE', '5')))}"
            if dflash_enabled and dflash_draft_path
            else f", draft={Path(mtp_draft_path).name}, "
            f"block_size={getattr(getattr(adapter._mtp_draft_model, 'config', None), 'block_size', 'default')}"
            if mtp_enabled and mtp_draft_path
            else ""
        )
        + ")"
    )
    print(
        "runtime route "
        f"({'text-only mlx-lm' if not getattr(adapter, '_vlm_enabled', False) else 'mlx-vlm/MLLM'})"
    )
    print(
        "prefill/decode options ("
        f"skip_intermediate_logits_requested={os.environ.get('MINIMLX_PREFILL_SKIP_LOGITS') == '1'}, "
        f"kv_bits={os.environ.get('MINIMLX_KV_CACHE_BITS', 'off')}, "
        f"kv_start={os.environ.get('MINIMLX_QUANTIZED_KV_START', 'n/a')})"
    )
    raw_eos_ids = getattr(adapter.tokenizer, "eos_token_ids", ())
    if raw_eos_ids is None:
        raw_eos_ids = ()
    elif isinstance(raw_eos_ids, int):
        raw_eos_ids = (raw_eos_ids,)
    eos_ids = sorted(int(token) for token in raw_eos_ids)
    print(f"generation stop policy (ignore_eos={args.ignore_eos}, eos_ids={eos_ids})")
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

    try:
        import mlx.core as mx
    except ImportError:
        mx = None
    context_peak_memory = []

    for context_index, context in enumerate(args.contexts):
        if mx is not None and hasattr(mx, "reset_peak_memory"):
            # Match oMLX's per-request memory reporting.  Keep model weights
            # resident as the baseline, but measure each context independently.
            mx.reset_peak_memory()
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
            raw_eos_ids = getattr(adapter.tokenizer, "eos_token_ids", ())
            if raw_eos_ids is None:
                raw_eos_ids = ()
            elif isinstance(raw_eos_ids, int):
                raw_eos_ids = (raw_eos_ids,)
            eos_ids = {int(token) for token in raw_eos_ids}
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

        measurements = []
        for repeat in range(args.repeats):
            if mtp_enabled:
                reset_mtp_verify_fused_attention_stats()
                reset_mtp_gdn_prework_routes()
                reset_mtp_vlm_gdn_prefill_stats()
            result, e2e_ms = timed_generate(
                prompt,
                max_tokens=args.max_tokens,
                context_tokens=context,
                prefill_step_size=args.step,
            )
            generated_tokens = len(result.generated_tokens)
            if len(result.prompt_tokens) != context or not 0 < generated_tokens <= args.max_tokens:
                raise RuntimeError(
                    f"shape mismatch at context={context}, repeat={repeat + 1}: "
                    f"prompt={len(result.prompt_tokens)} "
                    f"generated={generated_tokens} max_tokens={args.max_tokens}"
                )
            measurements.append({
                "ttft": result.prefill_ms,
                "decode": result.decode_ms,
                "tpot": result.decode_ms / (generated_tokens - 1) if generated_tokens > 1 else 0.0,
                "pp_tps": context * 1000.0 / result.prefill_ms if result.prefill_ms > 0 else 0.0,
                "tg_tps": generated_tokens * 1000.0 / result.decode_ms if result.decode_ms > 0 else 0.0,
                "generated": generated_tokens,
                "e2e": e2e_ms,
                "first_token": int(result.generated_tokens[0]),
                "first_token_text": adapter.tokenizer.decode(result.generated_tokens[:1]),
                "first_token_is_eos": int(result.generated_tokens[0]) in set(eos_ids),
            })
        if mtp_enabled:
            gdn_routes = mtp_gdn_prework_routes()
            gdn_prefill_routes, gdn_prefill_fallbacks = mtp_vlm_gdn_prefill_stats()
        median = {key: statistics.median(sample[key] for sample in measurements) for key in measurements[0]}
        print(
            f"context={context} TTFT={median['ttft']:.1f} ms "
            f"TPOT={median['tpot']:.2f} ms "
            f"PP_TPS={median['pp_tps']:.1f} "
            f"TG_TPS={median['tg_tps']:.2f} "
            f"generated={median['generated']}/{args.max_tokens} "
            f"first_token={median['first_token']} "
            f"first_token_is_eos={bool(median['first_token_is_eos'])} "
            f"E2E={median['e2e']:.1f} ms statistic=median"
        )
        print("  samples=" + "; ".join(
            f"{sample['ttft']:.1f}/{sample['tpot']:.2f}/{sample['pp_tps']:.1f}/{sample['tg_tps']:.2f}/{sample['e2e']:.1f}"
            for sample in measurements
        ) + " (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms)")
        if mx is not None:
            active = getattr(mx, "get_active_memory", None)
            peak = getattr(mx, "get_peak_memory", None)
            if callable(active) and callable(peak):
                peak_bytes = peak()
                context_peak_memory.append(peak_bytes)
                print(
                    f"  memory active={active() / (1024**3):.3f} GiB "
                    f"peak={peak_bytes / (1024**3):.3f} GiB"
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

    try:
        import mlx.core as mx

        if context_peak_memory:
            suite_peak = max(context_peak_memory)
        else:
            suite_peak = mx.get_peak_memory()
        print(f"suite_peak_memory_gib={suite_peak / (1024**3):.3f}")
    except (ImportError, AttributeError):
        print("suite_peak_memory_gib=unavailable")


if __name__ == "__main__":
    main()
