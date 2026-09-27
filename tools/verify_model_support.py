#!/usr/bin/env python3
"""Verify a model against MiniMLX's model-family adapter.

This is a compatibility smoke test, not a performance benchmark.  It checks
descriptor resolution, model loading, Qwen3.5 layer structure, and a short
greedy generation.  Quantized-only optimizations are reported separately so
BF16 models can share the family adapter without being treated as Q4 models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--compare-mtp-target-ar",
        action="store_true",
        help="In MTP mode, also greedy-decode with the same MLX-VLM target as a control.",
    )
    parser.add_argument(
        "--mtp-test-force-eos-at",
        type=int,
        metavar="OUTPUT_INDEX",
        help="Test-only: force an EOS target token at this generated-token index (>=1).",
    )
    parser.add_argument(
        "--mtp-test-force-reject-at",
        type=int,
        metavar="OUTPUT_INDEX",
        help="Test-only: force a target token different from the draft at this index (>=1).",
    )
    parser.add_argument(
        "--mtp-test-boundaries",
        action="store_true",
        help="In one model load, run forced-rejection and forced-EOS MTP/AR checks.",
    )
    args = parser.parse_args()

    if args.mtp_test_force_eos_at is not None and args.mtp_test_force_reject_at is not None:
        parser.error("choose only one MTP forced-boundary test per run")
    forced_index = (
        args.mtp_test_force_eos_at
        if args.mtp_test_force_eos_at is not None
        else args.mtp_test_force_reject_at
    )
    if forced_index is not None and forced_index != 1:
        parser.error("the deterministic boundary hook currently supports index 1 only")
    if forced_index is not None and args.max_tokens < 6:
        parser.error("forced boundary tests need --max-tokens >= 6 to observe continuation")
    if args.mtp_test_boundaries and forced_index is not None:
        parser.error("use --mtp-test-boundaries alone; it runs both forced cases")
    if args.mtp_test_boundaries and args.max_tokens < 6:
        parser.error("--mtp-test-boundaries requires --max-tokens >= 6")

    from minimlx_runtime.registry import describe_model
    from minimlx_runtime.runtime import Runtime

    descriptor = describe_model(args.model_dir)
    print("MiniMLX model compatibility verification")
    print(f"model_dir={descriptor.model_dir}")
    print(f"model_type={descriptor.model_type}")
    print(f"architecture={descriptor.architecture}")
    print(f"metadata={json.dumps(descriptor.metadata, sort_keys=True)}")

    runtime = Runtime()
    runtime.load(args.model_dir)
    adapter = runtime.adapter
    forced_target_tokens: dict[int, int] = {}
    restore_boundary_hook = None
    if forced_index is not None:
        if not adapter._mtp_enabled:
            raise RuntimeError("forced MTP boundary tests require MTP mode")
        if not args.compare_mtp_target_ar:
            raise RuntimeError("forced MTP boundary tests require --compare-mtp-target-ar")
        mode = "eos" if args.mtp_test_force_eos_at is not None else "reject"
        restore_boundary_hook = _install_mtp_boundary_test_hook(
            adapter, mode, forced_target_tokens
        )
    model = adapter.model
    core = model.language_model.model
    layers = core.layers
    linear_layers = sum(bool(getattr(layer, "is_linear", False)) for layer in layers)
    full_layers = len(layers) - linear_layers
    moe_layers = [
        layer for layer in layers
        if getattr(getattr(layer, "mlp", None), "switch_mlp", None) is not None
    ]
    q4_dense_mlp_layers = sum(
        hasattr(getattr(layer, "mlp", None), "bits")
        or hasattr(getattr(getattr(layer, "mlp", None), "gate_proj", None), "bits")
        for layer in layers
    )
    print(f"layers={len(layers)} linear_attention={linear_layers} full_attention={full_layers}")
    print(f"decode_backend={'mtp' if adapter._mtp_enabled else 'autoregressive'}")
    print(f"q4_dense_mlp_layers={q4_dense_mlp_layers}")
    print("q4_prefill_dense_active=" + str(q4_dense_mlp_layers > 0))
    if moe_layers:
        moe = moe_layers[0].mlp
        projection_names = ("gate_proj", "up_proj", "down_proj")
        quant_patterns: dict[tuple[int, ...], int] = {}
        quant_bits_by_projection = {name: set() for name in projection_names}
        expert_counts = set()
        q4_expert_layers = 0
        for layer in moe_layers:
            switch = layer.mlp.switch_mlp
            projections = tuple(getattr(switch, name) for name in projection_names)
            expert_counts.update(int(projection.weight.shape[0]) for projection in projections)
            pattern = tuple(
                int(projection.bits)
                for projection in projections
                if hasattr(projection, "bits")
            )
            if len(pattern) == len(projection_names):
                quant_patterns[pattern] = quant_patterns.get(pattern, 0) + 1
            if 4 in pattern:
                q4_expert_layers += 1
            for name, projection in zip(projection_names, projections):
                if hasattr(projection, "bits"):
                    quant_bits_by_projection[name].add(int(projection.bits))
        print(
            "moe=" + json.dumps({
                "layers": len(moe_layers),
                "experts": moe.num_experts,
                "top_k": moe.top_k,
                "expert_tensor_counts": sorted(expert_counts),
                "expert_quant_bits_by_projection": {
                    name: sorted(bits)
                    for name, bits in quant_bits_by_projection.items()
                },
                "expert_quant_pattern_layers": {
                    "/".join(map(str, pattern)): count
                    for pattern, count in sorted(quant_patterns.items())
                },
                "shared_expert": getattr(moe, "shared_expert", None) is not None,
                "shared_expert_gate": getattr(moe, "shared_expert_gate", None) is not None,
            }, sort_keys=True)
        )
        if expert_counts != {int(moe.num_experts)}:
            raise RuntimeError(f"inconsistent MoE expert tensor shapes: {expert_counts}")
        print(f"q4_moe_expert_layers={q4_expert_layers}")
    print("family_adapter=Qwen35Adapter")

    if args.mtp_test_boundaries:
        if not adapter._mtp_enabled:
            raise RuntimeError("--mtp-test-boundaries requires MTP mode")
        _run_mtp_boundary_tests(runtime, adapter, args.prompt, args.max_tokens)
        return

    try:
        result = runtime.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            prefill_step_size=1024,
        )
    finally:
        if restore_boundary_hook is not None:
            restore_boundary_hook()
    print(f"prompt_tokens={result.prompt_tokens}")
    print(f"generated_tokens={result.generated_tokens}")
    print(f"prefill_ms={result.prefill_ms:.3f}")
    print(f"decode_ms={result.decode_ms:.3f}")
    if adapter._mtp_enabled:
        drafter = adapter._mtp_draft_model
        accepted = list(getattr(drafter, "accept_lens", ()) or ())
        drafted = list(getattr(drafter, "draft_lens", ()) or ())
        if accepted and len(accepted) == len(drafted):
            accepted_total = sum(float(value) for value in accepted)
            drafted_total = sum(int(value) for value in drafted)
            acceptance = (
                100.0 * accepted_total / drafted_total if drafted_total else 0.0
            )
            print(
                "mtp_stats="
                f"rounds:{len(accepted)},"
                f"drafted:{drafted_total},"
                f"accepted:{accepted_total:g},"
                f"acceptance_pct:{acceptance:.1f}"
            )
        else:
            print("mtp_stats=unavailable (no speculative rounds)")
    mtp_target_ar_match = None
    if args.compare_mtp_target_ar:
        if not adapter._mtp_enabled:
            raise RuntimeError("--compare-mtp-target-ar requires MTP mode")
        import mlx.core as mx

        lm = adapter.model.language_model
        # Clear positional memoization left by the MTP request; this is a fresh
        # single-request, single-row autoregressive control on the same target.
        lm._rope_deltas = None
        lm._position_ids = None
        cache = lm.make_cache()
        current = mx.array([result.prompt_tokens], dtype=mx.uint32)
        eos_ids = set(getattr(adapter.tokenizer, "eos_token_ids", ()) or ())
        target_ar_tokens = []
        import time

        target_prefill_start = time.perf_counter()
        output = lm(current, cache=cache)
        token_array = mx.argmax(output.logits[:, -1, :], axis=-1)
        mx.eval(token_array)
        target_ar_tokens.append(int(token_array.reshape(-1).item()))
        target_ar_prefill_ms = (time.perf_counter() - target_prefill_start) * 1000.0

        target_decode_start = time.perf_counter()
        for _ in range(max(0, args.max_tokens - 1)):
            if target_ar_tokens[-1] in eos_ids:
                break
            current = mx.array([[target_ar_tokens[-1]]], dtype=mx.uint32)
            output = lm(current, cache=cache)
            token_array = mx.argmax(output.logits[:, -1, :], axis=-1)
            mx.eval(token_array)
            output_index = len(target_ar_tokens)
            target_token = int(token_array.reshape(-1).item())
            target_token = forced_target_tokens.get(output_index, target_token)
            target_ar_tokens.append(target_token)
        target_decode_ms = (time.perf_counter() - target_decode_start) * 1000.0
        print(f"mtp_target_ar_tokens={target_ar_tokens}")
        print(f"mtp_target_ar_prefill_ms={target_ar_prefill_ms:.3f}")
        print(f"mtp_target_ar_decode_ms={target_decode_ms:.3f}")
        if target_decode_ms > 0:
            target_tps = max(0, len(target_ar_tokens) - 1) * 1000.0 / target_decode_ms
            print(f"mtp_target_ar_tps={target_tps:.2f}")
        matches = target_ar_tokens == result.generated_tokens
        mtp_target_ar_match = matches
        print(f"mtp_matches_same_target_ar={matches}")
        if not matches:
            mismatch = next(
                (
                    index
                    for index, (mtp_token, ar_token) in enumerate(
                        zip(result.generated_tokens, target_ar_tokens)
                    )
                    if mtp_token != ar_token
                ),
                min(len(result.generated_tokens), len(target_ar_tokens)),
            )
            start = max(0, mismatch - 3)
            end = mismatch + 4
            print(
                f"mtp_first_mismatch=index:{mismatch},"
                f"mtp:{result.generated_tokens[mismatch:mismatch + 1]},"
                f"target_ar:{target_ar_tokens[mismatch:mismatch + 1]},"
                f"mtp_context:{result.generated_tokens[start:end]},"
                f"target_ar_context:{target_ar_tokens[start:end]}"
            )
            cursor = 1  # index zero is the first target bonus before MTP rounds
            trace = adapter._mtp_last_round_trace
            for round_index, round_info in enumerate(trace):
                round_end = cursor + len(round_info["tokens"])
                if cursor <= mismatch < round_end:
                    local_index = mismatch - cursor
                    accepted = round_info["accepted"]
                    role = "accepted_draft" if local_index < accepted else "target_bonus"
                    print(
                        f"mtp_mismatch_round={round_index},"
                        f"output_span:[{cursor},{round_end}),"
                        f"role:{role},"
                        f"accepted:{accepted},drafted:{round_info['drafted']},"
                        f"round_tokens:{round_info['tokens']}"
                    )
                    break
                cursor = round_end
    print(f"text={result.text!r}")
    import mlx.core as mx

    print(f"peak_memory_gib={mx.get_peak_memory() / (1024**3):.3f}")
    if args.mtp_test_force_eos_at is not None:
        eos_ids = set(getattr(adapter.tokenizer, "eos_token_ids", ()) or ())
        if (
            len(result.generated_tokens) <= args.mtp_test_force_eos_at
            or result.generated_tokens[-1] not in eos_ids
        ):
            raise RuntimeError("forced-EOS boundary test failed to stop at EOS")
        print(f"mtp_forced_eos_boundary=PASS index={args.mtp_test_force_eos_at}")
    if args.mtp_test_force_reject_at is not None:
        trace = adapter._mtp_last_round_trace
        if not trace or trace[0]["accepted"] != 0:
            raise RuntimeError("forced-rejection boundary test did not reject draft token 0")
        print(f"mtp_forced_rejection_boundary=PASS index={args.mtp_test_force_reject_at}")
    if mtp_target_ar_match is False:
        print("compatibility=FAIL (MTP differs from same-target greedy decode)")
        raise SystemExit(1)
    print("compatibility=PASS")


def _install_mtp_boundary_test_hook(adapter, mode, forced_target_tokens):
    """Install a test-only override around verifier argmax, not drafter argmax."""
    import mlx.core as mx

    lm = adapter.model.language_model
    original_argmax = lm.speculative_argmax_from_hidden
    original_verify_hidden = lm.speculative_verify_hidden
    draft_model = adapter._mtp_draft_model
    original_draft_block = draft_model.draft_block
    latest_draft_tokens: list[int] = []
    verifier_argmax_pending = False

    def mark_verifier_forward(*call_args, **call_kwargs):
        nonlocal verifier_argmax_pending
        result = original_verify_hidden(*call_args, **call_kwargs)
        verifier_argmax_pending = True
        return result

    def record_draft(*call_args, **call_kwargs):
        nonlocal latest_draft_tokens
        result = original_draft_block(*call_args, **call_kwargs)
        latest_draft_tokens = [int(token) for token in result.reshape(-1).tolist()]
        return result

    def controlled_argmax(hidden):
        nonlocal verifier_argmax_pending
        result = original_argmax(hidden)
        if result is None or not verifier_argmax_pending:
            return result
        verifier_argmax_pending = False
        values = [int(token) for token in result.reshape(-1).tolist()]
        for offset in range(len(values)):
            index = 1 + offset
            if index != 1 or index in forced_target_tokens:
                continue
            if mode == "eos":
                eos_ids = sorted(set(getattr(adapter.tokenizer, "eos_token_ids", ()) or ()))
                if not eos_ids:
                    raise RuntimeError("tokenizer exposes no EOS ids")
                token = eos_ids[0]
            elif mode == "reject":
                draft_index = offset
                if draft_index >= len(latest_draft_tokens):
                    raise RuntimeError("no corresponding draft token for forced rejection")
                vocab_size = int(lm.args.vocab_size)
                token = latest_draft_tokens[draft_index]
                eos_ids = set(getattr(adapter.tokenizer, "eos_token_ids", ()) or ())
                while token == latest_draft_tokens[draft_index] or token in eos_ids:
                    token = (token + 1) % vocab_size
            else:
                raise ValueError(f"unknown MTP boundary test mode: {mode}")
            values[offset] = token
            forced_target_tokens[index] = token
        return mx.array(values, dtype=result.dtype).reshape(result.shape)

    draft_model.draft_block = record_draft
    lm.speculative_argmax_from_hidden = controlled_argmax
    lm.speculative_verify_hidden = mark_verifier_forward

    def restore():
        draft_model.draft_block = original_draft_block
        lm.speculative_argmax_from_hidden = original_argmax
        lm.speculative_verify_hidden = original_verify_hidden

    return restore


def _run_mtp_boundary_tests(runtime, adapter, prompt, max_tokens) -> None:
    import mlx.core as mx

    lm = adapter.model.language_model
    eos_ids = set(getattr(adapter.tokenizer, "eos_token_ids", ()) or ())
    if not eos_ids:
        raise RuntimeError("tokenizer exposes no EOS ids")

    for mode in ("reject", "eos"):
        forced_tokens: dict[int, int] = {}
        restore = _install_mtp_boundary_test_hook(adapter, mode, forced_tokens)
        try:
            result = runtime.generate(prompt, max_tokens=max_tokens, prefill_step_size=1024)
        finally:
            restore()

        if 1 not in forced_tokens:
            raise RuntimeError(f"{mode} boundary hook did not reach target verifier index 1")
        if mode == "eos":
            if len(result.generated_tokens) != 2 or result.generated_tokens[-1] not in eos_ids:
                raise RuntimeError("forced EOS did not terminate immediately at output index 1")
        elif not adapter._mtp_last_round_trace or adapter._mtp_last_round_trace[0]["accepted"] != 0:
            raise RuntimeError("forced rejection did not reject the first draft token")

        lm._rope_deltas = None
        lm._position_ids = None
        cache = lm.make_cache()
        current = mx.array([result.prompt_tokens], dtype=mx.uint32)
        target_ar_tokens = []
        output = lm(current, cache=cache)
        token_array = mx.argmax(output.logits[:, -1, :], axis=-1)
        mx.eval(token_array)
        target_ar_tokens.append(int(token_array.reshape(-1).item()))
        for _ in range(max_tokens - 1):
            if target_ar_tokens[-1] in eos_ids:
                break
            current = mx.array([[target_ar_tokens[-1]]], dtype=mx.uint32)
            output = lm(current, cache=cache)
            token_array = mx.argmax(output.logits[:, -1, :], axis=-1)
            mx.eval(token_array)
            index = len(target_ar_tokens)
            target_ar_tokens.append(forced_tokens.get(index, int(token_array.reshape(-1).item())))

        if target_ar_tokens != result.generated_tokens:
            raise RuntimeError(
                f"forced {mode} MTP output differs from controlled same-target AR: "
                f"mtp={result.generated_tokens}, ar={target_ar_tokens}"
            )
        print(f"mtp_forced_{mode}_boundary=PASS index=1 rounds={len(adapter._mtp_last_round_trace)}")
        print(f"mtp_forced_{mode}_matches_same_target_ar=True")

    print("mtp_boundary_suite=PASS")


if __name__ == "__main__":
    main()
