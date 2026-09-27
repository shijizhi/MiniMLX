from __future__ import annotations

from pathlib import Path
import os
import time
from typing import Any

from ..types import ExecutionPlan, ModelDescriptor
from ..scheduling import choose_prefill_step_size
from ..prompt_cache import PrefixCache


class Qwen35Adapter:
    """First model-family adapter, backed by MiniMLX's own MLX-LM venv.

    PyMiniMLX is a reference project only and is deliberately not imported.
    This adapter is the seam where our own Qwen3.5 executor will replace the
    MLX-LM baseline in a later phase.
    """

    name = "qwen35-mlx-lm"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.descriptor: ModelDescriptor | None = None
        self._ane_projection: Any | None = None
        self._ane_gdn_stage: Any | None = None
        self._prefix_cache: PrefixCache | None = None
        self._session_prefix_caches: dict[str, PrefixCache] = {}
        self._greedy_skip_logsumexp = False
        self._prefill_keep_allocator_cache = False
        self._mtp_draft_model: Any | None = None
        self._mtp_enabled = False
        self._mtp_last_round_trace: list[dict[str, Any]] = []

    def can_load(self, descriptor: ModelDescriptor) -> bool:
        value = f"{descriptor.model_type} {descriptor.architecture}".lower()
        return "qwen3" in value and "5" in value

    def load(self, descriptor: ModelDescriptor) -> "Qwen35Adapter":
        mtp_draft_path = os.environ.get("MINIMLX_MTP_DRAFT_MODEL")
        if mtp_draft_path:
            return self._load_mtp_backend(descriptor, mtp_draft_path)

        try:
            from mlx_lm import load
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "mlx-lm is not installed in MiniMLX's .venv; "
                "run tools/bootstrap_venv.sh"
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "MLX could not access the Metal device. Run MiniMLX from an "
                "interactive macOS/RustRover process with Apple GPU access."
            ) from exc
        self.model, self.tokenizer = load(descriptor.model_dir)
        self.descriptor = descriptor
        self._install_gdn_prefill_kernel()
        self._install_q4_mlp_prefill_dense()
        self._install_q4_native_prefill()
        self._install_q4_native_mlp()
        self._install_q4_mlp_prefill_fusion()
        self._install_native_fa256_prefill()
        if os.environ.get("MINIMLX_FULL_ATTENTION_PREFILL_FORCE_FUSED") == "1":
            os.environ["MINIMLX_FA256_FORCE_FUSED"] = "1"
        self._install_fa256_force_fused()
        self._install_full_attention_kv_cache_growth()
        if os.environ.get("MINIMLX_FULL_ATTENTION_DECODE_FORCE_FUSED") == "1":
            from ..kernels.fa_decode import install_full_attention_decode_force_fused

            install_full_attention_decode_force_fused()
        self._install_ane_projection()
        self._install_ane_gdn_stage()
        self._greedy_skip_logsumexp = os.environ.get(
            "MINIMLX_GREEDY_SKIP_LOGSUMEXP"
        ) == "1"
        self._prefill_keep_allocator_cache = os.environ.get(
            "MINIMLX_PREFILL_KEEP_ALLOCATOR_CACHE"
        ) == "1"
        if os.environ.get("MINIMLX_ENABLE_PREFIX_CACHE") == "1":
            max_bytes = int(os.environ.get("MINIMLX_PREFIX_CACHE_MAX_BYTES", str(4 * 1024**3)))
            self._prefix_cache = PrefixCache(self.model, max_bytes)
        return self

    def _load_mtp_backend(
        self, descriptor: ModelDescriptor, draft_path: str
    ) -> "Qwen35Adapter":
        """Load the coherent MLX-VLM Qwen3.5 target/cache/verifier stack."""
        import importlib.metadata

        try:
            version = importlib.metadata.version("mlx-vlm")
            if version != "0.7.2":
                raise RuntimeError(
                    "The opt-in Qwen3.5 MTP backend is source-pinned to "
                    f"mlx-vlm==0.7.2; found {version}."
                )
            from mlx_lm.utils import load_tokenizer
            from mlx_vlm.speculative.drafters import (
                load_drafter,
                validate_drafter_compatibility,
            )
            from mlx_vlm.utils import load_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MTP mode requires mlx-vlm==0.7.2 in MiniMLX's .venv."
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "MLX-VLM MTP requires an interactive macOS process with "
                "Apple GPU/Metal access."
            ) from exc

        drafter, draft_kind = load_drafter(draft_path)
        if draft_kind != "mtp":
            raise RuntimeError(
                f"MINIMLX_MTP_DRAFT_MODEL resolved to draft kind {draft_kind!r}, "
                "not 'mtp'."
            )

        target = load_model(Path(descriptor.model_dir))
        validate_drafter_compatibility(target, drafter, draft_kind)
        target_lm = getattr(target, "language_model", None)
        required = (
            "make_cache",
            "speculative_verify_hidden",
            "speculative_logits_from_hidden",
            "speculative_argmax_from_hidden",
        )
        missing = [
            name
            for name in required
            if not callable(getattr(target_lm, name, None))
        ]
        if missing:
            raise RuntimeError(
                "Loaded target lacks the Qwen3.5 MTP verifier hooks: "
                + ", ".join(missing)
            )

        # Qwen3.5's quantized speculative argmax is shape-dependent on this
        # MLX-VLM build. Batched argmax has been checked against the target's
        # chunked-prefill greedy decode at 1K, 4K, 8K, and 16K contexts, and is
        # materially faster than evaluating each position independently.
        original_argmax = target_lm.speculative_argmax_from_hidden

        def positionwise_speculative_argmax(hidden):
            import mlx.core as mx

            mode = os.environ.get("MINIMLX_MTP_VERIFY_ARGMAX", "batched")
            if mode not in ("positionwise", "batched", "singleton_batch"):
                raise RuntimeError(
                    "MINIMLX_MTP_VERIFY_ARGMAX must be 'positionwise', "
                    "'batched', or 'singleton_batch'"
                )
            if mode == "batched":
                return original_argmax(hidden)
            if mode == "singleton_batch" and hidden.ndim >= 3 and hidden.shape[1] > 1:
                if target_lm.args.tie_word_embeddings:
                    from mlx_vlm.models.exact_speculative_verify import (
                        exact_speculative_verify_weight,
                    )

                    embedding = target_lm.model.embed_tokens
                    logits = exact_speculative_verify_weight(
                        embedding.weight, hidden
                    )
                    result = mx.argmax(logits, axis=-1) if logits is not None else None
                else:
                    from mlx_vlm.models.quantized_verifier import (
                        singleton_quantized_argmax,
                    )

                    result = singleton_quantized_argmax(target_lm.lm_head, hidden)
                if result is not None:
                    return result
            if hidden.ndim < 3 or hidden.shape[1] <= 1:
                return original_argmax(hidden)
            return mx.concatenate(
                [
                    original_argmax(hidden[:, position : position + 1, :])
                    for position in range(hidden.shape[1])
                ],
                axis=1,
            )

        target_lm.speculative_argmax_from_hidden = positionwise_speculative_argmax

        original_verify_hidden = target_lm.speculative_verify_hidden

        def selectable_verify_hidden(inputs, cache):
            if os.environ.get("MINIMLX_MTP_VERIFY_FORWARD", "exact") != "standard":
                return original_verify_hidden(inputs, cache)

            # Diagnostic candidate: use the ordinary batched decoder forward,
            # while retaining MLX-VLM's speculative cache transaction so
            # rejected draft positions can still be rolled back safely.
            from mlx_vlm.speculative.cache_state import start_speculative_cache

            transaction = start_speculative_cache(cache, inputs.shape[1])
            try:
                output = target_lm(
                    inputs,
                    cache=cache,
                    return_hidden=True,
                    skip_logits=True,
                )
                return output.hidden_states[-1], {}, transaction
            except BaseException:
                transaction.abort()
                raise

        target_lm.speculative_verify_hidden = selectable_verify_hidden

        self.model = target
        self.tokenizer = load_tokenizer(descriptor.model_dir)
        self.descriptor = descriptor
        self._mtp_draft_model = drafter
        self._mtp_enabled = True
        # The ordinary mlx-lm load path installs these on its model, but MTP
        # uses mlx-vlm's Qwen3.5 VLM target. Install the same opt-in native Q4
        # projection and direct gate/up/down routes on that target; the
        # installers retain their normal env gate and sequence threshold.
        from ..kernels.q4_native import (
            install_q4_native_mlp,
            install_q4_native_prefill,
        )

        install_q4_native_prefill(self.model, min_tokens=1024)
        install_q4_native_mlp(self.model, min_tokens=1024)
        from ..kernels.mtp_verify import install_mtp_verify_fused_attention
        from ..kernels.mtp_gdn_prework import install_mtp_gdn_prework
        from ..kernels.mtp_cache_commit import install_mtp_full_accept_cache_fastpath
        from ..kernels.gdn import install_mtp_vlm_gdn_prefill
        from ..kernels.fa256 import install_native_fa256_prefill

        install_mtp_vlm_gdn_prefill()
        install_native_fa256_prefill()
        install_mtp_verify_fused_attention()
        install_mtp_gdn_prework()
        install_mtp_full_accept_cache_fastpath()
        return self

    def _generate_mtp_greedy(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        prefill_step_size: int,
    ):
        """Greedy MTP using MLX-VLM's matching Qwen3.5 target transactions."""
        import mlx.core as mx

        from mlx_lm.generate import generation_stream
        from mlx_vlm.speculative.mtp import (
            _buffer_mtp_target_cache,
            _mtp_rounds,
        )

        if max_tokens <= 0:
            return prompt_tokens, [], "", 0.0, 0.0
        if not prompt_tokens:
            raise ValueError("MTP greedy path requires a non-empty prompt")

        lm = self.model.language_model
        cache = lm.make_cache()
        prompt = mx.array(prompt_tokens, dtype=mx.uint32)
        from ..kernels.mtp_prompt_prefill import MTPPromptPrimer

        drafter = self._mtp_draft_model
        drafter.reset(self.model)
        primer = MTPPromptPrimer(drafter, mx.uint32)
        hidden = None
        processed = 0
        prefill_start = time.perf_counter()
        with mx.stream(generation_stream):
            while processed < len(prompt_tokens):
                count = min(prefill_step_size, len(prompt_tokens) - processed)
                output = lm(
                    prompt[processed : processed + count][None],
                    cache=cache,
                    return_hidden=True,
                    skip_logits=True,
                )
                hidden = output.hidden_states[-1]
                primer.add_chunk(prompt[processed : processed + count][None], hidden)
                mx.eval([entry.state for entry in cache if entry is not None])
                processed += count

            if hidden is None:
                raise ValueError("MTP prefill produced no target hidden states")
            first = lm.speculative_argmax_from_hidden(hidden[:, -1:, :])
            if first is None:
                first = mx.argmax(
                    lm.speculative_logits_from_hidden(hidden[:, -1:, :]), axis=-1
                )
            mx.eval(first)
            first_token = int(first.reshape(-1).item())
            primer.finish(first_token, lambda logits: mx.argmax(logits, axis=-1), True)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

        _buffer_mtp_target_cache(cache, self._mtp_draft_model, None)
        # Qwen3.5 MTP's set_shared_kv() intentionally ignores this argument;
        # MLX-VLM 0.7.2's generic cache extractor assumes layer_type exists,
        # which Qwen3_5DecoderLayer does not provide.
        shared_kv: dict = {}
        generated = [first_token]
        round_trace = []
        diagnostic_round = int(os.environ.get("MINIMLX_MTP_DIAG_ROUND", "-1"))
        diagnostic_token = int(os.environ.get("MINIMLX_MTP_DIAG_TOKEN", "-1"))
        original_verify_hidden = getattr(lm, "speculative_verify_hidden", None)
        if (
            diagnostic_round >= 0 or diagnostic_token >= 0
        ) and callable(original_verify_hidden):
            def verify_hidden_with_diagnostic(verify_input, verify_cache):
                result = original_verify_hidden(verify_input, verify_cache)
                accept_lens = getattr(self._mtp_draft_model, "accept_lens", ())
                round_matches = (
                    diagnostic_round >= 0
                    and len(accept_lens) == diagnostic_round
                )
                verify_hidden = result[0] if isinstance(result, tuple) else result
                token_relative = diagnostic_token - len(generated)
                token_matches = (
                    diagnostic_token >= 0
                    and 0 <= token_relative < verify_hidden.shape[1]
                )
                if round_matches or token_matches:
                    relative = token_relative if token_matches else 0
                    verifier_all_tokens = lm.speculative_argmax_from_hidden(verify_hidden)
                    verifier_token = verifier_all_tokens.reshape(-1)[relative]

                    saved_position_ids = getattr(lm, "_position_ids", None)
                    saved_rope_deltas = getattr(lm, "_rope_deltas", None)
                    lm._position_ids = None
                    lm._rope_deltas = None
                    replay_cache = lm.make_cache()
                    same_round_prefix = [
                        int(token)
                        for token in verify_input[0, 1 : relative + 1].tolist()
                    ]
                    replay_prompt = mx.array(prompt_tokens, dtype=mx.uint32)
                    for offset in range(0, len(prompt_tokens), prefill_step_size):
                        replay_output = lm(
                            replay_prompt[offset : offset + prefill_step_size][None],
                            cache=replay_cache,
                            return_hidden=True,
                            skip_logits=True,
                        )
                        mx.eval(
                            [entry.state for entry in replay_cache if entry is not None]
                        )
                    replay_output = None
                    for token in generated + same_round_prefix:
                        replay_output = lm(
                            mx.array([[token]], dtype=mx.uint32),
                            cache=replay_cache,
                        )
                    if replay_output is None:
                        raise RuntimeError("MTP diagnostic replay prefix is empty")
                    replay_token = mx.argmax(replay_output.logits[:, -1, :], axis=-1)
                    mx.eval(verifier_token, replay_token)
                    print(
                        "[mtp-diagnostic] "
                        f"round={len(accept_lens)} "
                        f"generated_prefix_tokens={len(generated)} "
                        f"target_output_index={diagnostic_token if token_matches else len(generated)} "
                        f"verify_relative={relative} "
                        f"verify_input={verify_input[0].tolist()} "
                        f"verifier_next={int(verifier_token.reshape(-1).item())} "
                        f"replay_prefix_extra={same_round_prefix} "
                        f"replay_next={int(replay_token.reshape(-1).item())}"
                    )
                    lm._position_ids = saved_position_ids
                    lm._rope_deltas = saved_rope_deltas
                return result

            lm.speculative_verify_hidden = verify_hidden_with_diagnostic
        eos_ids = set(getattr(self.tokenizer, "eos_token_ids", ()) or ())
        mtp_profile = os.environ.get("MINIMLX_MTP_PROFILE_ROUNDS") == "1"
        profile_samples: dict[str, list[float]] = {}
        layer_profile_samples: dict[str, list[float]] = {}
        profile_originals: list[tuple[Any, str, Any]] = []
        if mtp_profile:
            import mlx_vlm.speculative.mtp as mtp_module

            from mlx_vlm.models.qwen3_5.language import _EXACT_SPECULATIVE_VERIFIER

            verifier = _EXACT_SPECULATIVE_VERIFIER
            original_layer = verifier._layer

            def timed_layer(layer, *args, **kwargs):
                started = time.perf_counter()
                value = original_layer(layer, *args, **kwargs)
                mx.eval(value)
                group = "linear" if layer.is_linear else "full"
                layer_profile_samples.setdefault(group, []).append(
                    (time.perf_counter() - started) * 1000.0
                )
                return value

            setattr(verifier, "_layer", timed_layer)
            profile_originals.append((verifier, "_layer", original_layer))

            def instrument(owner, name, sync_result):
                original = getattr(owner, name)

                def timed(*args, **kwargs):
                    started = time.perf_counter()
                    value = original(*args, **kwargs)
                    sync_result(value)
                    profile_samples.setdefault(name, []).append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    return value

                setattr(owner, name, timed)
                profile_originals.append((owner, name, original))

            def sync_arrays(value):
                if isinstance(value, mx.array):
                    mx.eval(value)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        sync_arrays(item)

            def sync_verify(value):
                mx.eval(value.hidden, value.target_tokens)

            for method_name in ("_attention", "_gated_delta", "_feed_forward"):
                instrument(verifier, method_name, sync_arrays)
            instrument(mtp_module, "_mtp_verify_target", sync_verify)
            instrument(mtp_module, "_mtp_acceptance_walk", sync_arrays)

            original_commit = mtp_module._MTPVerifyResult.commit

            def timed_commit(verify_result, model, caches, accepted, block_size):
                started = time.perf_counter()
                value = original_commit(
                    verify_result, model, caches, accepted, block_size
                )
                cache_arrays = []

                def collect_cache_arrays(item):
                    if isinstance(item, mx.array):
                        cache_arrays.append(item)
                    elif isinstance(item, (tuple, list)):
                        for child in item:
                            collect_cache_arrays(child)

                for cache_entry in caches:
                    collect_cache_arrays(getattr(cache_entry, "state", None))
                if cache_arrays:
                    mx.eval(*cache_arrays)
                profile_samples.setdefault("verify_commit", []).append(
                    (time.perf_counter() - started) * 1000.0
                )
                return value

            mtp_module._MTPVerifyResult.commit = timed_commit
            profile_originals.append(
                (mtp_module._MTPVerifyResult, "commit", original_commit)
            )
            drafter = self._mtp_draft_model
            instrument(drafter, "draft_block", sync_arrays)

            def sync_drafter_state(_value):
                state = getattr(drafter, "_seed_hidden", None)
                token = getattr(drafter, "_seed_token", None)
                arrays = [item for item in (state, token) if isinstance(item, mx.array)]
                if arrays:
                    mx.eval(*arrays)

            instrument(drafter, "accept_verified_tokens", sync_drafter_state)
        decode_start = time.perf_counter()
        if first_token not in eos_ids and max_tokens > 1:
            sampler = lambda logits: mx.argmax(logits, axis=-1)
            # _mtp_rounds normally resets and primes from the retained full
            # prompt hidden. This request has already primed incrementally
            # alongside target prefill, so preserve that cache and skip its
            # duplicate full-sequence priming callback.
            original_reset = drafter.reset
            had_reset_override = "reset" in drafter.__dict__
            previous_reset_override = drafter.__dict__.get("reset")
            had_prefill_override = "prefill_from_target_hidden" in drafter.__dict__
            previous_prefill_override = drafter.__dict__.get(
                "prefill_from_target_hidden"
            )
            preserve_primed_cache = True

            def preserve_reset(_target_model, left_padding=None):
                nonlocal preserve_primed_cache
                if preserve_primed_cache and left_padding is None:
                    preserve_primed_cache = False
                    return drafter._cache
                return original_reset(_target_model, left_padding)

            def already_primed(*_args, **_kwargs):
                return None

            drafter.reset = preserve_reset
            drafter.prefill_from_target_hidden = already_primed
            try:
                with mx.stream(generation_stream):
                    for token, _ in _mtp_rounds(
                        self.model,
                        self._mtp_draft_model,
                        cache,
                        hidden,
                        shared_kv,
                        prompt_tokens=prompt[None],
                        first_bonus=first_token,
                        max_tokens=max_tokens,
                        sampler=sampler,
                        token_dtype=mx.uint32,
                        greedy_sampling=True,
                    ):
                        token = int(token)
                        generated.append(token)
                        accepted = getattr(self._mtp_draft_model, "accept_lens", ())
                        drafted = getattr(self._mtp_draft_model, "draft_lens", ())
                        round_index = len(accepted) - 1
                        if round_index >= 0:
                            while len(round_trace) <= round_index:
                                i = len(round_trace)
                                round_trace.append({
                                    "accepted": int(accepted[i]),
                                    "drafted": int(drafted[i]) if i < len(drafted) else 0,
                                    "tokens": [],
                                })
                            round_trace[round_index]["tokens"].append(token)
                        if token in eos_ids or len(generated) >= max_tokens:
                            break
            finally:
                if had_reset_override:
                    drafter.reset = previous_reset_override
                else:
                    delattr(drafter, "reset")
                if had_prefill_override:
                    drafter.prefill_from_target_hidden = previous_prefill_override
                else:
                    delattr(drafter, "prefill_from_target_hidden")
                if (
                    diagnostic_round >= 0 or diagnostic_token >= 0
                ) and callable(original_verify_hidden):
                    lm.speculative_verify_hidden = original_verify_hidden
                for owner, name, original in reversed(profile_originals):
                    if owner is self._mtp_draft_model and name in ("draft_block", "accept_verified_tokens"):
                        delattr(owner, name)
                    else:
                        setattr(owner, name, original)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        if mtp_profile:
            for name, samples in profile_samples.items():
                total = sum(samples)
                print(
                    f"[mtp-profile] stage={name} calls={len(samples)} "
                    f"total_ms={total:.1f} mean_ms={total / len(samples):.2f}"
                )
            for name, samples in layer_profile_samples.items():
                total = sum(samples)
                print(
                    f"[mtp-layer-profile] group={name} calls={len(samples)} "
                    f"total_ms={total:.1f} mean_ms={total / len(samples):.2f}"
                )
        self._mtp_last_round_trace = round_trace
        return (
            prompt_tokens,
            generated,
            self.tokenizer.decode(generated),
            prefill_ms,
            decode_ms,
        )

    def _install_full_attention_kv_cache_growth(self) -> None:
        """Select the KVCache allocation step for the current benchmark shape."""
        step = int(os.environ.get("MINIMLX_FULL_ATTENTION_KV_CACHE_STEP", "256"))
        if step < 256:
            raise RuntimeError("MINIMLX_FULL_ATTENTION_KV_CACHE_STEP must be >= 256")
        from mlx_lm.models.cache import KVCache

        KVCache.step = step

    def _install_fa256_force_fused(self) -> None:
        from ..kernels.fa256 import install_fa256_force_fused

        install_fa256_force_fused()

    def _install_native_fa256_prefill(self) -> None:
        from ..kernels.fa256 import install_native_fa256_prefill

        install_native_fa256_prefill()

    def _install_ane_projection(self) -> None:
        """Install the opt-in fixed-shape ANE experiment for one projection."""
        if os.environ.get("MINIMLX_ENABLE_ANE") != "1":
            return
        from ..ane import load_from_environment

        layer_index = int(os.environ.get("MINIMLX_ANE_LAYER", "17"))
        layer = self.model.language_model.model.layers[layer_index]
        projection = layer.linear_attn.in_proj_z
        input_dim = projection.weight.shape[1] * 32 // projection.bits
        output_dim = projection.weight.shape[0]
        ane_projection = load_from_environment(input_dim, output_dim)
        if ane_projection is None:
            return
        projection._minimlx_ane_projection = ane_projection
        projection_type = type(projection)
        if not getattr(projection_type, "_minimlx_ane_dispatch_installed", False):
            stock_call = projection_type.__call__

            def dispatch(self_projection: Any, value: Any) -> Any:
                candidate = getattr(self_projection, "_minimlx_ane_projection", None)
                if (
                    candidate is not None
                    and value.ndim == 3
                    and value.shape[0] == 1
                    and value.shape[1] == candidate.tokens
                    and value.shape[2] == candidate.input_dim
                ):
                    return candidate(value)
                return stock_call(self_projection, value)

            projection_type.__call__ = dispatch
            projection_type._minimlx_ane_dispatch_installed = True
        self._ane_projection = ane_projection

    def _install_ane_gdn_stage(self) -> None:
        """Opt-in shared-boundary ANE dispatch for one long-prefill GDN layer."""
        if os.environ.get("MINIMLX_ENABLE_ANE_GDN_STAGE") != "1":
            return
        import mlx.core as mx

        from ..ane import ANEGDNStage, load_from_environment

        layer_index = int(os.environ.get("MINIMLX_ANE_GDN_LAYER", "17"))
        layer = self.model.language_model.model.layers[layer_index]
        mixer = layer.linear_attn
        projections = (
            mixer.in_proj_qkv,
            mixer.in_proj_z,
            mixer.in_proj_b,
            mixer.in_proj_a,
        )
        if len({p.weight.shape[1] for p in projections}) != 1:
            raise RuntimeError("GDN ANE stage projections have different input widths")
        input_dim = projections[0].weight.shape[1] * 32 // projections[0].bits
        output_dims = tuple(p.weight.shape[0] for p in projections)
        total_output = sum(output_dims)
        ane_projection = load_from_environment(
            input_dim, total_output, enable_env="MINIMLX_ENABLE_ANE_GDN_STAGE"
        )
        if ane_projection is None:
            return
        stage = ANEGDNStage(ane_projection, *output_dims)

        class _StageProjection:
            def __init__(self, fn, stock):
                self._fn = fn
                self._stock = stock

            def __call__(self, value):
                if (
                    value.ndim == 3
                    and value.shape[0] == 1
                    and value.shape[1] == self._fn.__self__.projection.tokens
                    and value.shape[2] == self._fn.__self__.projection.input_dim
                ):
                    return self._fn(value)
                return self._stock(value)

        mixer.in_proj_qkv = _StageProjection(stage.qkv, projections[0])
        mixer.in_proj_z = _StageProjection(stage.z, projections[1])
        mixer.in_proj_b = _StageProjection(stage.beta, projections[2])
        mixer.in_proj_a = _StageProjection(stage.a, projections[3])
        self._ane_gdn_stage = stage

    def _install_q4_mlp_prefill_dense(self) -> None:
        from ..kernels.mlp import install_q4_mlp_prefill_dense

        install_q4_mlp_prefill_dense(self.model)

    def _install_q4_mlp_prefill_fusion(self) -> None:
        """Fuse only long-prefill gate/up Q4 projections; decode is unchanged."""
        from ..kernels.mlp import install_q4_mlp_prefill_fusion

        install_q4_mlp_prefill_fusion(self.model, min_tokens=1024)

    def _install_q4_native_prefill(self) -> None:
        from ..kernels.q4_native import install_q4_native_prefill

        # Prefill is chunked at 1024 tokens in the MiniMLX long-context
        # configuration.  Keep the native Q4 path enabled for complete
        # production chunks; a 2048 gate would only affect the diagnostic
        # full-context projection probe, not the actual chunk loop.
        install_q4_native_prefill(self.model, min_tokens=1024)

    def _install_q4_native_mlp(self) -> None:
        """Route Qwen3.5 long-prefill MLPs through the native Q4 QMM.

        oMLX patches the Qwen3.5 MLP class itself (gate/up -> SiLU -> down),
        not merely the three QuantizedLinear objects.  The class-level route
        is required because MiniMLX's dense-prefill patch otherwise bypasses
        the projection-level dispatcher entirely.
        """
        from ..kernels.q4_native import install_q4_native_mlp

        install_q4_native_mlp(self.model, min_tokens=1024)

    def _install_gdn_prefill_kernel(self) -> None:
        """Route supported Qwen GDN prefills through MiniMLX's kernel.

        The patch is intentionally narrow: the custom dispatcher returns to
        mlx-lm for decode, masks, vector gating, and unsupported dimensions.
        Set MINIMLX_DISABLE_CUSTOM_GDN=1 to make the baseline explicit.
        """
        if os.environ.get("MINIMLX_DISABLE_CUSTOM_GDN") == "1":
            return
        from mlx_lm.models import gated_delta
        from ..kernels.gdn import gated_delta_prefill

        if getattr(gated_delta, "_minimlx_gdn_installed", False):
            return

        stock_kernel = gated_delta.gated_delta_kernel

        def dispatch(q, k, v, g, beta, state, mask=None):
            candidate = gated_delta_prefill(q, k, v, g, beta, state, mask)
            if candidate is not None:
                return candidate
            return stock_kernel(q, k, v, g, beta, state, mask)

        gated_delta.gated_delta_kernel = dispatch
        gated_delta._minimlx_gdn_installed = True

    def create_plan(self, phase: str) -> ExecutionPlan:
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"unsupported phase: {phase}")
        return ExecutionPlan(phase=phase, use_custom_kernels=True)

    def clear_session(self, session_id: str) -> None:
        self._session_prefix_caches.pop(session_id, None)

    def _generate_greedy_fast(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        prefill_step_size: int,
    ):
        """Greedy-only path which avoids computing unused log probabilities.

        This path is deliberately narrow.  The normal mlx-lm generation path
        remains the fallback for prefix caches, processors, and non-greedy
        samplers.  ``argmax(logits)`` is mathematically identical to
        ``argmax(logits - logsumexp(logits))``.
        """
        import mlx.core as mx
        from mlx_lm.generate import generation_stream

        if not prompt_tokens:
            raise ValueError("greedy fast path requires a non-empty prompt")
        cache = self.model.make_cache()
        prompt = mx.array(prompt_tokens, dtype=mx.uint32)
        processed = 0
        prefill_start = time.perf_counter()
        # Keep prefill, cache-state evaluation, chunk synchronization, and the
        # first-token computation on the same stream.  mlx_lm.generate_step
        # places its complete prefill loop in generation_stream; doing only
        # decode there leaves the fast path's work and explicit stream syncs on
        # different queues.
        with mx.stream(generation_stream):
            while len(prompt_tokens) - processed > 1:
                remaining = len(prompt_tokens) - processed - 1
                count = min(prefill_step_size, remaining)
                self.model(
                    prompt[processed : processed + count][None], cache=cache
                )
                mx.eval([c.state for c in cache])
                self._clear_prefill_allocator_cache(mx, generation_stream)
                processed += count

            logits = self.model(prompt[processed:][None], cache=cache)[:, -1, :]
            sampled = mx.argmax(logits, axis=-1)
            # TTFT ends when the first token is available to the caller, not
            # when its evaluation has merely been submitted.
            first_token = int(sampled.item())
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

        generated: list[int] = [first_token]
        decode_start = time.perf_counter()
        with mx.stream(generation_stream):
            generated.extend(
                self._greedy_decode_device_tail(mx, sampled, cache, max_tokens - 1)
            )
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        output = self.tokenizer.decode(generated)
        return prompt_tokens, generated, output, prefill_ms, decode_ms

    def _greedy_decode_device_tail(self, mx, sampled, cache, token_count: int) -> list[int]:
        """Decode remaining greedy tokens without per-token host round-trips."""
        if token_count <= 0:
            return []

        # Keep sampled IDs as device arrays for the next forward. The
        # non-streaming Runtime API does not need a Python token after every
        # step; extracting ints and rebuilding mx.array inputs adds a
        # host/device synchronization to the decode loop.
        next_token = sampled.astype(mx.uint32)
        generated_tail: list[Any] = []
        for _ in range(token_count):
            next_input = next_token[:, None]
            logits = self.model(next_input, cache=cache)[:, -1, :]
            next_token = mx.argmax(logits, axis=-1).astype(mx.uint32)
            mx.async_eval(next_token)
            generated_tail.append(next_token)

        generated_array = mx.concatenate(generated_tail, axis=0)
        mx.eval(generated_array)
        return [int(token) for token in generated_array.tolist()]

    @staticmethod
    def _clear_prefill_allocator_cache(mx, stream) -> None:
        """Mirror oMLX's synchronized Metal-pool cleanup at chunk boundaries."""
        mx.synchronize(stream)
        mx.clear_cache()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 16,
        context_tokens: int | None = None,
        prefill_step_size: int = 2048,
        session_id: str | None = None,
    ):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        from mlx_lm import stream_generate

        encode = getattr(self.tokenizer, "encode", None)
        if context_tokens is not None:
            prompt_tokens = self._encode_prefix(encode, prompt, context_tokens)
            generation_prompt = prompt_tokens
        else:
            prompt_tokens = encode(prompt, add_special_tokens=False) if encode else []
            generation_prompt = prompt

        effective_prefill_step_size = choose_prefill_step_size(
            len(prompt_tokens), prefill_step_size
        )

        if self._mtp_enabled:
            return self._generate_mtp_greedy(
                prompt_tokens,
                max_tokens,
                effective_prefill_step_size,
            )

        prefix_cache = self._prefix_cache
        if session_id is not None and self._prefix_cache is not None:
            prefix_cache = self._session_prefix_caches.setdefault(
                session_id,
                PrefixCache(
                    self.model,
                    self._prefix_cache.max_bytes,
                ),
            )

        prompt_cache = None
        cached_prefix_len = 0
        if prefix_cache is not None and context_tokens is not None:
            hit = prefix_cache.lookup(prompt_tokens)
            if hit is not None:
                cached_prefix_len, prompt_cache = hit
                generation_prompt = prompt_tokens[cached_prefix_len:]
                if os.environ.get("MINIMLX_PREFIX_CACHE_TRACE") == "1":
                    print(
                        "MiniMLX prefix cache hit: "
                        f"prefix={cached_prefix_len}, suffix={len(generation_prompt)}",
                        flush=True,
                    )

        cache_snapshot_taken = False

        def on_prompt_progress(processed: int, total: int) -> None:
            nonlocal cache_snapshot_taken
            if (
                prefix_cache is not None
                and prompt_cache is not None
                and not cache_snapshot_taken
                and processed == total
            ):
                prefix_cache.put(prompt_tokens, prompt_cache)
                cache_snapshot_taken = True
                if os.environ.get("MINIMLX_PREFIX_CACHE_TRACE") == "1":
                    print(
                        f"MiniMLX prefix cache stored: tokens={len(prompt_tokens)}, "
                        f"bytes={prefix_cache.bytes}",
                        flush=True,
                    )

        if prefix_cache is not None and context_tokens is not None and prompt_cache is None:
            prompt_cache = self.model.make_cache()

        if (
            prefix_cache is None
            and prompt_cache is None
            and session_id is None
            and max_tokens > 0
        ):
            return self._generate_greedy_fast(
                prompt_tokens,
                max_tokens,
                effective_prefill_step_size,
            )

        stream_kwargs: dict[str, Any] = {
            "prompt": generation_prompt,
            "max_tokens": max_tokens,
            "prefill_step_size": effective_prefill_step_size,
        }
        if prompt_cache is not None:
            stream_kwargs["prompt_cache"] = prompt_cache
            stream_kwargs["prompt_progress_callback"] = on_prompt_progress
        responses = stream_generate(self.model, self.tokenizer, **stream_kwargs)
        text_parts: list[str] = []
        output_tokens: list[int] = []
        prefill_ms = 0.0
        decode_tps = 0.0
        prompt_count = 0
        keep_allocator_cache = (
            self._prefill_keep_allocator_cache
            and context_tokens is not None
            and context_tokens >= 16384
        )
        original_clear_cache = None
        if keep_allocator_cache:
            import mlx.core as mx

            original_clear_cache = mx.clear_cache
            mx.clear_cache = lambda: None
        try:
            for response in responses:
                text_parts.append(response.text)
                output_tokens.append(int(response.token))
                prompt_count = response.prompt_tokens
                prefill_ms = prompt_count * 1000.0 / response.prompt_tps
                decode_tps = response.generation_tps
        finally:
            if original_clear_cache is not None:
                mx.clear_cache = original_clear_cache
        output = "".join(text_parts)
        if os.environ.get("MINIMLX_PREFIX_CACHE_TRACE") == "1" and prefix_cache is not None:
            hits, misses, stores, bytes_used = prefix_cache.stats()
            print(
                "MiniMLX prefix cache stats: "
                f"hits={hits}, misses={misses}, stores={stores}, bytes={bytes_used}",
                flush=True,
            )
        if os.environ.get("MINIMLX_ANE_TRACE") == "1" and self._ane_projection is not None:
            calls, total_ms = self._ane_projection.stats()
            print(
                f"MiniMLX ANE projection stats: calls={calls}, "
                f"bridge_total={total_ms:.2f} ms",
                flush=True,
            )
        if os.environ.get("MINIMLX_FA256_TRACE") == "1":
            from ..kernels.fa256 import fa256_stats

            route_calls, fallback_calls = fa256_stats()
            print(
                f"MiniMLX FA256 stats: route_calls={route_calls}, "
                f"fallback_calls={fallback_calls}",
                file=__import__("sys").stderr,
                flush=True,
            )
        encode = getattr(self.tokenizer, "encode", None)
        decode_ms = len(output_tokens) * 1000.0 / decode_tps if decode_tps else 0.0
        return prompt_tokens, output_tokens, output, prefill_ms, decode_ms


    def profile_prefill(
        self,
        prompt: str,
        *,
        context_tokens: int,
        warmup: int = 1,
        prefill_step_size: int = 1024,
    ) -> dict[str, Any]:
        """Measure the real prefill graph at decoder-layer boundaries.

        This is deliberately a diagnostic path.  It synchronizes after every
        decoder layer so that a large lazy MLX graph cannot hide where time is
        spent.  The normal generate/benchmark path is untouched.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")

        import mlx.core as mx
        from mlx_lm.models.qwen3_5 import DecoderLayer

        encode = getattr(self.tokenizer, "encode", None)
        if encode is None:
            raise RuntimeError("tokenizer does not expose encode()")
        token_ids = self._encode_prefix(encode, prompt, context_tokens)
        tokens = mx.array(token_ids, dtype=mx.uint32)[None]

        def run_once() -> None:
            cache = self.model.make_cache()
            logits = self.model(tokens, cache=cache)
            mx.eval(logits)

        for _ in range(max(0, warmup)):
            run_once()

        groups: list[dict[str, Any]] = []
        original_call = DecoderLayer.__call__
        layer_numbers = {
            id(layer): index for index, layer in enumerate(self.model.layers)
        }
        layer_kinds = ["linear" if layer.is_linear else "full" for layer in self.model.layers]
        group_ends = {
            index
            for index in range(len(layer_kinds))
            if index == len(layer_kinds) - 1
            or layer_kinds[index + 1] != layer_kinds[index]
        }
        active_group: dict[str, Any] | None = None

        def timed_call(layer: Any, hidden_states: Any, mask: Any, cache: Any):
            nonlocal active_group
            index = layer_numbers[id(layer)]
            if active_group is None:
                active_group = {
                    "start_layer": index,
                    "end_layer": index,
                    "kind": layer_kinds[index],
                    "start": time.perf_counter(),
                }
            result = original_call(layer, hidden_states, mask, cache)
            active_group["end_layer"] = index
            if index in group_ends:
                mx.eval(result)
                elapsed = (time.perf_counter() - active_group["start"]) * 1000.0
                groups.append(
                    {
                        "start_layer": active_group["start_layer"],
                        "end_layer": active_group["end_layer"],
                        "kind": active_group["kind"],
                        "milliseconds": elapsed,
                    }
                )
                active_group = None
            return result

        total_start = time.perf_counter()
        try:
            DecoderLayer.__call__ = timed_call
            cache = self.model.make_cache()
            logits = self.model(tokens, cache=cache)
            mx.eval(logits)
        finally:
            DecoderLayer.__call__ = original_call
        total_ms = (time.perf_counter() - total_start) * 1000.0

        # Measure the non-layer boundaries separately, without per-group
        # synchronization.  These runs explain the otherwise unattributed
        # portion of the grouped diagnostic.
        text_model = self.model.language_model
        core = text_model.model
        embedding_start = time.perf_counter()
        embeddings = core.embed_tokens(tokens)
        mx.eval(embeddings)
        embedding_ms = (time.perf_counter() - embedding_start) * 1000.0

        boundary_cache = self.model.make_cache()
        backbone_start = time.perf_counter()
        hidden = core(tokens, cache=boundary_cache, input_embeddings=embeddings)
        mx.eval(hidden)
        backbone_ms = (time.perf_counter() - backbone_start) * 1000.0

        lm_head_start = time.perf_counter()
        if text_model.args.tie_word_embeddings:
            logits = core.embed_tokens.as_linear(hidden)
        else:
            logits = text_model.lm_head(hidden)
        mx.eval(logits)
        lm_head_ms = (time.perf_counter() - lm_head_start) * 1000.0

        # Reproduce mlx-lm.generate_step's actual prefill boundary.  Intermediate
        # chunk logits are intentionally not evaluated because generate_step
        # discards them; only cache state is synchronized.  The final token
        # evaluates logits and the greedy sample.
        from mlx_lm.generate import generation_stream

        def actual_prefill_once() -> dict[str, Any]:
            cache = self.model.make_cache()
            prompt_1d = mx.array(token_ids, dtype=mx.uint32)
            chunk_samples: list[float] = []
            processed = 0
            total_start = time.perf_counter()
            with mx.stream(generation_stream):
                while len(token_ids) - processed > 1:
                    remaining = (len(token_ids) - processed) - 1
                    count = min(prefill_step_size, remaining)
                    start = time.perf_counter()
                    self.model(
                        prompt_1d[processed : processed + count][None],
                        cache=cache,
                    )
                    mx.eval([c.state for c in cache])
                    self._clear_prefill_allocator_cache(mx, generation_stream)
                    chunk_samples.append((time.perf_counter() - start) * 1000.0)
                    processed += count
                start = time.perf_counter()
                logits = self.model(prompt_1d[processed:][None], cache=cache)
                logits = logits[:, -1, :]
                sampled = mx.argmax(logits, axis=-1)
                mx.eval(sampled, logits)
                final_ms = (time.perf_counter() - start) * 1000.0
            return {
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
                "chunk_ms": sum(chunk_samples),
                "final_token_ms": final_ms,
                "chunks": chunk_samples,
            }

        for _ in range(max(0, warmup)):
            actual_prefill_once()
        actual = actual_prefill_once()

        linear_ms = sum(item["milliseconds"] for item in groups if item["kind"] == "linear")
        full_ms = sum(item["milliseconds"] for item in groups if item["kind"] == "full")
        projection_stages = self._profile_projection_stages(embeddings)
        return {
            "prompt_tokens": len(token_ids),
            "total_ms": total_ms,
            "linear_attention_ms": linear_ms,
            "full_attention_ms": full_ms,
            "unattributed_ms": total_ms - linear_ms - full_ms,
            "embedding_ms": embedding_ms,
            "backbone_ms": backbone_ms,
            "lm_head_ms": lm_head_ms,
            "actual_prefill_ms": actual["total_ms"],
            "actual_chunk_ms": actual["chunk_ms"],
            "actual_final_token_ms": actual["final_token_ms"],
            "actual_chunks": actual["chunks"],
            "groups": groups,
            "projection_stages": projection_stages,
        }

    def _profile_projection_stages(self, embeddings: Any) -> list[dict[str, Any]]:
        """Time real Qwen3.5 Q4 projection calls used by the production graph.

        This is a diagnostic sample of one linear, one full-attention, and one
        MLP layer.  It intentionally measures MLX's production quantized
        modules directly, matching the operation classes covered by oMLX's
        Qwen3.5 Q4 patch without copying its implementation.
        """
        import mlx.core as mx

        stages: list[dict[str, Any]] = []

        def timed(name: str, layer_index: int, projection: Any, fn: Any, path: str) -> None:
            if (
                path == "mlx_quantized_matmul"
                and getattr(projection, "_minimlx_q4_native_prefill", False)
                and embeddings.ndim == 3
                and embeddings.shape[-2]
                >= getattr(projection, "_minimlx_q4_native_min_tokens", 1024)
            ):
                path = "minimlx_q4_native"
            start = time.perf_counter()
            eligible = (
                getattr(projection, "bits", None) == 4
                and getattr(projection, "group_size", None) in (64, 128)
                and getattr(projection, "mode", None) == "affine"
            )
            try:
                value = fn()
                mx.eval(value)
            except Exception as exc:
                stages.append({
                    "name": name,
                    "layer": layer_index,
                    "milliseconds": (time.perf_counter() - start) * 1000.0,
                    "path": f"ERROR:{type(exc).__name__}",
                    "threshold": getattr(projection, "_minimlx_q4_native_min_tokens", 1024),
                    "eligible": eligible,
                })
                return
            stages.append({
                "name": name,
                "layer": layer_index,
                "milliseconds": (time.perf_counter() - start) * 1000.0,
                "path": path,
                "threshold": getattr(projection, "_minimlx_q4_native_min_tokens", 1024),
                "eligible": eligible,
            })

        def projection_probe_input(projection: Any, reference: Any) -> Any:
            """Build a shape-compatible sample for an output projection.

            Attention output projections consume the concatenated head/value
            state, not the decoder hidden state used by input projections.
            The probe only measures the projection kernel, so zeros are enough
            once the input width matches the projection weights.
            """
            input_width = int(projection.weight.shape[-1])
            bits = getattr(projection, "bits", None)
            if bits is not None:
                # QuantizedLinear stores packed weights: for example, a
                # 4-bit weight has one stored column for eight input values.
                input_width = input_width * 32 // int(bits)
            if reference.shape[-1] == input_width:
                return reference
            return mx.zeros((*reference.shape[:-1], input_width), dtype=reference.dtype)

        layers = self.model.language_model.model.layers
        linear_index = next((i for i, layer in enumerate(layers) if layer.is_linear), None)
        full_index = next((i for i, layer in enumerate(layers) if not layer.is_linear), None)
        if linear_index is not None:
            layer = layers[linear_index]
            x = layer.input_layernorm(embeddings)
            mx.eval(x)
            mixer = layer.linear_attn
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                projection = getattr(mixer, name, None)
                if projection is not None:
                    projection_input = (
                        projection_probe_input(projection, x)
                        if name == "out_proj"
                        else x
                    )
                    timed(
                        f"linear.{name}", linear_index, projection,
                        lambda p=projection, value=projection_input: p(value),
                        "mlx_quantized_matmul",
                    )
        if full_index is not None:
            layer = layers[full_index]
            x = layer.input_layernorm(embeddings)
            mx.eval(x)
            attention = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                projection = getattr(attention, name, None)
                if projection is not None:
                    projection_input = (
                        projection_probe_input(projection, x)
                        if name == "o_proj"
                        else x
                    )
                    timed(
                        f"full.{name}", full_index, projection,
                        lambda p=projection, value=projection_input: p(value),
                        "mlx_quantized_matmul",
                    )
        mlp_layer = layers[0]
        x = mlp_layer.post_attention_layernorm(embeddings)
        mx.eval(x)
        mlp = mlp_layer.mlp
        mlp_inputs: dict[str, Any] = {}
        native_mlp_active = getattr(
            type(mlp), "_minimlx_q4_native_mlp_installed", False
        )
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(mlp, name, None)
            if projection is not None:
                if native_mlp_active:
                    path = "minimlx_q4_native_mlp"
                elif getattr(mlp, "_minimlx_prefill_dense", None) is not None:
                    path = "minimlx_dense_bf16"
                elif getattr(mlp, "_minimlx_fused_gate_up", None) is not None:
                    path = "minimlx_q4_fused_qmm"
                else:
                    path = "mlx_quantized_matmul"
                if (
                    path == "mlx_quantized_matmul"
                    and getattr(projection, "_minimlx_q4_native_prefill", False)
                    and embeddings.shape[-2]
                    >= getattr(projection, "_minimlx_q4_native_min_tokens", 1024)
                ):
                    path = "minimlx_q4_native"
                if name == "down_proj":
                    gate = mlp_inputs.get("gate")
                    up = mlp_inputs.get("up")
                    if gate is None or up is None:
                        continue
                    # SiLU is exposed by mlx.nn, not mlx.core. Keep this
                    # probe aligned with the actual MLP implementation.
                    from mlx import nn

                    down_input = nn.silu(gate) * up
                    mx.eval(down_input)
                    timed(
                        f"mlp.{name}", 0, projection,
                        lambda p=projection, value=down_input: p(value), path
                    )
                else:
                    start = time.perf_counter()
                    try:
                        value = projection(x)
                        mx.eval(value)
                    except Exception as exc:
                        stages.append({
                            "name": f"mlp.{name}",
                            "layer": 0,
                            "milliseconds": (time.perf_counter() - start) * 1000.0,
                            "path": f"ERROR:{type(exc).__name__}",
                            "threshold": getattr(projection, "_minimlx_q4_native_min_tokens", 1024),
                            "eligible": (
                                getattr(projection, "bits", None) == 4
                                and getattr(projection, "group_size", None) in (64, 128)
                                and getattr(projection, "mode", None) == "affine"
                            ),
                        })
                        continue
                    mlp_inputs[name.removesuffix("_proj")] = value
                    stages.append({
                        "name": f"mlp.{name}",
                        "layer": 0,
                        "milliseconds": (time.perf_counter() - start) * 1000.0,
                        "path": path,
                        "threshold": getattr(projection, "_minimlx_q4_native_min_tokens", 1024),
                        "eligible": (
                            getattr(projection, "bits", None) == 4
                            and getattr(projection, "group_size", None) in (64, 128)
                            and getattr(projection, "mode", None) == "affine"
                        ),
                    })
        return stages

    def profile_projection_fusion(
        self,
        prompt: str,
        *,
        context_tokens: int,
        warmup: int = 1,
        repeats: int = 5,
    ) -> dict[str, Any]:
        """A/B shared-input quantized projection fusion on the first linear layer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        import mlx.core as mx

        encode = getattr(self.tokenizer, "encode", None)
        token_ids = self._encode_prefix(encode, prompt, context_tokens)
        tokens = mx.array(token_ids, dtype=mx.uint32)[None]
        core = self.model.language_model.model
        layer = core.layers[0]
        if not layer.is_linear:
            raise RuntimeError("L0 is not a linear-attention layer")
        embeddings = core.embed_tokens(tokens)
        mx.eval(embeddings)
        inputs = layer.input_layernorm(embeddings)
        mx.eval(inputs)

        qkv = layer.linear_attn.in_proj_qkv
        z = layer.linear_attn.in_proj_z
        required = ("weight", "scales", "biases", "group_size", "bits", "mode")
        if any(not hasattr(qkv, name) or not hasattr(z, name) for name in required):
            return {"supported": False, "reason": "projection metadata unavailable"}
        if (qkv.group_size, qkv.bits, qkv.mode) != (z.group_size, z.bits, z.mode):
            return {"supported": False, "reason": "quantization metadata differs"}
        if qkv.weight.shape[1] != z.weight.shape[1]:
            return {"supported": False, "reason": "input dimensions differ"}

        weight = mx.concatenate([qkv.weight, z.weight], axis=0)
        scales = mx.concatenate([qkv.scales, z.scales], axis=0)
        biases = mx.concatenate([qkv.biases, z.biases], axis=0)
        split = qkv.weight.shape[0]

        def direct() -> tuple[Any, Any]:
            left = qkv(inputs)
            right = z(inputs)
            mx.eval(left, right)
            return left, right

        def fused() -> tuple[Any, Any]:
            output = mx.quantized_matmul(
                inputs,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=qkv.group_size,
                bits=qkv.bits,
                mode=qkv.mode,
            )
            left, right = output[..., :split], output[..., split:]
            mx.eval(left, right)
            return left, right

        for _ in range(max(0, warmup)):
            direct()
            fused()
        direct_times: list[float] = []
        fused_times: list[float] = []
        max_error = 0.0
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            direct_left, direct_right = direct()
            direct_times.append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            fused_left, fused_right = fused()
            fused_times.append((time.perf_counter() - start) * 1000.0)
            error = mx.maximum(
                mx.max(mx.abs(direct_left - fused_left)),
                mx.max(mx.abs(direct_right - fused_right)),
            )
            mx.eval(error)
            max_error = max(max_error, float(error.item()))

        direct_times.sort()
        fused_times.sort()
        return {
            "supported": True,
            "direct_ms": direct_times[len(direct_times) // 2],
            "fused_ms": fused_times[len(fused_times) // 2],
            "speedup": direct_times[len(direct_times) // 2]
            / fused_times[len(fused_times) // 2],
            "max_abs_error": max_error,
            "input_shape": list(inputs.shape),
            "qkv_output": int(qkv.weight.shape[0]),
            "z_output": int(z.weight.shape[0]),
        }

    @staticmethod
    def _encode_prefix(encode, text: str, target_tokens: int) -> list[int]:
        """Encode only enough source text to build the requested context.

        oMLX's code corpus is much larger than the selected context window.
        Encoding it as one string causes tokenizer max-length warnings and
        wastes substantial CPU time before the prefix is discarded.
        """
        if target_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        result: list[int] = []
        # Large enough to preserve normal BPE merges across the useful prefix,
        # while still far below the model tokenizer's maximum length.
        chunk_size = 65536
        for start in range(0, len(text), chunk_size):
            chunk = text[start : start + chunk_size]
            result.extend(encode(chunk, add_special_tokens=False))
            if len(result) >= target_tokens:
                return result[:target_tokens]
        if not result:
            raise ValueError("cannot build a fixed-token prompt from empty text")
        repeats = (target_tokens + len(result) - 1) // len(result)
        return (result * repeats)[:target_tokens]
