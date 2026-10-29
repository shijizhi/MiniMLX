"""Qwen3.5-only optimization implementation composed over baseline adapters."""

from __future__ import annotations

import os
import time
from typing import Any

from ..scheduling import choose_prefill_step_size

class Qwen35OptimizationMixin:
    """Optimization algorithms; runtime state is owned by the host adapter.

    This mixin is used only by Qwen35PerformanceOverlay. Attribute access is
    proxied there to the loaded baseline host, keeping weights and lifecycle
    outside the optimization layer.
    """

    def _install_qwen35_skip_logits(self) -> None:
        """Allow long-prefill chunks to update cache without materializing logits.

        mlx-lm 0.31.x's Qwen3.5 text wrapper always applies lm_head, while the
        long greedy prefill path only needs the cache state.  The model's inner
        transformer already returns hidden states, so expose that narrow option
        without changing the stock call contract used by normal generation.
        """
        if os.environ.get("MINIMLX_PREFILL_SKIP_LOGITS", "1") != "1":
            return
        language_model = getattr(self.model, "language_model", None)
        if language_model is None or not hasattr(language_model, "model"):
            return
        model_type = type(language_model)
        if getattr(model_type, "_minimlx_skip_logits_installed", False):
            return
        stock_call = model_type.__call__

        def dispatch(model_instance, inputs, cache=None, input_embeddings=None,
                     skip_logits=False):
            if not skip_logits:
                return stock_call(
                    model_instance,
                    inputs,
                    cache=cache,
                    input_embeddings=input_embeddings,
                )
            return model_instance.model(
                inputs, cache, input_embeddings=input_embeddings
            )

        model_type.__call__ = dispatch
        model_type._minimlx_skip_logits_installed = True
        model_type._minimlx_skip_logits_original = stock_call
        self._qwen35_skip_logits_class = model_type
        self._qwen35_skip_logits_original = stock_call

    @staticmethod
    def _install_gdn_decode_conv_silu(model) -> None:
        if os.environ.get("MINIMLX_GDN_DECODE_FUSED_CONV_SILU") != "1":
            return
        from ..kernels.gdn_conv_silu import install_gdn_decode_conv_silu

        count = install_gdn_decode_conv_silu(model)
        if count < 1:
            raise RuntimeError(
                "GDN conv+SiLU candidate was requested but no compatible Qwen GDN "
                "layers were found; refusing to benchmark an inactive candidate."
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
        kv_bits = os.environ.get("MINIMLX_KV_CACHE_BITS")
        # The hand-written fast path owns a plain cache.  Use mlx-lm's
        # generator when KV quantization is explicitly requested so its
        # quantized cache transition is applied at the right token boundary.
        if kv_bits not in (None, "", "0"):
            raise AssertionError(
                "_generate_greedy_fast must not be selected with KV quantization"
            )
        cache = self.model.make_cache()
        prompt = mx.array(prompt_tokens, dtype=mx.uint32)
        model_call = self.model
        if hasattr(self.model, "language_model"):
            model_call = self.model.language_model
        processed = 0
        skip_intermediate_logits = (
            os.environ.get("MINIMLX_PREFILL_SKIP_LOGITS", "1") == "1"
        )
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
                prefill_kwargs = {"cache": cache}
                if skip_intermediate_logits:
                    # No intermediate prefill logits are consumed by greedy
                    # generation. Some mlx-lm model versions expose this
                    # switch, while older versions reject the keyword.
                    prefill_kwargs["skip_logits"] = True
                try:
                    model_call(
                        prompt[processed : processed + count][None], **prefill_kwargs
                    )
                except TypeError as exc:
                    if not skip_intermediate_logits or "skip_logits" not in str(exc):
                        raise
                    # Retry the same untouched chunk through the stock model
                    # contract and keep the capability disabled for the rest
                    # of this request. This is a compatibility fallback, not
                    # a benchmark failure.
                    skip_intermediate_logits = False
                    model_call(
                        prompt[processed : processed + count][None], cache=cache
                    )
                mx.eval([c.state for c in cache])
                self._clear_prefill_allocator_cache(mx, generation_stream)
                processed += count

            logits = model_call(prompt[processed:][None], cache=cache)[:, -1, :]
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

    def _generate_optimized(
        self,
        prompt: str,
        max_tokens: int = 16,
        context_tokens: int | None = None,
        prefill_step_size: int = 2048,
        session_id: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_context_tokens: int | None = None,
        on_token=None,
    ):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        from mlx_lm import stream_generate

        # The context benchmark needs to be able to distinguish a real
        # end-of-turn from a decode/prefill error.  mlx-lm implements EOS
        # stopping inside stream_generate by reading tokenizer.eos_token_ids,
        # so make the policy explicit instead of silently changing it in a
        # benchmark wrapper.
        ignore_eos = os.environ.get("MINIMLX_IGNORE_EOS") == "1"

        encode = getattr(self.tokenizer, "encode", None)
        if max_context_tokens is not None:
            if max_context_tokens <= 0:
                raise ValueError("max_context_tokens must be positive")
            prompt_tokens = encode(prompt, add_special_tokens=False) if encode else []
            if len(prompt_tokens) > max_context_tokens:
                prompt_tokens = prompt_tokens[-max_context_tokens:]
            generation_prompt = prompt_tokens
        else:
            prompt_tokens = None
            generation_prompt = prompt
        if context_tokens is not None:
            prompt_tokens = self._encode_prefix(encode, prompt, context_tokens)
            generation_prompt = prompt_tokens
        elif prompt_tokens is None:
            prompt_tokens = encode(prompt, add_special_tokens=False) if encode else []

        effective_prefill_step_size = choose_prefill_step_size(
            len(prompt_tokens), prefill_step_size
        )

        if self._dflash_enabled:
            if min_p != 0.0 or repetition_penalty != 1.0 or presence_penalty != 0.0 or frequency_penalty != 0.0:
                raise ValueError(
                    "DFlash currently supports temperature, top-p, and top-k sampling; "
                    "disable min-p and repetition/presence/frequency penalties."
                )
            from dflash.model_mlx import stream_generate as dflash_stream_generate
            configured_block_size = int(
                getattr(getattr(self._dflash_draft_model, "config", None), "block_size", 5)
            )
            block_size = min(
                configured_block_size,
                int(os.environ.get("MINIMLX_DFLASH_BLOCK_SIZE", "5")),
            )
            if block_size < 2:
                raise ValueError("MINIMLX_DFLASH_BLOCK_SIZE must be at least 2")

            original_eos_ids = None
            if os.environ.get("MINIMLX_IGNORE_EOS") == "1":
                original_eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
                self.tokenizer.eos_token_ids = set()
            try:
                responses = dflash_stream_generate(
                    self.model,
                    self._dflash_draft_model,
                    self.tokenizer,
                    generation_prompt,
                    block_size=block_size,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    prefill_step_size=effective_prefill_step_size,
                )
                text_parts: list[str] = []
                output_tokens: list[int] = []
                prefill_ms = 0.0
                decode_tps = 0.0
                for response in responses:
                    text_parts.append(response.text)
                    response_tokens = [int(token) for token in response.tokens]
                    output_tokens.extend(response_tokens)
                    if on_token is not None:
                        on_token(response.text, response_tokens)
                    if response.prompt_tokens and response.prompt_tps > 0:
                        prefill_ms = response.prompt_tokens * 1000.0 / response.prompt_tps
                    decode_tps = response.generation_tps
            finally:
                if original_eos_ids is not None:
                    self.tokenizer.eos_token_ids = original_eos_ids
            text = "".join(text_parts)
            decode_ms = len(output_tokens) * 1000.0 / decode_tps if decode_tps else 0.0
            return prompt_tokens, output_tokens, text, prefill_ms, decode_ms

        greedy = (
            temperature == 0.0
            and min_p == 0.0
            and repetition_penalty == 1.0
            and presence_penalty == 0.0
            and frequency_penalty == 0.0
        )
        if self._mtp_enabled and greedy:
            return self._generate_mtp_greedy(
                prompt_tokens,
                max_tokens,
                effective_prefill_step_size,
                on_token=on_token,
            )

        prefix_cache = self._prefix_cache
        if session_id is not None and self._prefix_cache is not None:
            from ..prompt_cache import PrefixCache

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
            and greedy
            and on_token is None
            and os.environ.get("MINIMLX_KV_CACHE_BITS") in (None, "", "0")
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
        kv_bits = os.environ.get("MINIMLX_KV_CACHE_BITS")
        if kv_bits not in (None, "", "0"):
            bits = int(kv_bits)
            if bits not in (4, 8):
                raise ValueError("MINIMLX_KV_CACHE_BITS must be 4, 8, or 0")
            stream_kwargs.update(
                kv_bits=bits,
                kv_group_size=int(os.environ.get("MINIMLX_KV_CACHE_GROUP_SIZE", "64")),
                quantized_kv_start=int(
                    os.environ.get("MINIMLX_QUANTIZED_KV_START", "512")
                ),
            )
        if not greedy:
            from mlx_lm.sample_utils import make_logits_processors, make_sampler

            stream_kwargs["sampler"] = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )
            processors = make_logits_processors(
                repetition_penalty=(
                    repetition_penalty if repetition_penalty != 1.0 else None
                ),
                presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
                frequency_penalty=frequency_penalty if frequency_penalty != 0.0 else None,
            )
            if processors:
                stream_kwargs["logits_processors"] = processors
        if prompt_cache is not None:
            stream_kwargs["prompt_cache"] = prompt_cache
            stream_kwargs["prompt_progress_callback"] = on_prompt_progress
        original_eos_ids = None
        if ignore_eos:
            original_eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
            self.tokenizer.eos_token_ids = set()
        try:
            responses = stream_generate(self.model, self.tokenizer, **stream_kwargs)
        except BaseException:
            if original_eos_ids is not None:
                self.tokenizer.eos_token_ids = original_eos_ids
            raise
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
                token_id = int(response.token)
                output_tokens.append(token_id)
                if on_token is not None:
                    on_token(response.text, [token_id])
                prompt_count = response.prompt_tokens
                prefill_ms = prompt_count * 1000.0 / response.prompt_tps
                decode_tps = response.generation_tps
        finally:
            if original_eos_ids is not None:
                self.tokenizer.eos_token_ids = original_eos_ids
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
        # Keep the oMLX-aligned benchmark byte/token exact.  Its largest
        # prompt is only 16K tokens, so encoding the complete decoded prompt
        # avoids introducing artificial BPE boundaries at character chunks.
        if len(text) <= 256 * 1024:
            result = list(encode(text, add_special_tokens=False))
            if len(result) >= target_tokens:
                return result[:target_tokens]
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

    def _qwen_cached_image_features(self, prepared, images):
        """Compute/cache Qwen3.5 vision features before token generation."""
        if not images or self._vision_feature_cache is None:
            return None
        if getattr(getattr(self.model, "config", None), "model_type", None) != "qwen3_5":
            return None
        vision_tower = getattr(self.model, "vision_tower", None)
        if vision_tower is None or prepared.pixel_values is None:
            return None
        cached = self._vision_feature_cache.get(images)
        if cached is not None:
            self._vision_feature_cache_hits += 1
            return cached
        self._vision_feature_cache_misses += 1
        import mlx.core as mx

        dtype = vision_tower.patch_embed.proj.weight.dtype
        pixel_values = prepared.pixel_values.astype(dtype)
        grid_thw = prepared.extra.get("image_grid_thw")
        features, _ = vision_tower(pixel_values, grid_thw)
        mx.eval(features)
        self._vision_feature_cache.put(images, features)
        self._vision_feature_cache_stores += 1
        return features

    def vision_feature_cache_metrics(self) -> dict[str, int]:
        return {
            "hits": self._vision_feature_cache_hits,
            "misses": self._vision_feature_cache_misses,
            "stores": self._vision_feature_cache_stores,
            "entries": len(getattr(self._vision_feature_cache, "_cache", {}))
            if self._vision_feature_cache is not None
            else 0,
        }
