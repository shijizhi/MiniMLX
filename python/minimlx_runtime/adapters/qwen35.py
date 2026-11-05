from __future__ import annotations

from pathlib import Path
import os
import re
import time
import hashlib
from queue import Queue
from threading import RLock, Thread
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..types import Acceleration, ExecutionPlan, ModelDescriptor, Modality
from ..vision_cache import VisionInputCache
from .mlx_vlm import MLXVLMAdapter, _apply_vlm_chat_template, _video_input

_DFLASH_DRAFT_LOAD_LOCK = RLock()

if TYPE_CHECKING:
    from ..prompt_cache import PrefixCache

def _token_ids_as_set(value: Any) -> set[int]:
    """Normalize tokenizer EOS metadata, which varies between backends."""
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(token) for token in value}

def _cleanup_failed_mtp_request(mx: Any, stream: Any, reset_drafter: Any, target: Any,
                                target_cache: list[Any], shared_kv: dict) -> None:
    """Retire request-local MTP state after a failed speculative decode.

    Synchronize before dropping cache arrays so lazy Metal work cannot outlive
    the request. Resetting the drafter also detaches its request-local cache;
    cleanup failures must not replace the original inference exception.
    """
    try:
        mx.synchronize(stream)
    except Exception:
        pass
    try:
        reset_drafter(target)
    except Exception:
        pass
    target_cache.clear()
    shared_kv.clear()
    try:
        mx.clear_cache()
    except Exception:
        pass

@dataclass(frozen=True)
class PreparedVisionInputs:
    """Backend-neutral boundary before VLM prefill/feature injection."""

    prompt: str
    input_ids: Any
    pixel_values: Any | None
    attention_mask: Any | None
    extra: dict[str, Any]

def _vlm_phase_timings(
    *,
    prompt_tokens: int,
    generated_tokens: int,
    prompt_tokens_per_second: float,
    generation_tokens_per_second: float,
    elapsed_ms: float,
) -> tuple[float, float]:
    """Convert mlx-vlm's per-phase rates into MiniMLX latency metrics."""
    prefill_ms = (
        prompt_tokens * 1000.0 / prompt_tokens_per_second
        if prompt_tokens_per_second > 0
        else elapsed_ms
    )
    decode_ms = (
        generated_tokens * 1000.0 / generation_tokens_per_second
        if generation_tokens_per_second > 0
        else max(0.0, elapsed_ms - prefill_ms)
    )
    return prefill_ms, decode_ms

class Qwen35Adapter(MLXVLMAdapter):
    """Qwen3.5 family coordinator composed over standard MLX-LM/MLX-VLM baselines."""

    name = "qwen35-mlx-lm"

    def __init__(self) -> None:
        super().__init__()
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.processor: Any | None = None
        self.descriptor: ModelDescriptor | None = None
        self._baseline_adapter: Any | None = None
        self._optimization_overlay: Any | None = None
        self._optimization_active = False
        self._ane_projection: Any | None = None
        self._ane_gdn_stage: Any | None = None
        self._prefix_cache: PrefixCache | None = None
        self._session_prefix_caches: dict[str, PrefixCache] = {}
        self._vlm_prompt_cache_states = self._prompt_cache_states
        self._greedy_skip_logsumexp = False
        self._qwen35_skip_logits_class: Any | None = None
        self._qwen35_skip_logits_original: Any | None = None
        self._prefill_keep_allocator_cache = False
        self._mtp_draft_model: Any | None = None
        self._mtp_enabled = False
        self._dflash_draft_model: Any | None = None
        self._dflash_enabled = False
        self._mtp_target_language_model: Any | None = None
        self._mtp_argmax_hook: Any | None = None
        self._mtp_last_round_trace: list[dict[str, Any]] = []
        self._vlm_enabled = False
        self._vision_input_cache = VisionInputCache(
            int(os.environ.get("MINIMLX_VISION_INPUT_CACHE_MAX_BYTES", "0"))
        )
        self._vision_feature_cache: Any | None = None
        self._vision_feature_cache_hits = 0
        self._vision_feature_cache_misses = 0
        self._vision_feature_cache_stores = 0

    def can_load(self, descriptor: ModelDescriptor) -> bool:
        value = f"{descriptor.model_type} {descriptor.architecture}".lower()
        return "qwen3" in value and "5" in value

    @staticmethod
    def _mtp_draft_path(plan: ExecutionPlan | None) -> str | None:
        if plan is None or Acceleration.MTP not in plan.accelerations:
            return None
        return plan.acceleration_components.get(Acceleration.MTP)

    @staticmethod
    def _dflash_draft_path(plan: ExecutionPlan | None) -> str | None:
        if plan is None or Acceleration.DFLASH not in plan.accelerations:
            return None
        return plan.acceleration_components.get(Acceleration.DFLASH)

    def load(
        self, descriptor: ModelDescriptor, plan: ExecutionPlan | None = None
    ) -> "Qwen35Adapter":
        if (Path(descriptor.model_dir) / "offload_index.json").is_file():
            if plan is None:
                raise ValueError("Paged Qwen3.5/3.6 requires an execution plan")
            if self._mtp_draft_path(plan) or self._dflash_draft_path(plan):
                raise RuntimeError("External MTP/DFlash drafts are not yet supported with Qwen3.5/3.6 SSD paging")
            return self._load_vlm_backend(descriptor, plan)
        # Qwen3.5 checkpoints can advertise vision components on text-only
        # requests. Keep the selected plan's text baseline in that case.
        if (
            plan is not None
            and plan.backend_adapter == "qwen35-vlm"
            and os.environ.get("MINIMLX_FORCE_TEXT_RUNTIME") != "1"
        ):
            return self._load_vlm_backend(descriptor, plan)
        mtp_draft_path = self._mtp_draft_path(plan)
        if mtp_draft_path:
            return self._load_mtp_backend(descriptor, mtp_draft_path)
        dflash_draft_path = self._dflash_draft_path(plan)

        from .mlx_lm_standard import StandardTextAdapter

        baseline = StandardTextAdapter().load(descriptor, plan)
        self._baseline_adapter = baseline
        self.model, self.tokenizer = baseline.model, baseline.tokenizer
        self.descriptor = descriptor
        self._optimization_active = bool(
            plan and "qwen35-performance-overlay" in plan.optimization_providers
        )
        from .qwen35_overlay import Qwen35PerformanceOverlay

        self._optimization_overlay = Qwen35PerformanceOverlay(self)
        if dflash_draft_path:
            try:
                from dflash import model_mlx
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "DFlash acceleration requires the 'dflash' package in MiniMLX's .venv; "
                    "install it with `pip install dflash` and restart the runtime."
                ) from exc
            # The upstream MLX loader accepts Hub IDs; MiniMLX stores complete
            # checkpoints in its own model directory. Redirect just the
            # snapshot resolver while loading, then restore it immediately.
            with _DFLASH_DRAFT_LOAD_LOCK:
                original_snapshot_download = model_mlx.snapshot_download
                model_mlx.snapshot_download = lambda _repo, **_kwargs: str(
                    Path(dflash_draft_path).expanduser().resolve()
                )
                try:
                    self._dflash_draft_model = model_mlx.load_draft(dflash_draft_path)
                finally:
                    model_mlx.snapshot_download = original_snapshot_download
            self._dflash_enabled = True
            baseline._dflash_draft_model = self._dflash_draft_model
        if self._optimization_active:
            self._optimization_overlay.install_text()
        return self

    def unload(self) -> None:
        """Drop model/session references so the runtime can reclaim device memory."""
        overlay = self._optimization_overlay
        if overlay is not None:
            overlay.uninstall()
        if self.model is not None:
            import sys

            fusion = sys.modules.get("minimlx_runtime.kernels.gdn_conv_silu")
            if fusion is not None:
                fusion.restore_gdn_decode_conv_silu(self.model)
        if (
            self._qwen35_skip_logits_class is not None
            and self._qwen35_skip_logits_original is not None
        ):
            self._qwen35_skip_logits_class.__call__ = (
                self._qwen35_skip_logits_original
            )
            self._qwen35_skip_logits_class._minimlx_skip_logits_installed = False
            self._qwen35_skip_logits_class = None
            self._qwen35_skip_logits_original = None
        self._session_prefix_caches.clear()
        self._vlm_prompt_cache_states.clear()
        self._prefix_cache = None
        self._mtp_last_round_trace.clear()
        self._mtp_draft_model = None
        self._dflash_draft_model = None
        self._dflash_enabled = False
        self._mtp_target_language_model = None
        self._mtp_argmax_hook = None
        self.tokenizer = None
        self.model = None
        self.processor = None
        self._vision_input_cache.clear()
        if self._vision_feature_cache is not None:
            self._vision_feature_cache.clear()
        self._vision_feature_cache = None
        self._vision_feature_cache_hits = 0
        self._vision_feature_cache_misses = 0
        self._vision_feature_cache_stores = 0
        self.descriptor = None
        self._ane_projection = None
        self._ane_gdn_stage = None
        self._mtp_enabled = False
        self._vlm_enabled = False
        self._greedy_skip_logsumexp = False
        self._prefill_keep_allocator_cache = False
        baseline, self._baseline_adapter = self._baseline_adapter, None
        self._optimization_overlay = None
        self._optimization_active = False
        if baseline is not None:
            baseline.unload()
        # Avoid importing MLX in headless use. Clearing its allocator cache after
        # dropping this adapter's references lets Metal reclaim released buffers.
        import sys

        mx = sys.modules.get("mlx.core")
        if mx is not None:
            mx.clear_cache()

    def _load_vlm_backend(
        self, descriptor: ModelDescriptor, plan: ExecutionPlan
    ) -> "Qwen35Adapter":
        """Use the shared mlx-vlm lifecycle, then enable Qwen3.5 cache hooks."""
        baseline = MLXVLMAdapter().load(
            descriptor,
            replace(plan, backend_adapter="mlx-vlm-runtime"),
        )
        self._baseline_adapter = baseline
        self.model = baseline.model
        self.tokenizer = baseline.tokenizer
        self.processor = baseline.processor
        self.descriptor = descriptor
        self._vlm_enabled = True
        self._optimization_active = "qwen35-performance-overlay" in plan.optimization_providers
        from .qwen35_overlay import Qwen35PerformanceOverlay

        self._optimization_overlay = Qwen35PerformanceOverlay(self)
        if self._optimization_active:
            self._optimization_overlay.install_vlm()
        return self

    def _prepare(self, messages, image_parts, video_parts=None, chat_template_kwargs=None):
        prompt, images, prepared, generation_extra = self._multimodal_context(
            messages, image_parts, video_parts, chat_template_kwargs
        )
        return (
            prompt,
            images,
            prepared.input_ids,
            prepared.pixel_values,
            prepared.attention_mask,
            generation_extra,
        )

    def _multimodal_generation_extras(self, images, extra):
        options = dict(extra)
        options.update(
            vision_cache=self._vision_feature_cache,
            # Qwen3.5's mlx-vlm model uses this source key for feature lookup.
            _image_key=images or None,
        )
        return options

    def _multimodal_context(self, prompt, image_parts, video_parts=None, chat_template_kwargs=None):
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        images = [self._image_input(part) for part in image_parts]
        videos, video_metadata, cleanup_paths = [], [], []
        try:
            for part in video_parts or ():
                video, metadata, cleanup = _video_input(part)
                videos.append(video)
                if metadata is not None:
                    video_metadata.append(metadata)
                if cleanup is not None:
                    cleanup_paths.append(cleanup)
        except Exception:
            for path in cleanup_paths:
                Path(path).unlink(missing_ok=True)
            raise
        if video_metadata and len(video_metadata) != len(videos):
            for path in cleanup_paths:
                Path(path).unlink(missing_ok=True)
            raise ValueError("timestamped frame videos cannot be mixed with video-file references in one request")
        try:
            prompt = _apply_vlm_chat_template(
                self.processor,
                self.model.config,
                prompt,
                len(images),
                video_parts,
                chat_template_kwargs,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            if images or videos:
                for path in cleanup_paths:
                    Path(path).unlink(missing_ok=True)
                raise RuntimeError(
                    "the vision checkpoint chat template could not render multimodal content"
                ) from exc
            prompt = "\n".join(
                item.get("text", "")
                for message in prompt
                for item in message.get("content", ())
                if item.get("type") == "text"
            )
        except Exception:
            for path in cleanup_paths:
                Path(path).unlink(missing_ok=True)
            raise

        try:
            if videos:
                prepared = self._prepare_vlm_inputs(
                    prompt,
                    images,
                    videos=videos,
                    video_metadata=video_metadata,
                    prepare_inputs=prepare_inputs,
                    should_add_special_tokens=should_add_special_tokens,
                )
            else:
                cache_key = self._vision_input_cache_key(prompt, image_parts)
                prepared = self._vision_input_cache.get(cache_key)
                if prepared is None:
                    prepared = self._prepare_vlm_inputs(
                        prompt,
                        images,
                        prepare_inputs=prepare_inputs,
                        should_add_special_tokens=should_add_special_tokens,
                    )
                    self._vision_input_cache.put(cache_key, prepared)
        finally:
            for path in cleanup_paths:
                Path(path).unlink(missing_ok=True)

        cached_image_features = (
            self._optimization_overlay._qwen_cached_image_features(prepared, images)
            if not videos and self._optimization_overlay is not None
            else None
        )
        generation_extra = dict(prepared.extra)
        if cached_image_features is not None:
            generation_extra["cached_image_features"] = cached_image_features
        return prompt, images, prepared, generation_extra

    def vision_cache_stats(self) -> tuple[int, int, int, int]:
        """Return (hits, misses, stores, bytes) for prepared VLM inputs."""
        return self._vision_input_cache.stats()

    def vision_feature_cache_metrics(self) -> dict[str, int]:
        if self._optimization_overlay is None:
            return {"hits": 0, "misses": 0, "stores": 0, "entries": 0}
        return self._optimization_overlay.vision_feature_cache_metrics()

    def _vision_input_cache_key(self, prompt: str, image_parts: list[Any]) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.descriptor.model_dir if self.descriptor else "").encode())
        digest.update(prompt.encode())
        for part in image_parts:
            identity = part.data_base64 or part.uri or ""
            if part.uri and part.uri.startswith("file://"):
                from urllib.parse import unquote, urlparse

                path = Path(unquote(urlparse(part.uri).path))
                try:
                    stat = path.stat()
                    identity = f"{identity}:{stat.st_size}:{stat.st_mtime_ns}"
                except OSError:
                    pass
            digest.update(identity.encode())
            digest.update((part.mime_type or "").encode())
        return digest.hexdigest()

    def _prepare_vlm_inputs(
        self,
        prompt: str,
        images: list[Any],
        *,
        prepare_inputs,
        should_add_special_tokens,
        videos=None,
        video_metadata=None,
    ) -> PreparedVisionInputs:
        """Run mlx-vlm preprocessing once, before the generation loop.

        This mirrors oMLX's explicit ``prepare_inputs`` boundary. The returned
        tensors are intentionally kept private to the backend; a future
        scheduler can replace the final ``generate`` call with cached vision
        features and a prefill adapter without changing the request protocol.
        """
        inputs = prepare_inputs(
            self.processor,
            images=images or None,
            videos=videos or None,
            video_metadata=video_metadata or None,
            prompts=prompt,
            image_token_index=getattr(self.model.config, "image_token_index", None),
            add_special_tokens=should_add_special_tokens(
                self.model.config.model_type, self.processor
            ),
        )
        input_ids = inputs.pop("input_ids")
        pixel_values = inputs.pop("pixel_values", None)
        attention_mask = inputs.pop("attention_mask", None)
        return PreparedVisionInputs(
            prompt=prompt,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            extra=inputs,
        )

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
            from mlx_vlm.speculative.drafters import (
                load_drafter,
                validate_drafter_compatibility,
            )
            from mlx_vlm.utils import load_model, load_tokenizer
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

        # The optimized quantized-argmax hook is not equivalent to argmax of
        # the target logits for M>1 on this MLX-VLM build. The device
        # diagnostic found identical verifier/replay hidden states and logits,
        # but different tokens from that fused argmax. Keep the semantic
        # reference (project logits, then argmax) as the default; faster
        # specialized variants remain opt-in for explicit A/B validation.
        original_argmax = target_lm.speculative_argmax_from_hidden
        reported_argmax_modes: set[str] = set()

        def positionwise_speculative_argmax(hidden):
            import mlx.core as mx

            mode = os.environ.get("MINIMLX_MTP_VERIFY_ARGMAX", "logits")
            if mode not in ("logits", "positionwise", "batched", "singleton_batch"):
                raise RuntimeError(
                    "MINIMLX_MTP_VERIFY_ARGMAX must be 'logits', 'positionwise', "
                    "'batched', or 'singleton_batch'"
                )
            if mode not in reported_argmax_modes:
                print(f"[mtp] verifier_argmax_hook=active mode={mode}")
                reported_argmax_modes.add(mode)
            if mode == "batched":
                return original_argmax(hidden)
            if mode == "logits":
                if hidden.ndim < 3 or hidden.shape[1] <= 1:
                    logits = target_lm.speculative_logits_from_hidden(hidden)
                    return mx.argmax(logits, axis=-1)
                # MLX-VLM's affine quantized head changes kernel strategy with
                # the token dimension. The M=4 projection can select a
                # different token from the identical M=1 row (observed at
                # verifier position 0), so project each verifier row with
                # singleton shape before argmax.
                return mx.concatenate(
                    [
                        mx.argmax(
                            target_lm.speculative_logits_from_hidden(
                                hidden[:, position : position + 1, :]
                            ),
                            axis=-1,
                        )
                        for position in range(hidden.shape[1])
                    ],
                    axis=1,
                )
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
        if target_lm.speculative_argmax_from_hidden is not positionwise_speculative_argmax:
            raise RuntimeError("Failed to install the Qwen3.5 MTP verifier argmax hook")

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
        # mlx-vlm's tokenizer loader uses pathlib operations on model_path.
        # ModelDescriptor stores paths as strings, so normalize at this API
        # boundary just as the VLM model loader does above.
        self.tokenizer = load_tokenizer(Path(descriptor.model_dir))
        self.descriptor = descriptor
        self._mtp_draft_model = drafter
        self._mtp_enabled = True
        self._mtp_target_language_model = target_lm
        self._mtp_argmax_hook = positionwise_speculative_argmax
        from .qwen35_overlay import Qwen35PerformanceOverlay

        self._optimization_overlay = Qwen35PerformanceOverlay(self)
        # Keep the first MTP path on one coherent MLX-VLM runtime.  The
        # MiniMLX native Q4/FA256 extensions are safe on the ordinary mlx-lm
        # route, but loading them into the MLX-VLM target can register a second
        # nanobind MLX device domain and abort the Python process.  Re-enable
        # them only for an explicit MTP compatibility experiment after the
        # stock correctness path is validated.
        if os.environ.get("MINIMLX_MTP_ENABLE_MINIMLX_KERNELS") == "1":
            from ..kernels.q4_native import (
                install_q4_native_mlp,
                install_q4_native_prefill,
            )
            from ..kernels.mtp_verify import install_mtp_verify_fused_attention
            from ..kernels.mtp_gdn_prework import install_mtp_gdn_prework
            from ..kernels.mtp_cache_commit import install_mtp_full_accept_cache_fastpath
            from ..kernels.gdn import install_mtp_vlm_gdn_prefill
            from ..kernels.fa256 import install_native_fa256_prefill

            install_q4_native_prefill(self.model, min_tokens=1024)
            install_q4_native_mlp(self.model, min_tokens=1024)
            install_mtp_vlm_gdn_prefill()
            install_native_fa256_prefill()
            install_mtp_verify_fused_attention()
            install_mtp_gdn_prework()
            install_mtp_full_accept_cache_fastpath()
        else:
            print(
                "[mtp] MiniMLX native Q4/FA256 kernels disabled; "
                "using coherent MLX-VLM target path",
                flush=True,
            )
        return self

    def _generate_mtp_greedy(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        prefill_step_size: int,
        on_token=None,
    ):
        """Greedy MTP using MLX-VLM's matching Qwen3.5 target transactions."""
        import mlx.core as mx

        from mlx_vlm.speculative.mtp import (
            _buffer_mtp_target_cache,
            _mtp_rounds,
        )
        generation_stream = mx.new_thread_local_stream(mx.default_device())

        if max_tokens <= 0:
            return prompt_tokens, [], "", 0.0, 0.0
        if not prompt_tokens:
            raise ValueError("MTP greedy path requires a non-empty prompt")

        lm = self.model.language_model
        if lm is not self._mtp_target_language_model:
            raise RuntimeError(
                "Loaded Qwen3.5 language model instance changed after installing "
                "the MTP verifier hook"
            )
        if lm.speculative_argmax_from_hidden is not self._mtp_argmax_hook:
            raise RuntimeError(
                "MTP verifier argmax hook was replaced after model load"
            )
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
        stream_detokenizer = None
        if on_token is not None:
            stream_detokenizer = self.tokenizer.detokenizer
            stream_detokenizer.reset()
            stream_detokenizer.add_token(first_token)
            on_token(stream_detokenizer.last_segment, [first_token])
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
                    verify_slice = verify_hidden[:, relative : relative + 1, :]
                    verifier_all_tokens = lm.speculative_argmax_from_hidden(verify_hidden)
                    verifier_token = verifier_all_tokens.reshape(-1)[relative]
                    # Preserve the same verifier hidden row and compare its
                    # ordinary LM-head projection with the specialized
                    # speculative projection. This distinguishes a forward/
                    # cache discrepancy from a quantized readout discrepancy.
                    speculative_logits = lm.speculative_logits_from_hidden(verify_slice)
                    speculative_logits_token = mx.argmax(
                        speculative_logits, axis=-1
                    )
                    ordinary_head = (
                        lm.model.embed_tokens.as_linear
                        if lm.args.tie_word_embeddings
                        else lm.lm_head
                    )
                    ordinary_logits = ordinary_head(verify_slice)
                    ordinary_logits_token = mx.argmax(ordinary_logits, axis=-1)

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
                            return_hidden=True,
                        )
                    if replay_output is None:
                        raise RuntimeError("MTP diagnostic replay prefix is empty")
                    replay_hidden = replay_output.hidden_states[-1][:, -1:, :]
                    replay_logits = replay_output.logits[:, -1:, :]
                    replay_token = mx.argmax(replay_logits, axis=-1)
                    mx.eval(
                        verifier_token,
                        speculative_logits_token,
                        ordinary_logits_token,
                        replay_token,
                        verify_slice,
                        replay_hidden,
                        speculative_logits,
                        ordinary_logits,
                        replay_logits,
                    )
                    hidden_max_abs = mx.max(mx.abs(verify_slice - replay_hidden))
                    speculative_vs_ordinary_max_abs = mx.max(
                        mx.abs(speculative_logits - ordinary_logits)
                    )
                    ordinary_vs_replay_max_abs = mx.max(
                        mx.abs(ordinary_logits - replay_logits)
                    )
                    mx.eval(
                        hidden_max_abs,
                        speculative_vs_ordinary_max_abs,
                        ordinary_vs_replay_max_abs,
                    )
                    print(
                        "[mtp-diagnostic] "
                        f"round={len(accept_lens)} "
                        f"generated_prefix_tokens={len(generated)} "
                        f"target_output_index={diagnostic_token if token_matches else len(generated)} "
                        f"verify_relative={relative} "
                        f"verify_input={verify_input[0].tolist()} "
                        f"verifier_next={int(verifier_token.reshape(-1).item())} "
                        f"speculative_logits_next={int(speculative_logits_token.reshape(-1).item())} "
                        f"ordinary_head_next={int(ordinary_logits_token.reshape(-1).item())} "
                        f"replay_next={int(replay_token.reshape(-1).item())} "
                        f"hidden_max_abs={float(hidden_max_abs.item()):.8g} "
                        f"speculative_vs_ordinary_logits_max_abs={float(speculative_vs_ordinary_max_abs.item()):.8g} "
                        f"ordinary_vs_replay_logits_max_abs={float(ordinary_vs_replay_max_abs.item()):.8g} "
                        f"replay_prefix_extra={same_round_prefix}"
                    )
                    lm._position_ids = saved_position_ids
                    lm._rope_deltas = saved_rope_deltas
                return result

            lm.speculative_verify_hidden = verify_hidden_with_diagnostic
        eos_ids = _token_ids_as_set(getattr(self.tokenizer, "eos_token_ids", ()))
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
                        if on_token is not None:
                            stream_detokenizer.add_token(token)
                            on_token(stream_detokenizer.last_segment, [token])
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
            except BaseException:
                _cleanup_failed_mtp_request(
                    mx,
                    generation_stream,
                    original_reset,
                    self.model,
                    cache,
                    shared_kv,
                )
                # A propagated traceback retains this frame. Drop its large
                # lazy-graph roots explicitly rather than pinning failed KV
                # state until the caller releases the exception.
                hidden = None
                output = None
                primer = None
                self._mtp_last_round_trace.clear()
                raise
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
        if stream_detokenizer is not None:
            stream_detokenizer.finalize()
            final_delta = stream_detokenizer.last_segment
            if final_delta:
                on_token(final_delta, [])
        return (
            prompt_tokens,
            generated,
            self.tokenizer.decode(generated),
            prefill_ms,
            decode_ms,
        )

    def create_plan(self, phase: str) -> ExecutionPlan:
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"unsupported phase: {phase}")
        return ExecutionPlan(phase=phase, use_custom_kernels=True)

    def clear_session(self, session_id: str) -> None:
        self._session_prefix_caches.pop(session_id, None)
        self._vlm_prompt_cache_states.pop(session_id, None)

    def generate(self, *args, **kwargs):
        if self._optimization_active and self._optimization_overlay is not None:
            return self._optimization_overlay.generate(*args, **kwargs)
        if (
            not self._mtp_enabled
            and not self._dflash_enabled
            and self._baseline_adapter is not None
        ):
            return self._baseline_adapter.generate(*args, **kwargs)
        if self._optimization_overlay is None:
            raise RuntimeError("Qwen3.5 generation overlay is unavailable")
        return self._optimization_overlay.generate(*args, **kwargs)

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 16,
        context_tokens: int | None = None,
        prefill_step_size: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_context_tokens: int | None = None,
        ignore_eos: bool = False,
    ):
        specialized_stream = (
            self._optimization_active or self._dflash_enabled or self._mtp_enabled
        )
        if self._baseline_adapter is None and not specialized_stream:
            raise RuntimeError("Qwen3.5 baseline adapter is unavailable")
        if not (
            self._optimization_overlay is not None
            and specialized_stream
        ):
            yield from self._baseline_adapter.stream_generate(
                prompt, max_tokens, context_tokens, prefill_step_size,
                temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_context_tokens=max_context_tokens, ignore_eos=ignore_eos,
            )
            return

        events: Queue = Queue()
        emitted_text: list[str] = []

        def send_token(text, token_ids):
            events.put(("delta", text, token_ids))

        def run_generation():
            original_eos_ids = None
            if ignore_eos:
                original_eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
                self.tokenizer.eos_token_ids = set()
            try:
                result = self._optimization_overlay.generate(
                    prompt, max_tokens, context_tokens, prefill_step_size,
                    temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    max_context_tokens=max_context_tokens,
                    on_token=send_token,
                )
                events.put(("done", result))
            except BaseException as exc:
                events.put(("error", exc))
            finally:
                if original_eos_ids is not None:
                    self.tokenizer.eos_token_ids = original_eos_ids

        worker = Thread(target=run_generation, name="minimlx-qwen-stream", daemon=True)
        worker.start()
        while True:
            event = events.get()
            if event[0] == "delta":
                _, text, token_ids = event
                if text:
                    emitted_text.append(text)
                yield {"event": "delta", "delta": text, "token_ids": token_ids}
            elif event[0] == "error":
                raise event[1]
            else:
                _, result = event
                prompt_tokens, generated, text, prefill_ms, decode_ms = result
                already_emitted = "".join(emitted_text)
                if text.startswith(already_emitted) and len(text) > len(already_emitted):
                    yield {
                        "event": "delta",
                        "delta": text[len(already_emitted):],
                        "token_ids": [],
                    }
                yield {
                    "event": "done",
                    "text": text,
                    "usage": {
                        "prompt_tokens": len(prompt_tokens),
                        "completion_tokens": len(generated),
                    },
                    "metrics": {"prefill_ms": prefill_ms, "decode_ms": decode_ms},
                }
                break

    def profile_prefill(self, *args, **kwargs):
        if self._optimization_overlay is None:
            raise RuntimeError("Qwen3.5 optimization overlay is unavailable")
        return self._optimization_overlay.profile_prefill(*args, **kwargs)

    def profile_projection_fusion(self, *args, **kwargs):
        if self._optimization_overlay is None:
            raise RuntimeError("Qwen3.5 optimization overlay is unavailable")
        return self._optimization_overlay.profile_projection_fusion(*args, **kwargs)
