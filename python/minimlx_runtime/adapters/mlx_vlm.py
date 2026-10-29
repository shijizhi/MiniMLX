"""Family-agnostic generation adapter for checkpoints supported by mlx-vlm."""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from ..types import ComponentKind, ExecutionPlan, ModelDescriptor


def _image_input(part: Any) -> Any:
    """Validate a transport image and convert it to an mlx-vlm input value."""
    max_bytes = max(1, int(os.environ.get("MINIMLX_MAX_IMAGE_BYTES", str(16 * 1024 * 1024))))
    max_pixels = max(1, int(os.environ.get("MINIMLX_MAX_IMAGE_PIXELS", str(40 * 1000 * 1000))))

    def decode(raw: bytes):
        if len(raw) > max_bytes:
            raise ValueError(f"image exceeds MINIMLX_MAX_IMAGE_BYTES ({max_bytes})")
        from PIL import Image

        image = Image.open(BytesIO(raw))
        image.load()
        if image.width * image.height > max_pixels:
            raise ValueError(f"image exceeds MINIMLX_MAX_IMAGE_PIXELS ({max_pixels})")
        return image

    def validate_file(path: Path) -> None:
        try:
            if path.stat().st_size > max_bytes:
                raise ValueError(f"image exceeds MINIMLX_MAX_IMAGE_BYTES ({max_bytes})")
            from PIL import Image

            with Image.open(path) as image:
                image.load()
                if image.width * image.height > max_pixels:
                    raise ValueError(f"image exceeds MINIMLX_MAX_IMAGE_PIXELS ({max_pixels})")
        except OSError as exc:
            raise ValueError(f"cannot read image file: {path}") from exc

    if part.uri is not None:
        from urllib.parse import unquote, urlparse

        parsed = urlparse(part.uri)
        if parsed.scheme == "data" and ";base64" in parsed.path:
            try:
                return decode(base64.b64decode(parsed.path.split(",", 1)[1], validate=True))
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("invalid base64 data URI image") from exc
        if parsed.scheme == "file":
            path = Path(unquote(parsed.path))
            validate_file(path)
            return str(path)
        if parsed.scheme in {"http", "https"} and os.environ.get("MINIMLX_ALLOW_REMOTE_IMAGES") != "1":
            raise ValueError("remote images are disabled; use a local URI or base64 payload")
        return part.uri
    if part.data_base64 is not None:
        try:
            return decode(base64.b64decode(part.data_base64, validate=True))
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("invalid base64 image payload") from exc
    raise ValueError("image content requires a URI or base64 payload")


def _video_input(part: Any) -> tuple[Any, dict[str, Any] | None, str | None]:
    """Convert a video reference or timestamped frames to mlx-vlm inputs."""
    if part.frames is None:
        if part.uri is not None:
            from urllib.parse import unquote, urlparse

            parsed = urlparse(part.uri)
            if parsed.scheme in ("", "file"):
                path = unquote(parsed.path) if parsed.scheme == "file" else part.uri
                if not Path(path).is_file():
                    raise ValueError(f"video file does not exist: {path}")
                max_bytes = max(1, int(os.environ.get("MINIMLX_MAX_VIDEO_BYTES", str(128 * 1024 * 1024))))
                if Path(path).stat().st_size > max_bytes:
                    raise ValueError(f"video exceeds MINIMLX_MAX_VIDEO_BYTES ({max_bytes})")
                return str(path), None, None
            raise ValueError("remote videos are disabled; use a local URI or base64 payload")
        raw = base64.b64decode(part.data_base64 or "", validate=True)
        max_bytes = max(1, int(os.environ.get("MINIMLX_MAX_VIDEO_BYTES", str(128 * 1024 * 1024))))
        if len(raw) > max_bytes:
            raise ValueError(f"video exceeds MINIMLX_MAX_VIDEO_BYTES ({max_bytes})")
        suffix = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm"}.get(
            part.mime_type, ".mp4"
        )
        with tempfile.NamedTemporaryFile(prefix="minimlx-video-", suffix=suffix, delete=False) as stream:
            stream.write(raw)
            path = stream.name
        return path, None, path

    import numpy as np

    frames = [_image_input(frame).convert("RGB") for frame in part.frames]
    max_pixels = max(1, int(os.environ.get("MINIMLX_MAX_IMAGE_PIXELS", str(40 * 1000 * 1000))))
    if any(frame.width * frame.height > max_pixels for frame in frames):
        raise ValueError(f"video frame exceeds MINIMLX_MAX_IMAGE_PIXELS ({max_pixels})")
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    normalized = []
    for frame in frames:
        if frame.size != (width, height):
            canvas = frame.__class__.new("RGB", (width, height))
            canvas.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
            frame = canvas
        normalized.append(np.asarray(frame))
    timestamps = [float(frame.timestamp_seconds) for frame in part.frames]
    if timestamps != sorted(timestamps):
        raise ValueError("video frame timestamps must be in chronological order")
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:]) if right > left]
    fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 and timestamps[-1] > timestamps[0] else 1.0
    indices = [round(timestamp * fps) for timestamp in timestamps]
    metadata = {
        "total_num_frames": max(indices, default=0) + 1,
        "fps": fps or 1.0,
        "frames_indices": indices,
        "width": width,
        "height": height,
        "duration": timestamps[-1] + (min(gaps) if gaps else 1.0),
    }
    return np.stack(normalized), metadata, None


def _apply_vlm_chat_template(processor, config, messages, num_images, video_parts, template_kwargs):
    """Keep native typed media in the processor template when video is present."""
    if video_parts and callable(getattr(processor, "apply_chat_template", None)):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(template_kwargs or {}),
        )
    from mlx_vlm.prompt_utils import apply_chat_template

    return apply_chat_template(
        processor,
        config,
        messages,
        num_images=num_images,
        **(template_kwargs or {}),
    )


def _phase_timings(result: Any, prompt_tokens: int, elapsed_ms: float) -> tuple[float, float]:
    generated = int(getattr(result, "generation_tokens", 0) or 0)
    prompt_tps = float(getattr(result, "prompt_tps", 0.0) or 0.0)
    generation_tps = float(getattr(result, "generation_tps", 0.0) or 0.0)
    prefill_ms = prompt_tokens * 1000.0 / prompt_tps if prompt_tps > 0 else elapsed_ms
    decode_ms = (
        generated * 1000.0 / generation_tps
        if generation_tps > 0
        else max(0.0, elapsed_ms - prefill_ms)
    )
    return prefill_ms, decode_ms


class MLXVLMAdapter:
    """Use mlx-vlm's public loader/generator for any complete VLM checkpoint.

    The adapter does not classify model families itself. mlx-vlm owns the
    architecture registry, weight mapping, processor, and family-specific
    image embedding path; MiniMLX owns transport validation and lifecycle.
    """

    name = "mlx-vlm-model"
    required_components = frozenset(
        {
            ComponentKind.TOKENIZER,
            ComponentKind.TEXT_WEIGHTS,
            ComponentKind.PROCESSOR,
            ComponentKind.VISION_ENCODER,
        }
    )

    _image_input = staticmethod(_image_input)

    def __init__(self) -> None:
        self.model: Any | None = None
        self.processor: Any | None = None
        self.tokenizer: Any | None = None
        self.descriptor: ModelDescriptor | None = None
        self._prompt_cache_states: dict[str, Any] = {}
        self._apc_manager: Any | None = None
        self._vlm_enabled = False

    @classmethod
    def can_load(cls, descriptor: ModelDescriptor) -> bool:
        # Route by declared modality; the resolver separately explains missing
        # checkpoint components instead of hiding the model as an unknown family.
        return bool(descriptor.metadata.get("has_vision"))

    def load(
        self, descriptor: ModelDescriptor, plan: ExecutionPlan | None = None
    ) -> "MLXVLMAdapter":
        if plan is not None and plan.backend_adapter not in {
            "mlx-vlm-runtime",
            "qwen35-vlm",
            "deepseek41-mlx-vlm-runtime",
        }:
            raise ValueError(f"MLXVLMAdapter cannot load backend {plan.backend_adapter!r}")
        try:
            from mlx_vlm.utils import load
        except ModuleNotFoundError as exc:
            raise RuntimeError("VLM inference requires mlx-vlm in MiniMLX's Python environment") from exc
        except ImportError as exc:
            raise RuntimeError("mlx-vlm could not access the Apple Metal device") from exc

        load_options: dict[str, Any] = {}
        if plan is not None and plan.backend_adapter == "deepseek41-mlx-vlm-runtime":
            # mlx-vlm's offload directory activates its native routed-expert
            # pager. Keep loading lazy and bound the resident expert cache.
            load_options = {
                "lazy": True,
                "expert_cache_gb": max(
                    0.0,
                    float(os.environ.get("MINIMLX_DEEPSEEK41_EXPERT_CACHE_GB", "8")),
                ),
            }
        try:
            self.model, self.processor = load(descriptor.model_dir, **load_options)
        except (ImportError, ModuleNotFoundError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"mlx-vlm could not load this checkpoint ({descriptor.model_type}/"
                f"{descriptor.architecture}): {exc}"
            ) from exc
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.descriptor = descriptor
        self._apc_manager = self._create_apc_manager(descriptor)
        self._vlm_enabled = True
        if (
            os.environ.get("MINIMLX_GDN_DECODE_CUSTOM_RECURRENCE") == "1"
            or os.environ.get("MINIMLX_GDN_DECODE_FUSED_QK_NORM") == "1"
        ):
            from ..kernels.gdn_decode import install_gdn_decode_custom_recurrence

            if not install_gdn_decode_custom_recurrence():
                raise RuntimeError("GDN recurrence experiment could not install its dispatch hooks")
        return self

    @staticmethod
    def _create_apc_manager(descriptor: ModelDescriptor) -> Any | None:
        """Opt into mlx-vlm's shared prefix cache without changing its defaults."""
        if os.environ.get("MINIMLX_ENABLE_VLM_APC", "0") != "1":
            return None
        try:
            from mlx_vlm.apc import from_env

            model_identity = hashlib.sha256(
                os.path.realpath(descriptor.model_dir).encode("utf-8")
            ).hexdigest()[:24]
            memory_gb = max(
                0.0,
                float(os.environ.get("MINIMLX_VLM_APC_MEMORY_GB", "1.0")),
            )
            # Keep this first experiment model-local and memory-bounded. Disk
            # persistence needs a checkpoint fingerprint/lifecycle policy and
            # is deliberately not enabled by this adapter yet.
            return from_env(
                model_namespace=f"minimlx-{model_identity}",
                overrides={
                    "enabled": True,
                    "disk_enabled": False,
                    "memory_max_gb": memory_gb,
                },
            )
        except Exception as exc:
            raise RuntimeError(f"could not initialize optional mlx-vlm APC: {exc}") from exc

    def unload(self) -> None:
        self._prompt_cache_states.clear()
        if self._apc_manager is not None:
            self._apc_manager.clear()
            self._apc_manager.close()
            self._apc_manager = None
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.descriptor = None
        self._vlm_enabled = False
        import sys

        mx = sys.modules.get("mlx.core")
        if mx is not None:
            mx.clear_cache()

    def clear_session(self, session_id: str) -> None:
        self._prompt_cache_states.pop(session_id, None)

    def vision_feature_cache_metrics(self) -> dict[str, int]:
        # Generic feature injection is intentionally delegated to each
        # mlx-vlm model implementation; no cross-family cache is assumed.
        return {}

    def inference_cache_metrics(self) -> dict[str, int]:
        """Expose upstream prefix-cache counters for paired runtime tests."""
        if self._apc_manager is None:
            return {}
        snapshot = self._apc_manager.stats_snapshot()
        return {
            # mlx-vlm exposes these counters as lookups_hit/lookups_miss.
            # Keep the shorter aliases for compatibility with APC versions
            # that expose the dataclass field names directly.
            "prefix_cache_hits": int(snapshot.get("lookups_hit", snapshot.get("hits", 0))),
            "prefix_cache_misses": int(snapshot.get("lookups_miss", snapshot.get("misses", 0))),
            "prefix_cache_matched_tokens": int(snapshot.get("matched_tokens", 0)),
            "prefix_cache_resident_bytes": int(snapshot.get("resident_bytes", 0)),
        }

    def _prepare(
        self,
        messages: list[dict],
        image_parts: list[Any],
        video_parts: list[Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        if self.model is None or self.processor is None:
            raise RuntimeError("vision-language model is not loaded")
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        images = [_image_input(part) for part in image_parts]
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
                messages,
                len(images),
                video_parts,
                chat_template_kwargs,
            )
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
        finally:
            for path in cleanup_paths:
                Path(path).unlink(missing_ok=True)
        input_ids = inputs.pop("input_ids")
        pixel_values = inputs.pop("pixel_values", None)
        attention_mask = inputs.pop("attention_mask", None)
        return prompt, images, input_ids, pixel_values, attention_mask, inputs

    def _prepare_request(self, messages, image_parts, video_parts, chat_template_kwargs):
        """Call overrides that predate optional VLM media/template arguments."""
        import inspect

        try:
            parameters = inspect.signature(self._prepare).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            parameters, accepts_kwargs = {}, True
        optional = {}
        if accepts_kwargs or "video_parts" in parameters:
            optional["video_parts"] = video_parts
        if accepts_kwargs or "chat_template_kwargs" in parameters:
            optional["chat_template_kwargs"] = chat_template_kwargs
        return self._prepare(messages, image_parts, **optional)

    def _prepare_raw_text_request(self, messages):
        """Prepare a raw completion prompt without applying a chat template."""
        if self.model is None or self.processor is None:
            raise RuntimeError("VLM model is not loaded")
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        text_parts = []
        for message in messages:
            for part in message.get("content", ()):
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
        prompt = "\n".join(text_parts)
        inputs = prepare_inputs(
            self.processor,
            images=None,
            videos=None,
            prompts=prompt,
            image_token_index=getattr(self.model.config, "image_token_index", None),
            add_special_tokens=should_add_special_tokens(
                self.model.config.model_type, self.processor
            ),
        )
        input_ids = inputs.pop("input_ids")
        pixel_values = inputs.pop("pixel_values", None)
        attention_mask = inputs.pop("attention_mask", None)
        return prompt, [], input_ids, pixel_values, attention_mask, inputs

    def _multimodal_generation_extras(
        self, images: list[Any], extra: dict[str, Any]
    ) -> dict[str, Any]:
        """Backend/family hook for optional optimized multimodal state."""
        return extra

    @staticmethod
    def _generation_options(
        max_tokens: int,
        prefill_step_size: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> dict[str, Any]:
        return {
            "max_tokens": max_tokens,
            "prefill_step_size": prefill_step_size,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": None if repetition_penalty == 1.0 else repetition_penalty,
            "presence_penalty": None if presence_penalty == 0.0 else presence_penalty,
            "frequency_penalty": None if frequency_penalty == 0.0 else frequency_penalty,
            "verbose": False,
        }

    def generate_multimodal(
        self,
        messages: list[dict],
        image_parts: list[Any],
        max_tokens: int = 16,
        prefill_step_size: int = 2048,
        session_id: str | None = None,
        context_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_context_tokens: int | None = None,
        video_parts: list[Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        if self.model is None or self.processor is None or not self._vlm_enabled:
            raise RuntimeError("vision-language model is not loaded")
        from mlx_vlm.generate import PromptCacheState, generate

        if os.environ.get("MINIMLX_RAW_COMPLETION") == "1" and not image_parts and not video_parts:
            prompt, images, input_ids, pixel_values, attention_mask, extra = (
                self._prepare_raw_text_request(messages)
            )
        else:
            prompt, images, input_ids, pixel_values, attention_mask, extra = self._prepare_request(
                messages, image_parts, video_parts, chat_template_kwargs
            )
        if context_tokens is not None:
            if images or video_parts:
                raise ValueError("fixed context_tokens is only supported for text-only VLM requests")
            if context_tokens <= 0 or input_ids.shape[-1] < context_tokens:
                raise ValueError("text prompt is shorter than the requested fixed context")
            input_ids = input_ids[..., :context_tokens]
            if attention_mask is not None:
                attention_mask = attention_mask[..., :context_tokens]
        if max_context_tokens is not None and not images and not video_parts:
            if max_context_tokens <= 0:
                raise ValueError("max_context_tokens must be positive")
            input_ids = input_ids[..., -max_context_tokens:]
            if attention_mask is not None:
                attention_mask = attention_mask[..., -max_context_tokens:]

        kwargs = self._generation_options(
            max_tokens,
            prefill_step_size,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
        )
        if os.environ.get("MINIMLX_IGNORE_EOS") == "1":
            # mlx-vlm owns EOS handling through tokenizer.stopping_criteria;
            # changing mlx-lm's tokenizer.eos_token_ids is not sufficient for
            # multimodal generation through this backend.
            kwargs["stopping_criteria"] = lambda _token: False
        kwargs.update(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mask=attention_mask,
            **self._multimodal_generation_extras(images, extra),
        )
        if self._apc_manager is not None:
            kwargs["apc_manager"] = self._apc_manager
        if session_id is not None:
            kwargs["prompt_cache_state"] = self._prompt_cache_states.setdefault(
                session_id, PromptCacheState()
            )

        started = time.perf_counter()
        result = generate(self.model, self.processor, prompt, image=images or None, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        prompt_ids = input_ids.reshape(-1).tolist()
        prefill_ms, decode_ms = _phase_timings(result, len(prompt_ids), elapsed_ms)
        return (
            prompt_ids,
            list(getattr(result, "token_ids", None) or []),
            str(getattr(result, "text", "")),
            prefill_ms,
            decode_ms,
        )

    def stream_multimodal(
        self,
        messages: list[dict],
        image_parts: list[Any],
        max_tokens: int = 16,
        prefill_step_size: int = 2048,
        session_id: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        video_parts: list[Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        if self.model is None or self.processor is None or not self._vlm_enabled:
            raise RuntimeError("vision-language model is not loaded")
        from mlx_vlm.generate import PromptCacheState, stream_generate

        if os.environ.get("MINIMLX_RAW_COMPLETION") == "1" and not image_parts and not video_parts:
            prompt, images, input_ids, pixel_values, attention_mask, extra = (
                self._prepare_raw_text_request(messages)
            )
        else:
            prompt, images, input_ids, pixel_values, attention_mask, extra = self._prepare_request(
                messages, image_parts, video_parts, chat_template_kwargs
            )
        kwargs = self._generation_options(
            max_tokens,
            prefill_step_size,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
        )
        if os.environ.get("MINIMLX_IGNORE_EOS") == "1":
            # Keep text-only VLM-backend benchmarks on the same fixed-length
            # decode protocol as the mlx-lm path.
            kwargs["stopping_criteria"] = lambda _token: False
        kwargs.update(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mask=attention_mask,
            **self._multimodal_generation_extras(images, extra),
        )
        if self._apc_manager is not None:
            kwargs["apc_manager"] = self._apc_manager
        if session_id is not None:
            kwargs["prompt_cache_state"] = self._prompt_cache_states.setdefault(
                session_id, PromptCacheState()
            )
        for result in stream_generate(
            self.model, self.processor, prompt, image=images or None, **kwargs
        ):
            text = str(getattr(result, "text", "") or "")
            token_ids = list(getattr(result, "token_ids", None) or [])
            if text or token_ids:
                yield {
                    "text": text,
                    "token_ids": token_ids,
                    "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
                    "is_draft": bool(getattr(result, "is_draft", False)),
                }
