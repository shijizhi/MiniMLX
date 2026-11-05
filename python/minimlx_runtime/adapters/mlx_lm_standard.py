from __future__ import annotations

import json
import importlib.metadata
import os
import time
from pathlib import Path
from typing import Any

from ..model_catalog import STANDARD_MLX_LM_TEXT_FAMILIES
from ..api.memory import read_system_memory
from ..offload import Qwen38OffloadSession, qwen38_should_use_ssd_offload
from ..types import Acceleration, ExecutionPlan, ModelDescriptor


class StandardTextAdapter:
    """Shared baseline for text families implemented by upstream mlx-lm."""

    name = "mlx-lm-standard"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.descriptor: ModelDescriptor | None = None
        self._dflash_draft_model: Any | None = None
        self.offload_session: Qwen38OffloadSession | None = None

    @property
    def _dflash_enabled(self) -> bool:
        return self._dflash_draft_model is not None

    @staticmethod
    def can_load(descriptor: ModelDescriptor) -> bool:
        model_type = descriptor.model_type.lower()
        architecture = descriptor.architecture.lower()
        if model_type not in STANDARD_MLX_LM_TEXT_FAMILIES:
            return False
        causal_lm = architecture.endswith("forcausallm")
        qwen35_text = (
            model_type in {"qwen3_5", "qwen3_5_moe"}
            and architecture.endswith("forconditionalgeneration")
        )
        if not causal_lm and not qwen35_text:
            return False

        # mlx-lm applies a small set of documented model_type aliases before
        # importing a model module. Mirror those aliases for family matching;
        # the actual loader remains the authority for checkpoint compatibility.
        aliases = {"mistral": "llama", "iquestcoder": "llama"}
        module_name = aliases.get(model_type, model_type)
        try:
            distribution = importlib.metadata.distribution("mlx-lm")
            return any(
                str(path).replace("\\", "/")
                == f"mlx_lm/models/{module_name}.py"
                for path in distribution.files or ()
            )
        except importlib.metadata.PackageNotFoundError:
            return False

    def load(
        self, descriptor: ModelDescriptor, plan: ExecutionPlan | None = None
    ) -> "StandardTextAdapter":
        try:
            from mlx_lm import load
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "mlx-lm is not installed in MiniMLX's .venv; "
                "run tools/bootstrap_venv.sh"
            ) from exc
        tokenizer_config = (
            {"fix_mistral_regex": True}
            if descriptor.model_type.lower() == "mistral"
            else None
        )
        is_qwen38 = bool(descriptor.metadata.get("is_qwen38_flash_next"))
        offload_mode = os.environ.get("MINIMLX_QWEN38_OFFLOAD", "auto").lower()
        if offload_mode not in {"auto", "0", "1", "true", "false"}:
            raise ValueError("MINIMLX_QWEN38_OFFLOAD must be auto, 0, or 1")
        snapshot = read_system_memory() if is_qwen38 else None
        auto_offload = is_qwen38 and qwen38_should_use_ssd_offload(
            int(descriptor.metadata.get("total_weight_bytes") or 0),
            total_memory_bytes=snapshot.total_bytes if snapshot else None,
            available_memory_bytes=snapshot.available_bytes if snapshot else None,
        )
        offload_enabled = is_qwen38 and (
            offload_mode in {"1", "true"}
            or (offload_mode == "auto" and auto_offload)
        )
        load_options: dict[str, Any] = {"tokenizer_config": tokenizer_config}
        if offload_enabled:
            try:
                cache_bytes = max(
                    0,
                    int(
                        os.environ.get(
                            "MINIMLX_QWEN38_EXPERT_CACHE_BYTES",
                            str(512 * 1024 * 1024),
                        )
                    ),
                )
                self.offload_session = Qwen38OffloadSession.open(
                    descriptor.model_dir,
                    cache_bytes=cache_bytes,
                    prefetch_batch_size=max(
                        1,
                        int(os.environ.get("MINIMLX_QWEN38_PREFETCH_BATCH", "4")),
                    ),
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "Qwen3.8-Flash-Next offload index could not be prepared: "
                    f"{exc}"
                ) from exc
            # mlx-lm's lazy loader keeps checkpoint arrays unevaluated until
            # execution. The custom loader removes routed expert tensors from
            # the initial weight set and replaces SwitchGLU with the lazy
            # expert implementation.
            if os.environ.get("MINIMLX_QWEN38_CUSTOM_EXECUTOR", "1") == "1":
                from ..qwen38_executor import load_qwen38_offload_model

                self.model, self.tokenizer = load_qwen38_offload_model(
                    descriptor.model_dir,
                    self.offload_session,
                    tokenizer_config=tokenizer_config,
                )
            else:
                load_options["lazy"] = True
                self.model, self.tokenizer = load(
                    descriptor.model_dir, **load_options
                )
        else:
            self.model, self.tokenizer = load(
                descriptor.model_dir, **load_options
            )
        self.descriptor = descriptor
        draft_path = (
            plan.acceleration_components.get(Acceleration.DFLASH)
            if plan is not None and Acceleration.DFLASH in plan.accelerations
            else None
        )
        if draft_path:
            try:
                from dflash import model_mlx
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError("DFlash acceleration requires the 'dflash' package") from exc
            from .qwen35 import _DFLASH_DRAFT_LOAD_LOCK

            with _DFLASH_DRAFT_LOAD_LOCK:
                original_snapshot_download = model_mlx.snapshot_download
                model_mlx.snapshot_download = lambda _repo, **_kwargs: str(
                    Path(draft_path).expanduser().resolve()
                )
                try:
                    self._dflash_draft_model = model_mlx.load_draft(draft_path)
                finally:
                    model_mlx.snapshot_download = original_snapshot_download
        return self

    def unload(self) -> None:
        self.tokenizer = None
        self.model = None
        self.descriptor = None
        self._dflash_draft_model = None
        self.offload_session = None

    def clear_session(self, _session_id: str) -> None:
        # The baseline MLX-LM path does not retain per-session KV caches.
        return None

    def format_messages(self, messages, chat_template_kwargs: dict | None = None) -> str | None:
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            return None
        formatted_messages = [
            {
                "role": message.role,
                "content": "\n".join(
                    part.text or ""
                    for part in message.content
                    if part.modality.value == "text"
                ),
            }
            for message in messages
        ]
        template_kwargs = {
            **(chat_template_kwargs or {}),
            "tokenize": False,
            "add_generation_prompt": True,
        }
        return apply_template(formatted_messages, **template_kwargs)

    def generate(
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
    ):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        from mlx_lm import stream_generate

        encode = getattr(self.tokenizer, "encode", None)
        prompt_tokens = list(encode(prompt, add_special_tokens=False)) if encode else []
        generation_prompt: str | list[int] = prompt
        if context_tokens is not None:
            if context_tokens <= 0:
                raise ValueError("context_tokens must be positive")
            prompt_tokens = prompt_tokens[-context_tokens:]
            generation_prompt = prompt_tokens
        if max_context_tokens is not None:
            if max_context_tokens <= 0:
                raise ValueError("max_context_tokens must be positive")
            prompt_tokens = prompt_tokens[-max_context_tokens:]
            generation_prompt = prompt_tokens

        if self._dflash_draft_model is not None:
            if min_p != 0.0 or repetition_penalty != 1.0 or presence_penalty != 0.0 or frequency_penalty != 0.0:
                raise ValueError("DFlash supports temperature, top-p, and top-k sampling only")
            from dflash.model_mlx import stream_generate as dflash_stream_generate

            block_size = min(
                int(getattr(getattr(self._dflash_draft_model, "config", None), "block_size", 5)),
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
                    self.model, self._dflash_draft_model, self.tokenizer, generation_prompt,
                    block_size=block_size, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    prefill_step_size=prefill_step_size,
                )
                text_parts: list[str] = []
                output_tokens: list[int] = []
                prefill_ms = 0.0
                decode_tps = 0.0
                for response in responses:
                    text_parts.append(response.text)
                    output_tokens.extend(int(token) for token in response.tokens)
                    if response.prompt_tokens and response.prompt_tps > 0:
                        prefill_ms = response.prompt_tokens * 1000.0 / response.prompt_tps
                    decode_tps = response.generation_tps
            finally:
                if original_eos_ids is not None:
                    self.tokenizer.eos_token_ids = original_eos_ids
            decode_ms = len(output_tokens) * 1000.0 / decode_tps if decode_tps else 0.0
            return prompt_tokens, output_tokens, "".join(text_parts), prefill_ms, decode_ms

        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "prefill_step_size": prefill_step_size,
        }
        if (
            temperature != 0.0
            or min_p != 0.0
            or repetition_penalty != 1.0
            or presence_penalty != 0.0
            or frequency_penalty != 0.0
        ):
            from mlx_lm.sample_utils import make_logits_processors, make_sampler

            kwargs["sampler"] = make_sampler(
                temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p
            )
            processors = make_logits_processors(
                repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
                presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
                frequency_penalty=frequency_penalty if frequency_penalty != 0.0 else None,
            )
            if processors:
                kwargs["logits_processors"] = processors

        text: list[str] = []
        generated_tokens: list[int] = []
        prefill_ms = 0.0
        decode_tps = 0.0
        started = time.perf_counter()
        for response in stream_generate(
            self.model, self.tokenizer, generation_prompt, **kwargs
        ):
            text.append(response.text)
            generated_tokens.append(int(response.token))
            prompt_tps = float(response.prompt_tps or 0.0)
            if prompt_tps > 0:
                prefill_ms = int(response.prompt_tokens) * 1000.0 / prompt_tps
            decode_tps = float(response.generation_tps or 0.0)
        decode_ms = (
            len(generated_tokens) * 1000.0 / decode_tps
            if decode_tps > 0
            else max(0.0, (time.perf_counter() - started) * 1000.0 - prefill_ms)
        )
        return prompt_tokens, generated_tokens, "".join(text), prefill_ms, decode_ms

    def benchmark_generate_tokens(
        self, prompt_tokens: list[int], *, max_tokens: int, prefill_step_size: int
    ):
        """Benchmark exact token prompts, matching oMLX's direct-token path."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        from ..types import GenerationResult
        if self._dflash_draft_model is not None:
            from dflash.model_mlx import stream_generate

            block_size = min(
                int(getattr(getattr(self._dflash_draft_model, "config", None), "block_size", 5)),
                int(os.environ.get("MINIMLX_DFLASH_BLOCK_SIZE", "5")),
            )
            generation_kwargs: dict[str, Any] = {
                "block_size": block_size,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "prefill_step_size": prefill_step_size,
            }
            generation_args = (self.model, self._dflash_draft_model, self.tokenizer)
        else:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            generation_kwargs = {
                "max_tokens": max_tokens,
                "prefill_step_size": prefill_step_size,
                "sampler": make_sampler(temp=0.0, top_p=1.0),
            }
            generation_args = (self.model, self.tokenizer)
        # Match oMLX's single-request benchmark fallback metrics. In the
        # batched engine used by oMLX, prompt_tps/generation_tps are unset;
        # PP_TPS therefore uses prompt_tokens / TTFT, while decode throughput
        # uses the interval from the first generated token to the last one.
        started = time.perf_counter()
        first_token_ms: float | None = None
        first_generated_at: float | None = None
        last_generated_at: float | None = None
        generated: list[int] = []
        text: list[str] = []
        original_eos_ids = None
        if os.environ.get("MINIMLX_IGNORE_EOS") == "1":
            original_eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
            self.tokenizer.eos_token_ids = set()
        try:
            for response in stream_generate(
                *generation_args, prompt_tokens, **generation_kwargs
            ):
                observed = time.perf_counter()
                response_tokens = (
                    response.tokens
                    if hasattr(response, "tokens")
                    else (response.token,)
                )
                response_tokens = [int(token) for token in response_tokens]
                if response_tokens:
                    if first_generated_at is None:
                        first_generated_at = observed
                        first_token_ms = (observed - started) * 1000.0
                    last_generated_at = observed
                generated.extend(response_tokens)
                text.append(response.text)
        finally:
            if original_eos_ids is not None:
                self.tokenizer.eos_token_ids = original_eos_ids
        # Match oMLX's batched-engine fallback metrics: its engine does not
        # provide prompt_tps/generation_tps for this path, so PP_TPS uses
        # TTFT and decode throughput uses first-to-last token timestamps.
        prefill_ms = first_token_ms or 0.0
        decode_ms = (
            max(0.0, last_generated_at - first_generated_at) * 1000.0
            if first_generated_at is not None and last_generated_at is not None
            else 0.0
        )
        return GenerationResult(
            prompt_tokens=list(prompt_tokens),
            generated_tokens=generated,
            text="".join(text),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            ttft_ms=first_token_ms,
        )

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
        """Yield decoded text as MLX generates tokens, followed by exact usage."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        if self._dflash_draft_model is not None:
            from dflash.model_mlx import stream_generate as generate

            generation_kwargs: dict[str, Any] = {
                "block_size": min(
                    int(getattr(getattr(self._dflash_draft_model, "config", None), "block_size", 5)),
                    int(os.environ.get("MINIMLX_DFLASH_BLOCK_SIZE", "5")),
                ),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "prefill_step_size": prefill_step_size,
            }
            target = self._dflash_draft_model
        else:
            from mlx_lm import stream_generate as generate

            generation_kwargs = {
                "max_tokens": max_tokens,
                "prefill_step_size": prefill_step_size,
            }
            target = None
        encode = getattr(self.tokenizer, "encode", None)
        prompt_tokens = list(encode(prompt, add_special_tokens=False)) if encode else []
        generation_prompt: str | list[int] = prompt
        for limit in (context_tokens, max_context_tokens):
            if limit is not None:
                if limit <= 0:
                    raise ValueError("context limits must be positive")
                prompt_tokens = prompt_tokens[-limit:]
                generation_prompt = prompt_tokens
        if target is not None:
            if min_p != 0.0 or repetition_penalty != 1.0 or presence_penalty != 0.0 or frequency_penalty != 0.0:
                raise ValueError("DFlash supports temperature, top-p, and top-k sampling only")
            if generation_kwargs["block_size"] < 2:
                raise ValueError("MINIMLX_DFLASH_BLOCK_SIZE must be at least 2")
            responses = generate(
                self.model, target, self.tokenizer, generation_prompt, **generation_kwargs
            )
        else:
            if (
                temperature != 0.0 or min_p != 0.0 or repetition_penalty != 1.0
                or presence_penalty != 0.0 or frequency_penalty != 0.0
            ):
                from mlx_lm.sample_utils import make_logits_processors, make_sampler

                generation_kwargs["sampler"] = make_sampler(
                    temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p
                )
                processors = make_logits_processors(
                    repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
                    presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
                    frequency_penalty=frequency_penalty if frequency_penalty != 0.0 else None,
                )
                if processors:
                    generation_kwargs["logits_processors"] = processors
            responses = generate(self.model, self.tokenizer, generation_prompt, **generation_kwargs)
        generated_tokens: list[int] = []
        text_parts: list[str] = []
        prefill_ms = decode_ms = 0.0
        ignore_eos = ignore_eos or os.environ.get("MINIMLX_IGNORE_EOS") == "1"
        original_eos_ids = None
        if ignore_eos:
            original_eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
            self.tokenizer.eos_token_ids = set()
        try:
            for response in responses:
                ids = (
                    [int(token) for token in response.tokens]
                    if hasattr(response, "tokens")
                    else [int(response.token)]
                )
                generated_tokens.extend(ids)
                text_parts.append(response.text)
                prompt_tps = float(getattr(response, "prompt_tps", 0.0) or 0.0)
                prompt_count = int(getattr(response, "prompt_tokens", 0) or 0)
                if prompt_tps > 0:
                    prefill_ms = prompt_count * 1000.0 / prompt_tps
                decode_tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
                if decode_tps > 0:
                    decode_ms = len(generated_tokens) * 1000.0 / decode_tps
                if response.text or ids:
                    yield {"event": "delta", "delta": response.text, "token_ids": ids}
        finally:
            if original_eos_ids is not None:
                self.tokenizer.eos_token_ids = original_eos_ids
        yield {
            "event": "done",
            "text": "".join(text_parts),
            "usage": {"prompt_tokens": len(prompt_tokens), "completion_tokens": len(generated_tokens)},
            "metrics": {"prefill_ms": prefill_ms, "decode_ms": decode_ms},
        }
