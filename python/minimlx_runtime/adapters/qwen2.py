from __future__ import annotations

import time
from typing import Any

from ..types import ExecutionPlan, ModelDescriptor


class Qwen2Adapter:
    """Qwen2-family text generation on the project's standard MLX-LM path."""

    name = "qwen2-mlx-lm"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.descriptor: ModelDescriptor | None = None

    @staticmethod
    def can_load(descriptor: ModelDescriptor) -> bool:
        return (
            descriptor.model_type.lower() == "qwen2"
            and descriptor.architecture.lower() == "qwen2forcausallm"
        )

    def load(
        self, descriptor: ModelDescriptor, plan: ExecutionPlan | None = None
    ) -> "Qwen2Adapter":
        try:
            from mlx_lm import load
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "mlx-lm is not installed in MiniMLX's .venv; "
                "run tools/bootstrap_venv.sh"
            ) from exc
        self.model, self.tokenizer = load(descriptor.model_dir)
        self.descriptor = descriptor
        return self

    def unload(self) -> None:
        self.tokenizer = None
        self.model = None
        self.descriptor = None

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
