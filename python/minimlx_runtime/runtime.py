from __future__ import annotations


from .adapters import Qwen35Adapter
from .registry import ModelRegistry, describe_model
from .types import GenerationResult


class Runtime:
    """Long-lived model runtime used by the future Rust supervisor."""

    def __init__(self) -> None:
        self.registry = ModelRegistry([Qwen35Adapter()])
        self.adapter = None

    def load(self, model_dir: str | Path) -> None:
        descriptor = describe_model(model_dir)
        self.adapter = self.registry.resolve(descriptor).load(descriptor)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        context_tokens: int | None = None,
        prefill_step_size: int = 2048,
        session_id: str | None = None,
    ) -> GenerationResult:
        if self.adapter is None:
            raise RuntimeError("load() must be called before generate()")
        prompt_tokens, generated, text, prefill_ms, decode_ms = self.adapter.generate(
            prompt, max_tokens, context_tokens, prefill_step_size, session_id
        )
        return GenerationResult(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            text=text,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
        )

    def clear_session(self, session_id: str) -> None:
        if self.adapter is None:
            raise RuntimeError("load() must be called before clear_session")
        self.adapter.clear_session(session_id)
