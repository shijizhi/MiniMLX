from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelDescriptor:
    model_dir: str
    model_type: str
    architecture: str
    tokenizer_family: str
    capabilities: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPlan:
    phase: str
    batch_size: int = 1
    use_custom_kernels: bool = True
    use_continuous_batching: bool = False
    use_speculative_decoding: bool = False


@dataclass(frozen=True)
class GenerationResult:
    prompt_tokens: list[int]
    generated_tokens: list[int]
    text: str
    prefill_ms: float
    decode_ms: float

    @property
    def decode_tokens_per_second(self) -> float:
        if self.decode_ms <= 0 or not self.generated_tokens:
            return 0.0
        return len(self.generated_tokens) * 1000.0 / self.decode_ms
