from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .types import ModelDescriptor


class ModelAdapter(Protocol):
    name: str

    def can_load(self, descriptor: ModelDescriptor) -> bool: ...

    def load(self, descriptor: ModelDescriptor): ...


def describe_model(model_dir: str | Path) -> ModelDescriptor:
    path = Path(model_dir).expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text())
    model_type = str(config.get("model_type", "unknown"))
    architectures = config.get("architectures", [])
    architecture = str(architectures[0] if architectures else model_type)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = {}
    capabilities = {"prefill", "decode", "single_request"}
    if "qwen3" in model_type.lower() or "qwen3" in architecture.lower():
        capabilities.update({"hybrid_attention", "recurrent_state"})
    num_experts = text_config.get("num_experts", config.get("num_experts"))
    experts_per_token = text_config.get(
        "num_experts_per_tok", config.get("num_experts_per_tok")
    )
    if num_experts:
        capabilities.add("moe")
    return ModelDescriptor(
        model_dir=str(path),
        model_type=model_type,
        architecture=architecture,
        tokenizer_family="qwen" if "qwen" in model_type.lower() else "huggingface",
        capabilities=frozenset(capabilities),
        metadata={
            "quantization": config.get("quantization"),
            "dtype": config.get("dtype", text_config.get("dtype")),
            "num_hidden_layers": config.get(
                "num_hidden_layers", text_config.get("num_hidden_layers")
            ),
            "hidden_size": config.get("hidden_size", text_config.get("hidden_size")),
            "num_experts": num_experts,
            "num_experts_per_tok": experts_per_token,
            "moe_intermediate_size": text_config.get(
                "moe_intermediate_size", config.get("moe_intermediate_size")
            ),
            "shared_expert_intermediate_size": text_config.get(
                "shared_expert_intermediate_size",
                config.get("shared_expert_intermediate_size"),
            ),
        },
    )


class ModelRegistry:
    def __init__(self, adapters: list[ModelAdapter] | None = None) -> None:
        self.adapters = adapters or []

    def register(self, adapter: ModelAdapter) -> None:
        self.adapters.append(adapter)

    def resolve(self, descriptor: ModelDescriptor) -> ModelAdapter:
        for adapter in self.adapters:
            if adapter.can_load(descriptor):
                return adapter
        names = ", ".join(adapter.name for adapter in self.adapters) or "none"
        raise ValueError(
            f"no adapter can load model_type={descriptor.model_type!r}; "
            f"registered adapters: {names}"
        )
