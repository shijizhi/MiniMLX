"""Backend-neutral execution primitives for DeepSeek V4.1 Flash 1017."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class DeepSeek41Route:
    expert_ids: Any
    scores: Any


class DeepSeek41Router:
    """Reference top-k router with an optional image-token bias."""

    def __init__(self, num_experts: int = 384, top_k: int = 6) -> None:
        if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
            raise ValueError("invalid DeepSeek expert routing dimensions")
        self.num_experts = num_experts
        self.top_k = top_k

    def route(self, logits, *, image_token_mask=None, image_bias=None) -> DeepSeek41Route:
        import numpy as np

        values = np.asarray(logits)
        if values.shape[-1] != self.num_experts:
            raise ValueError(
                f"router logits must have {self.num_experts} experts, got {values.shape[-1]}"
            )
        adjusted = values.astype(np.float32, copy=True)
        if image_token_mask is not None:
            mask = np.asarray(image_token_mask, dtype=bool)
            if mask.shape != values.shape[:-1]:
                raise ValueError("image_token_mask must match router token dimensions")
            if image_bias is None:
                raise ValueError("image_bias is required for image-token routing")
            bias = np.asarray(image_bias, dtype=np.float32)
            if bias.shape != (self.num_experts,):
                raise ValueError("image_bias must contain one value per expert")
            adjusted[mask] += bias
        selected = np.argpartition(
            adjusted, -self.top_k, axis=-1
        )[..., -self.top_k :]
        selected_scores = np.take_along_axis(adjusted, selected, axis=-1)
        order = np.argsort(-selected_scores, axis=-1)
        selected = np.take_along_axis(selected, order, axis=-1)
        selected_scores = np.take_along_axis(selected_scores, order, axis=-1)
        selected_scores = _softmax(selected_scores, axis=-1)
        return DeepSeek41Route(selected, selected_scores)


def _softmax(values, axis: int):
    import numpy as np

    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


class DeepSeek41CEDCache:
    """Separate encoder output and decoder state for causal encoder-decoder execution."""

    def __init__(self) -> None:
        self.encoder_output: Any | None = None
        self.decoder_states: list[Any] = []

    def set_encoder_output(self, value: Any) -> None:
        self.encoder_output = value
        self.decoder_states.clear()

    def append_decoder_state(self, value: Any) -> None:
        if self.encoder_output is None:
            raise RuntimeError("encoder output must be set before decoder state")
        self.decoder_states.append(value)

    def clear(self) -> None:
        self.encoder_output = None
        self.decoder_states.clear()

    @property
    def decoder_length(self) -> int:
        return len(self.decoder_states)


class DeepSeek41EngramLookup:
    """Hash lookup contract for Engram tables with injectable model hash logic."""

    def __init__(
        self,
        read_rows: Callable[[Iterable[int]], Any],
        *,
        hash_fn: Callable[[tuple[int, ...]], int],
        orders: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        if not orders or any(order <= 0 for order in orders):
            raise ValueError("Engram orders must be positive")
        self._read_rows = read_rows
        self._hash_fn = hash_fn
        self.orders = orders

    def lookup(self, token_ids: Iterable[int]) -> Any:
        tokens = tuple(int(token) for token in token_ids)
        rows: list[int] = []
        for order in self.orders:
            rows.extend(
                self._hash_fn(tokens[index - order : index])
                for index in range(order, len(tokens) + 1)
            )
        return self._read_rows(rows)
