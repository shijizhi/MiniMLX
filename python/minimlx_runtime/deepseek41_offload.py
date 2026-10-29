"""Headless storage contracts for DeepSeek V4.1 Flash 1017.

This module deliberately stops at tensor layout and cache management.  The
MLX executor is family-specific and must not be inferred from Qwen3.8 code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .offload import Qwen38OffloadSession


@dataclass(frozen=True)
class DeepSeek41TensorLayout:
    encoder: tuple[str, ...] = ()
    decoder: tuple[str, ...] = ()
    routed_experts: tuple[str, ...] = ()
    engram: tuple[str, ...] = ()
    sparse_indexer: tuple[str, ...] = ()
    hyper_connections: tuple[str, ...] = ()
    dspark: tuple[str, ...] = ()
    other: tuple[str, ...] = ()

    @classmethod
    def from_tensor_names(cls, names: Iterable[str]) -> "DeepSeek41TensorLayout":
        groups: dict[str, list[str]] = {
            "encoder": [],
            "decoder": [],
            "routed_experts": [],
            "engram": [],
            "sparse_indexer": [],
            "hyper_connections": [],
            "dspark": [],
            "other": [],
        }
        for name in sorted(set(names)):
            lowered = name.lower()
            if "engram" in lowered or "ngram" in lowered:
                group = "engram"
            elif any(token in lowered for token in ("indexer", "sparse_attention", "mla")):
                group = "sparse_indexer"
            elif any(token in lowered for token in ("hyper", "hc_", "hyper_connection")):
                group = "hyper_connections"
            elif any(token in lowered for token in ("dspark", "draft", "mtp")):
                group = "dspark"
            elif ".experts." in lowered or ".expert." in lowered:
                group = "routed_experts"
            elif "encoder" in lowered:
                group = "encoder"
            elif "decoder" in lowered or ".layers." in lowered:
                group = "decoder"
            else:
                group = "other"
            groups[group].append(name)
        return cls(**{key: tuple(value) for key, value in groups.items()})

    @classmethod
    def from_safetensors_index(cls, index_path: str | Path) -> "DeepSeek41TensorLayout":
        index = json.loads(Path(index_path).read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError("safetensors index must contain a weight_map object")
        return cls.from_tensor_names(weight_map.keys())


@dataclass(frozen=True)
class DeepSeek41CacheBudget:
    """Separate budgets because Engram and compressed KV have different access patterns."""

    resident_bytes: int
    encoder_bytes: int
    decoder_kv_bytes: int
    engram_page_bytes: int

    def __post_init__(self) -> None:
        if min(
            self.resident_bytes,
            self.encoder_bytes,
            self.decoder_kv_bytes,
            self.engram_page_bytes,
        ) < 0:
            raise ValueError("DeepSeek cache budgets must be non-negative")
        if self.encoder_bytes + self.decoder_kv_bytes + self.engram_page_bytes > self.resident_bytes:
            raise ValueError("DeepSeek cache budgets exceed resident budget")


class EngramPageStore:
    """Page-oriented access to one or more indexed Engram tensors."""

    def __init__(
        self,
        session: Qwen38OffloadSession,
        tensors: tuple[str, ...],
        *,
        page_rows: int = 4096,
    ) -> None:
        if not tensors:
            raise ValueError("EngramPageStore requires at least one tensor")
        if page_rows <= 0:
            raise ValueError("page_rows must be positive")
        self.session = session
        self.tensors = tensors
        self.page_rows = page_rows

    def read_page(self, tensor: str, page: int):
        if tensor not in self.tensors:
            raise KeyError(f"tensor is not an Engram tensor: {tensor}")
        if page < 0:
            raise ValueError("page must be non-negative")
        region = self.session.store.region(tensor)
        if not region.shape:
            raise ValueError(f"Engram tensor has no shape metadata: {tensor}")
        rows = region.shape[0]
        start = page * self.page_rows
        if start >= rows:
            raise ValueError(f"Engram page is outside tensor: {tensor} page={page}")
        count = min(self.page_rows, rows - start)
        row_bytes = region.length // rows
        return self.session.read_array_slice(
            tensor,
            offset=start * row_bytes,
            length=count * row_bytes,
            shape=(count, *region.shape[1:]),
        )
