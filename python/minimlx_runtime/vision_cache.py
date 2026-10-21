"""Bounded cache for prepared multimodal inputs.

This is deliberately separate from KV/prefix cache. Until the loaded VLM
exposes a stable embedding-injection contract, caching the processor output is
the safe reusable boundary; vision-feature caching will sit behind the same
key and lifecycle once that contract is available.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Hashable


def value_nbytes(value: Any) -> int:
    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if isinstance(value, dict):
        return sum(value_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(value_nbytes(item) for item in value)
    if is_dataclass(value):
        return sum(value_nbytes(getattr(value, field.name)) for field in fields(value))
    return 0


@dataclass(frozen=True)
class _Entry:
    value: Any
    nbytes: int


class VisionInputCache:
    """Small model-local LRU for immutable prepared VLM input tensors."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.stores = 0

    def get(self, key: Hashable) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry.value

    def put(self, key: Hashable, value: Any) -> None:
        if self.max_bytes <= 0:
            return
        nbytes = value_nbytes(value)
        if nbytes > self.max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous.nbytes
        self._entries[key] = _Entry(value, nbytes)
        self._bytes += nbytes
        self.stores += 1
        while self._bytes > self.max_bytes and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.nbytes

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def stats(self) -> tuple[int, int, int, int]:
        return self.hits, self.misses, self.stores, self._bytes
