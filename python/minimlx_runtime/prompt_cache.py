"""Bounded, model-local prefix cache for inference-only reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx


def clone_cache(model: Any, source: Sequence[Any]) -> list[Any]:
    """Clone MLX cache state without Python ``deepcopy`` semantics."""
    target = model.make_cache()
    for dst, src in zip(target, source):
        state = src.state
        if isinstance(state, (list, tuple)):
            state = type(state)(
                None if value is None else mx.array(value) for value in state
            )
        elif state is not None:
            state = mx.array(state)
        dst.state = state
        for name in ("lengths", "left_padding"):
            value = getattr(src, name, None)
            if value is not None:
                setattr(dst, name, mx.array(value))
    return target


@dataclass
class _Entry:
    tokens: tuple[int, ...]
    cache: list[Any]
    nbytes: int


class PrefixCache:
    """Small exact-token prefix cache with longest-prefix lookup.

    Entries own immutable snapshots. A clone is returned to every caller so
    generation cannot mutate the stored state. This first implementation is
    intentionally bounded and process-local; persistence and trie indexing
    can be added after the correctness gate.
    """

    def __init__(self, model: Any, max_bytes: int) -> None:
        self.model = model
        self.max_bytes = max(0, max_bytes)
        self._entries: list[_Entry] = []
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.stores = 0

    @staticmethod
    def _cache_nbytes(cache: Sequence[Any]) -> int:
        return sum(int(getattr(item, "nbytes", 0)) for item in cache)

    def lookup(self, tokens: Sequence[int]) -> tuple[int, list[Any]] | None:
        token_tuple = tuple(int(token) for token in tokens)
        candidates = [
            entry
            for entry in self._entries
            if len(entry.tokens) < len(token_tuple)
            and token_tuple[: len(entry.tokens)] == entry.tokens
        ]
        if not candidates:
            self.misses += 1
            return None
        entry = max(candidates, key=lambda item: len(item.tokens))
        self.hits += 1
        return len(entry.tokens), clone_cache(self.model, entry.cache)

    def put(self, tokens: Sequence[int], cache: Sequence[Any]) -> None:
        if self.max_bytes <= 0:
            return
        token_tuple = tuple(int(token) for token in tokens)
        snapshot = clone_cache(self.model, cache)
        nbytes = self._cache_nbytes(snapshot)
        if nbytes > self.max_bytes:
            return
        self._entries = [entry for entry in self._entries if entry.tokens != token_tuple]
        self._bytes = sum(entry.nbytes for entry in self._entries)
        self._entries.append(_Entry(token_tuple, snapshot, nbytes))
        self._bytes += nbytes
        self.stores += 1
        while self._bytes > self.max_bytes and self._entries:
            evicted = self._entries.pop(0)
            self._bytes -= evicted.nbytes

    @property
    def bytes(self) -> int:
        return self._bytes

    def stats(self) -> tuple[int, int, int, int]:
        return self.hits, self.misses, self.stores, self._bytes
