"""Headless planning and indexed reads for large-model offload.

This module deliberately does not load MLX.  It defines the resource contract
that a future Qwen3.8-Flash-Next executor can use for RAM/GPU/SSD placement.
"""

from __future__ import annotations

import json
import mmap
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3


def qwen38_should_use_ssd_offload(
    total_weight_bytes: int,
    *,
    total_memory_bytes: int | None,
    available_memory_bytes: int | None,
    weight_peak_multiplier: float = 1.25,
) -> bool:
    """Select SSD mode only when the full resident load is unsafe.

    ``None`` memory readings are treated conservatively.  The multiplier
    covers temporary load/quantization workspace and is intentionally kept
    separate from the SSD resident-set planner.
    """
    if total_weight_bytes < 0 or weight_peak_multiplier < 1.0:
        raise ValueError("invalid weight size or peak multiplier")
    if total_weight_bytes == 0:
        return False
    if total_memory_bytes is None or available_memory_bytes is None:
        return True
    from .api.memory import memory_reserve_bytes

    safe_available = max(
        0,
        available_memory_bytes - memory_reserve_bytes(total_memory_bytes),
    )
    return int(total_weight_bytes * weight_peak_multiplier) > safe_available


@dataclass(frozen=True)
class ResidentBudget:
    """Safe resident-memory budget for an offloaded model."""

    total_bytes: int
    available_bytes: int
    os_reserve_bytes: int = 2 * GIB
    runtime_reserve_bytes: int = 2 * GIB
    kv_reserve_bytes: int = 0
    configured_bytes: int | None = None

    def __post_init__(self) -> None:
        if min(
            self.total_bytes,
            self.available_bytes,
            self.os_reserve_bytes,
            self.runtime_reserve_bytes,
            self.kv_reserve_bytes,
        ) < 0:
            raise ValueError("memory budget values must be non-negative")

    @property
    def safe_bytes(self) -> int:
        capacity = max(
            0,
            min(
                self.total_bytes,
                self.available_bytes,
            )
            - self.os_reserve_bytes
            - self.runtime_reserve_bytes
            - self.kv_reserve_bytes,
        )
        if self.configured_bytes is not None:
            capacity = min(capacity, max(0, self.configured_bytes))
        return capacity


@dataclass(frozen=True)
class OffloadPlan:
    """Placement plan before any model weights are materialized."""

    model_path: str
    total_weight_bytes: int
    dense_resident_bytes: int
    expert_stream_bytes: int
    ple_stream_bytes: int
    resident_budget_bytes: int
    minimum_resident_bytes: int
    use_ssd_offload: bool
    reason: str

    @property
    def stream_bytes(self) -> int:
        return self.expert_stream_bytes + self.ple_stream_bytes

    @property
    def fits_without_offload(self) -> bool:
        return self.total_weight_bytes <= self.resident_budget_bytes

    @property
    def can_start(self) -> bool:
        return self.minimum_resident_bytes <= self.resident_budget_bytes


def make_qwen38_offload_plan(
    model_path: str | Path,
    *,
    total_weight_bytes: int,
    dense_resident_bytes: int,
    expert_bytes: int,
    ple_bytes: int,
    budget: ResidentBudget,
    use_ssd_offload: bool = True,
) -> OffloadPlan:
    """Build a conservative placement plan for Qwen3.8-Flash-Next.

    The caller supplies measured/indexed sizes because checkpoint layouts and
    quantization formats vary.  This function never guesses from directory
    size and never claims that a model fits merely because it is sparse.
    """

    sizes = (total_weight_bytes, dense_resident_bytes, expert_bytes, ple_bytes)
    if min(sizes) < 0:
        raise ValueError("model sizes must be non-negative")
    if dense_resident_bytes > total_weight_bytes:
        raise ValueError("dense resident size cannot exceed total weight size")
    if expert_bytes + ple_bytes > total_weight_bytes:
        raise ValueError("streamed components cannot exceed total weight size")

    resident = dense_resident_bytes
    minimum = resident
    if not use_ssd_offload:
        resident = total_weight_bytes
        minimum = total_weight_bytes
    can_start = minimum <= budget.safe_bytes
    if not can_start:
        reason = "minimum resident set exceeds safe memory budget"
    elif use_ssd_offload and not total_weight_bytes <= budget.safe_bytes:
        reason = "offload required; cold experts and PLE remain on SSD"
    else:
        reason = "full resident load fits the safe memory budget"

    return OffloadPlan(
        model_path=str(model_path),
        total_weight_bytes=total_weight_bytes,
        dense_resident_bytes=resident,
        expert_stream_bytes=expert_bytes if use_ssd_offload else 0,
        ple_stream_bytes=ple_bytes if use_ssd_offload else 0,
        resident_budget_bytes=budget.safe_bytes,
        minimum_resident_bytes=minimum,
        use_ssd_offload=use_ssd_offload,
        reason=reason,
    )


@dataclass(frozen=True)
class TensorRegion:
    """A byte range in an offload shard."""

    tensor: str
    path: Path
    offset: int
    length: int
    dtype: str | None = None
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.offset < 0 or self.length < 0:
            raise ValueError("tensor region offset and length must be non-negative")


class MmapWeightStore:
    """Read-only, bounded shard access for a MiniMLX offload index.

    The index format is intentionally small and explicit:
    ``{"tensors": {"name": {"file": "shard", "offset": 0, "length": 4096}}}``.
    It is a staging format for the future Qwen checkpoint packer, not a claim
    that every existing MLX checkpoint already has byte offsets.
    """

    def __init__(self, root: Path, regions: dict[str, TensorRegion]) -> None:
        self.root = root
        self._regions = regions

    @classmethod
    def from_index(cls, root: str | Path, index_path: str | Path) -> "MmapWeightStore":
        root = Path(root)
        index = json.loads(Path(index_path).read_text())
        raw_regions = index.get("tensors")
        if not isinstance(raw_regions, dict):
            raise ValueError("offload index must contain a tensors object")
        regions: dict[str, TensorRegion] = {}
        for tensor, raw in raw_regions.items():
            if not isinstance(tensor, str) or not isinstance(raw, dict):
                raise ValueError("invalid tensor region in offload index")
            filename = raw.get("file")
            resolved = (root / filename).resolve() if isinstance(filename, str) else None
            if (
                not isinstance(filename, str)
                or Path(filename).is_absolute()
                or resolved is None
                or not resolved.is_relative_to(root.resolve())
            ):
                raise ValueError("tensor shard must be a relative file path")
            regions[tensor] = TensorRegion(
                tensor=tensor,
                path=resolved,
                offset=int(raw.get("offset", -1)),
                length=int(raw.get("length", -1)),
                dtype=raw.get("dtype") if isinstance(raw.get("dtype"), str) else None,
                shape=tuple(raw.get("shape", ()))
                if isinstance(raw.get("shape", ()), list)
                else (),
            )
        return cls(root, regions)

    def region(self, tensor: str) -> TensorRegion:
        try:
            return self._regions[tensor]
        except KeyError as error:
            raise KeyError(f"tensor is not present in offload index: {tensor}") from error

    def read(self, tensor: str, *, offset: int = 0, length: int | None = None) -> bytes:
        region = self.region(tensor)
        if offset < 0 or offset > region.length:
            raise ValueError("read offset is outside tensor region")
        read_length = region.length - offset if length is None else length
        if read_length < 0 or offset + read_length > region.length:
            raise ValueError("read range is outside tensor region")
        with region.path.open("rb") as file:
            if region.offset + region.length > region.path.stat().st_size:
                raise ValueError(f"tensor region exceeds shard size: {region.path}")
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                start = region.offset + offset
                return mapped[start : start + read_length]


def build_safetensors_offload_index(
    model_root: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Build a bounded tensor index from a standard safetensors checkpoint.

    Safetensors stores tensor offsets in each shard header, so creating this
    index only reads metadata.  Tensor bytes are never materialized.  The
    generated index is deliberately compatible with :meth:`MmapWeightStore`
    and rejects shard paths that escape the checkpoint directory.
    """
    root = Path(model_root).expanduser().resolve()
    source = root / "model.safetensors.index.json"
    if not source.is_file():
        raise FileNotFoundError(f"missing safetensors index: {source}")
    raw = json.loads(source.read_text())
    weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index must contain a weight_map object")

    headers: dict[str, dict] = {}
    for filename in sorted(set(weight_map.values())):
        if not isinstance(filename, str):
            raise ValueError("weight_map shard names must be strings")
        shard = Path(filename)
        resolved = (root / shard).resolve()
        if shard.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError(f"weight shard escapes model directory: {filename!r}")
        with resolved.open("rb") as file:
            header_length_bytes = file.read(8)
            if len(header_length_bytes) != 8:
                raise ValueError(f"truncated safetensors header: {resolved}")
            header_length = int.from_bytes(header_length_bytes, "little")
            header = json.loads(file.read(header_length))
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header must be an object: {resolved}")
        headers[filename] = header

    regions: dict[str, dict[str, int | str]] = {}
    for tensor, filename in weight_map.items():
        if not isinstance(tensor, str) or not isinstance(filename, str):
            raise ValueError("weight_map tensor names and shard names must be strings")
        metadata = headers[filename].get(tensor)
        if not isinstance(metadata, dict):
            raise ValueError(f"tensor is missing from shard header: {tensor}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and value >= 0 for value in offsets)
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"invalid data_offsets for tensor: {tensor}")
        # The exact header length must come from the shard, not re-encoding
        # JSON, because whitespace and padding are part of the binary format.
        with (root / filename).open("rb") as file:
            header_length = int.from_bytes(file.read(8), "little")
        start, end = offsets
        regions[tensor] = {
            "file": filename,
            "offset": 8 + header_length + start,
            "length": end - start,
            "dtype": metadata.get("dtype"),
            "shape": metadata.get("shape", []),
        }

    destination = Path(output_path) if output_path is not None else root / ".minimlx-offload-index.json"
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"version": 1, "tensors": regions}, indent=2) + "\n")
    return destination


class ExpertCache:
    """Byte-bounded LRU cache for decoded expert payloads.

    Values are opaque to this module: the future MLX executor can store an
    ``mx.array`` or a staging buffer, while headless tests can use bytes.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("expert cache capacity must be non-negative")
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[str, tuple[int, object]] = OrderedDict()
        self._used_bytes = 0
        self.hits = 0
        self.misses = 0

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def get(self, key: str) -> object | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry[1]

    def put(self, key: str, value: object, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("expert cache entry size must be non-negative")
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._used_bytes -= previous[0]
        if size_bytes > self.capacity_bytes:
            return
        self._entries[key] = (size_bytes, value)
        self._used_bytes += size_bytes
        while self._used_bytes > self.capacity_bytes and self._entries:
            _, (evicted_size, _) = self._entries.popitem(last=False)
            self._used_bytes -= evicted_size

    def discard(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._used_bytes -= entry[0]

    def clear(self) -> None:
        self._entries.clear()
        self._used_bytes = 0


class PrefetchScheduler:
    """Small deterministic scheduler for expert/tensor prefetch requests."""

    def __init__(self, max_batch_size: int = 4) -> None:
        if max_batch_size <= 0:
            raise ValueError("prefetch batch size must be positive")
        self.max_batch_size = max_batch_size
        self._queue: list[str] = []
        self._queued: set[str] = set()

    def request(self, keys: list[str] | tuple[str, ...]) -> None:
        for key in keys:
            if key not in self._queued:
                self._queued.add(key)
                self._queue.append(key)

    def pop_batch(self) -> tuple[str, ...]:
        batch = tuple(self._queue[: self.max_batch_size])
        del self._queue[: self.max_batch_size]
        for key in batch:
            self._queued.discard(key)
        return batch

    def complete(self, keys: list[str] | tuple[str, ...]) -> None:
        completed = set(keys)
        if not completed:
            return
        self._queue = [key for key in self._queue if key not in completed]
        self._queued.difference_update(completed)


class Qwen38OffloadSession:
    """Storage session shared by the future Qwen3-Next expert executor."""

    def __init__(
        self,
        store: MmapWeightStore,
        *,
        cache_bytes: int = 512 * 1024 * 1024,
        prefetch_batch_size: int = 4,
    ) -> None:
        self.store = store
        self.cache = ExpertCache(cache_bytes)
        self.prefetcher = PrefetchScheduler(prefetch_batch_size)

    @classmethod
    def open(
        cls,
        model_root: str | Path,
        *,
        index_path: str | Path | None = None,
        cache_bytes: int = 512 * 1024 * 1024,
        prefetch_batch_size: int = 4,
    ) -> "Qwen38OffloadSession":
        root = Path(model_root).expanduser().resolve()
        selected_index = (
            Path(index_path)
            if index_path is not None
            else root / ".minimlx-offload-index.json"
        )
        if not selected_index.is_absolute():
            selected_index = root / selected_index
        if not selected_index.is_file():
            selected_index = build_safetensors_offload_index(
                root, output_path=selected_index
            )
        return cls(
            MmapWeightStore.from_index(root, selected_index),
            cache_bytes=cache_bytes,
            prefetch_batch_size=prefetch_batch_size,
        )

    def read(self, tensor: str, *, offset: int = 0, length: int | None = None) -> bytes:
        cache_key = f"{tensor}:{offset}:{length}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        value = self.store.read(tensor, offset=offset, length=length)
        self.cache.put(cache_key, value, len(value))
        return value

    def read_array(self, tensor: str):
        """Decode one indexed tensor as a NumPy array without MLX imports."""
        region = self.store.region(tensor)
        if not region.shape:
            raise ValueError(f"tensor metadata is incomplete for {tensor!r}")
        return self.read_array_slice(
            tensor,
            offset=0,
            length=region.length,
            shape=region.shape,
        )

    def read_array_slice(
        self,
        tensor: str,
        *,
        offset: int,
        length: int,
        shape: tuple[int, ...],
    ):
        """Decode a bounded tensor slice with caller-provided output shape."""
        import numpy as np

        region = self.store.region(tensor)
        if not region.dtype or not shape:
            raise ValueError(f"tensor metadata is incomplete for {tensor!r}")
        dtype_map = {
            "BOOL": np.bool_,
            "U8": np.uint8,
            "I8": np.int8,
            "I16": np.int16,
            "I32": np.int32,
            "I64": np.int64,
            "F16": np.float16,
            "F32": np.float32,
            "F64": np.float64,
        }
        dtype = dtype_map.get(region.dtype)
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype for {tensor!r}: {region.dtype}")
        result = np.frombuffer(
            self.read(tensor, offset=offset, length=length), dtype=dtype
        )
        expected = 1
        for dimension in shape:
            expected *= dimension
        if result.size != expected:
            raise ValueError(
                f"tensor byte length does not match shape for {tensor!r}: "
                f"{result.size} values versus {expected}"
            )
        return result.reshape(shape)

    def request_prefetch(self, tensors: list[str] | tuple[str, ...]) -> None:
        self.prefetcher.request(list(tensors))

    def prefetch_next_batch(self) -> dict[str, bytes]:
        batch = self.prefetcher.pop_batch()
        values = {tensor: self.read(tensor) for tensor in batch}
        self.prefetcher.complete(batch)
        return values
