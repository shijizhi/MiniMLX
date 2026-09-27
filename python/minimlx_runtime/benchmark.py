"""Small, deterministic benchmark primitives shared by all benchmark suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkPolicy:
    warmup: int = 3
    repeats: int = 5
    statistic: str = "median"

    def __post_init__(self) -> None:
        if self.warmup < 0 or self.repeats <= 0:
            raise ValueError("warmup must be >= 0 and repeats must be > 0")
        if self.statistic != "median":
            raise ValueError("only median is supported until other statistics are specified")


def measure(policy: BenchmarkPolicy, fn: Callable[[], T]) -> tuple[T, list[float]]:
    """Warm a callable, then return its final value and sorted elapsed samples."""
    for _ in range(policy.warmup):
        fn()
    values: list[float] = []
    final: T | None = None
    import time

    for _ in range(policy.repeats):
        start = time.perf_counter()
        final = fn()
        values.append((time.perf_counter() - start) * 1000.0)
    values.sort()
    return final, values


def summarize(samples: list[float]) -> float:
    if not samples:
        raise ValueError("cannot summarize an empty sample list")
    return float(median(samples))


def policy_metadata(policy: BenchmarkPolicy) -> dict[str, int | str]:
    return asdict(policy)
