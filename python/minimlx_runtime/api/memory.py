"""Best-effort system and MLX memory readings for API admission control.

The readings are intentionally observational: the guard only refuses new
requests. It never interrupts an active generation, since doing so would make
streaming responses non-resumable and can leave backend state half-cleaned.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024**3
DEFAULT_RESERVE_RATIO = 0.02
DEFAULT_MINIMUM_RESERVE = int(1.5 * GIB)
MAXIMUM_RESERVE = 4 * GIB


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int | None
    available_bytes: int | None
    used_bytes: int | None
    source: str

    def as_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    code: str | None = None
    message: str | None = None
    retry_after: str | None = None
    snapshot: MemorySnapshot | None = None


def _command_text(*args: str) -> str | None:
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def read_system_memory() -> MemorySnapshot:
    """Read system memory without treating MLX allocator stats as system RAM."""
    if platform.system() == "Darwin":
        total_text = _command_text("sysctl", "-n", "hw.memsize")
        vm_text = _command_text("vm_stat")
        try:
            total = int(total_text.strip()) if total_text else None
        except ValueError:
            total = None
        if not total:
            try:
                total = int(os.sysconf("SC_PHYS_PAGES")) * int(
                    os.sysconf("SC_PAGESIZE")
                )
            except (OSError, ValueError):
                total = None
        if not total or not vm_text:
            return MemorySnapshot(total, None, None, "darwin-unavailable")

        page_match = re.search(r"page size of (\d+) bytes", vm_text)
        page_size = int(page_match.group(1)) if page_match else 4096
        counts: dict[str, int] = {}
        for line in vm_text.splitlines():
            match = re.match(r"Pages ([^:]+):\s*(\d+)", line)
            if match:
                counts[match.group(1).lower()] = int(match.group(2))

        # Treat active, wired, and compressed pages as used. Inactive pages are
        # reclaimable and therefore remain part of the available estimate.
        used_pages = sum(
            counts.get(name, 0)
            for name in ("active", "wired down", "occupied by compressor")
        )
        used = min(total, used_pages * page_size)
        return MemorySnapshot(total, max(0, total - used), used, "darwin-vm_stat")

    if platform.system() == "Linux":
        try:
            values: dict[str, int] = {}
            with open("/proc/meminfo", encoding="ascii") as meminfo:
                for line in meminfo:
                    key, raw = line.split(":", 1)
                    match = re.search(r"\d+", raw)
                    if match:
                        values[key] = int(match.group()) * 1024
            total = values.get("MemTotal")
            available = values.get("MemAvailable")
            if total is not None and available is not None:
                available = min(total, available)
                return MemorySnapshot(total, available, total - available, "linux-meminfo")
        except (OSError, ValueError):
            pass

    return MemorySnapshot(None, None, None, "unavailable")


def memory_reserve_bytes(total_bytes: int) -> int:
    """Use oMLX aggressive-tier reserve: 2% of RAM, bounded to 1.5–4 GiB."""
    raw_ratio = os.environ.get("MINIMLX_MEMORY_RESERVE_RATIO")
    try:
        ratio = float(raw_ratio) if raw_ratio is not None else DEFAULT_RESERVE_RATIO
    except ValueError:
        ratio = DEFAULT_RESERVE_RATIO
    ratio = min(0.50, max(0.02, ratio))

    raw_minimum = os.environ.get("MINIMLX_MEMORY_RESERVE_GB")
    try:
        minimum = int(float(raw_minimum) * GIB) if raw_minimum is not None else DEFAULT_MINIMUM_RESERVE
    except ValueError:
        minimum = DEFAULT_MINIMUM_RESERVE
    minimum = min(max(int(1.5 * GIB), minimum), max(int(1.5 * GIB), total_bytes // 2))
    return min(total_bytes, MAXIMUM_RESERVE, max(int(total_bytes * ratio), minimum))


def max_parallel_requests() -> int:
    """Keep local inference single-flight unless the user opts into risk."""
    try:
        configured = int(os.environ.get("MINIMLX_MAX_PARALLEL_REQUESTS", "1"))
    except ValueError:
        configured = 1
    return min(8, max(1, configured))


class InferenceAdmission:
    """Bound inference concurrency and check system headroom before each turn."""

    def __init__(
        self,
        limit: int,
        memory_reader=None,
        additional_reserve_bytes: int = 0,
    ) -> None:
        self._slots = threading.BoundedSemaphore(min(8, max(1, limit)))
        self._memory_reader = memory_reader or read_system_memory
        self._additional_reserve_bytes = max(0, int(additional_reserve_bytes))

    def acquire(self) -> AdmissionDecision:
        if not self._slots.acquire(blocking=False):
            return AdmissionDecision(
                False,
                "server_busy",
                "MiniMLX is already serving the maximum number of requests. "
                "The current response is continuing; retry this request shortly.",
                "2",
            )
        try:
            snapshot = self._memory_reader()
            reason = admission_error(
                snapshot, self._additional_reserve_bytes
            )
        except Exception as exc:
            self._slots.release()
            return AdmissionDecision(
                False,
                "memory_check_unavailable",
                "MiniMLX could not verify system memory safety "
                f"({type(exc).__name__}); no inference was started. Please retry shortly.",
                "10",
            )
        if reason:
            self._slots.release()
            code = (
                "memory_check_unavailable"
                if snapshot.total_bytes is None or snapshot.available_bytes is None
                else "memory_pressure"
            )
            return AdmissionDecision(False, code, reason, "10", snapshot)
        return AdmissionDecision(True, snapshot=snapshot)

    def release(self) -> None:
        self._slots.release()


def admission_error(
    snapshot: MemorySnapshot, additional_reserve_bytes: int = 0
) -> str | None:
    """Return a user-facing refusal reason when the OS reserve is threatened."""
    if snapshot.total_bytes is None or snapshot.available_bytes is None:
        return (
            "MiniMLX paused this request because it could not verify available "
            "system memory. No inference was started; retry after the memory "
            "monitor is available."
        )
    reserve = min(
        snapshot.total_bytes,
        memory_reserve_bytes(snapshot.total_bytes)
        + max(0, int(additional_reserve_bytes)),
    )
    if snapshot.available_bytes >= reserve:
        return None
    return (
        "MiniMLX paused this request to protect system memory: "
        f"about {snapshot.available_bytes / GIB:.1f} GB is available, while "
        f"{reserve / GIB:.1f} GB is reserved for macOS, other apps, and the "
        "configured request context. "
        "The active response was not interrupted. Close other apps or wait for "
        "memory pressure to drop, then retry the request."
    )


def mlx_memory_snapshot() -> dict[str, int | str | bool | None]:
    """Expose allocator categories when MLX provides them; never infer missing data."""
    try:
        import mlx.core as mx
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__}

    result: dict[str, int | str | bool | None] = {"available": True}
    for output_key, function_name in (
        ("active_bytes", "get_active_memory"),
        ("peak_bytes", "get_peak_memory"),
        ("cache_bytes", "get_cache_memory"),
    ):
        function = getattr(mx, function_name, None)
        if function is None:
            result[output_key] = None
            continue
        try:
            result[output_key] = int(function())
        except Exception as exc:  # telemetry must not break inference admission
            result[output_key] = None
            result[f"{output_key}_error"] = type(exc).__name__
    return result


def _kv_cache_estimate(metadata: dict, context_tokens: int) -> int:
    def positive_int(key: str) -> int | None:
        value = metadata.get(key)
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    layers = positive_int("num_hidden_layers")
    attention_heads = positive_int("num_attention_heads")
    kv_heads = (
        positive_int("num_key_value_heads")
        or positive_int("num_kv_heads")
        or attention_heads
    )
    hidden_size = positive_int("hidden_size")
    if not all((layers, attention_heads, kv_heads, hidden_size)):
        return 0
    head_dim = hidden_size // attention_heads
    if head_dim <= 0:
        return 0

    # Two tensors (key/value), two bytes per element. Hybrid-attention models
    # are intentionally estimated as if every layer held a full KV cache.
    return int(context_tokens) * layers * kv_heads * head_dim * 4


def estimate_context_memory_bytes(runtime, context_tokens: int) -> int:
    """Conservative full-context K/V estimate including configured drafters."""
    if context_tokens <= 0:
        return 0
    descriptor = getattr(runtime, "descriptor", None)
    metadata = getattr(descriptor, "metadata", None)
    target_estimate = _kv_cache_estimate(metadata, context_tokens) if isinstance(metadata, dict) else 0
    total = target_estimate
    for variable in ("MINIMLX_MTP_DRAFT_MODEL", "MINIMLX_DFLASH_DRAFT_MODEL"):
        draft_path = os.environ.get(variable)
        if not draft_path:
            continue
        try:
            config = json.loads(
                (Path(draft_path).expanduser() / "config.json").read_text()
            )
            if not isinstance(config, dict):
                raise ValueError("model config is not an object")
            text_config = config.get("text_config")
            if not isinstance(text_config, dict):
                text_config = {}
            draft_metadata = {
                "num_hidden_layers": config.get("num_hidden_layers", text_config.get("num_hidden_layers", config.get("n_layer"))),
                "num_attention_heads": config.get("num_attention_heads", text_config.get("num_attention_heads", config.get("n_head"))),
                "num_key_value_heads": config.get("num_key_value_heads", text_config.get("num_key_value_heads", config.get("num_kv_heads"))),
                "hidden_size": config.get("hidden_size", text_config.get("hidden_size", config.get("n_embd"))),
            }
            draft_estimate = _kv_cache_estimate(draft_metadata, context_tokens)
        except (OSError, ValueError, TypeError):
            draft_estimate = 0
        # An opaque draft remains a real additional model. Reuse the target's
        # cache estimate as a conservative proxy instead of silently ignoring it.
        total += draft_estimate or target_estimate
    return total


def request_memory_snapshot(
    runtime, additional_reserve_bytes: int = 0
) -> dict[str, object]:
    system = read_system_memory()
    result: dict[str, object] = {"system": system.as_dict(), "mlx": mlx_memory_snapshot()}
    if system.total_bytes is not None and system.available_bytes is not None:
        reserve = min(
            system.total_bytes,
            memory_reserve_bytes(system.total_bytes)
            + max(0, int(additional_reserve_bytes)),
        )
        result["reserve_bytes"] = reserve
        result["admission"] = "paused" if system.available_bytes < reserve else "open"
    else:
        result["reserve_bytes"] = None
        result["admission"] = "unknown"
    result["active_model"] = getattr(
        getattr(runtime, "descriptor", None), "model_dir", None
    )
    result["estimated_request_memory_bytes"] = max(
        0, int(additional_reserve_bytes)
    )
    return result


class SystemMemoryMonitor:
    """Keep a low-frequency memory-pressure history while the API is serving."""

    def __init__(
        self,
        interval_seconds: float = 2.0,
        additional_reserve_bytes: int = 0,
    ) -> None:
        self._interval = max(0.5, interval_seconds)
        self._additional_reserve_bytes = max(0, int(additional_reserve_bytes))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: MemorySnapshot | None = None
        self._peak_used_bytes: int | None = None
        self._state = "unknown"
        self._sampled_at: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sample()
        self._thread = threading.Thread(
            target=self._run, name="minimlx-memory-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            latest = self._latest
            return {
                "system": latest.as_dict() if latest else None,
                "state": self._state,
                "sampled_at": self._sampled_at,
                "peak_used_bytes": self._peak_used_bytes,
                "reserve_bytes": (
                    min(
                        latest.total_bytes,
                        memory_reserve_bytes(latest.total_bytes)
                        + self._additional_reserve_bytes,
                    )
                    if latest and latest.total_bytes is not None
                    else None
                ),
            }

    def _sample(self) -> None:
        sample = read_system_memory()
        if sample.total_bytes is None or sample.available_bytes is None:
            state = "unknown"
        else:
            state = (
                "reserve"
                if admission_error(sample, self._additional_reserve_bytes) is not None
                else "normal"
            )
        with self._lock:
            previous_state = self._state
            self._latest = sample
            self._sampled_at = time.time()
            self._state = state
            if sample.used_bytes is not None:
                self._peak_used_bytes = max(
                    sample.used_bytes, self._peak_used_bytes or 0
                )
        logger = logging.getLogger("minimlx.api.memory")
        if state == "reserve" and previous_state != "reserve":
            logger.warning(
                "System memory reserve reached; new inference requests will be rejected"
            )
        elif previous_state == "reserve" and state == "normal":
            logger.info("System memory reserve recovered; inference admission resumed")

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._sample()
            except Exception:
                logging.getLogger("minimlx.api.memory").exception(
                    "Could not sample system memory"
                )
