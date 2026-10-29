"""Small OpenAI-compatible HTTP surface for local text/image generation."""

from __future__ import annotations

import json
import hmac
import ipaddress
import logging
import sqlite3
import socket
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from ..runtime import Runtime
from ..errors import SupportResolutionError
from ..types import ContentPart, InferenceRequest, InputMessage, ModelTask, Modality, VideoFrame
from .memory import (
    InferenceAdmission,
    SystemMemoryMonitor,
    estimate_context_memory_bytes,
    max_parallel_requests,
    request_memory_snapshot,
)

DEFAULT_API_MAX_TOKENS = 32768
DEFAULT_MAX_PREFILL_STEP_SIZE = 2048


class _DualStackHTTPServer(HTTPServer):
    """Bind an IPv6 wildcard socket and keep MLX calls on the owner thread."""

    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


# Retain the old import name for existing callers/tests.
_DualStackThreadingHTTPServer = _DualStackHTTPServer


class _ApiTelemetry:
    def __init__(self, history_db_path: str | None = None) -> None:
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._active = 0
        self._requests = 0
        self._failed = 0
        self._rejected_busy = 0
        self._rejected_memory = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._prefill_prompt_tokens = 0
        self._decode_completion_tokens = 0
        self._prefill_ms = 0.0
        self._decode_ms = 0.0
        self._vision_cache_hits = 0
        self._vision_cache_misses = 0
        self._vision_cache_observed = False
        self._recent: deque[dict[str, Any]] = deque(maxlen=30)
        self._history_db_path = None
        if history_db_path:
            from pathlib import Path

            expanded_path = str(Path(history_db_path).expanduser())
            Path(expanded_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(expanded_path) as database:
                database.execute(
                    """CREATE TABLE IF NOT EXISTS usage_history (
                        timestamp INTEGER NOT NULL,
                        model TEXT NOT NULL,
                        status TEXT NOT NULL,
                        duration_ms REAL NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL
                    )"""
                )
                database.execute(
                    "CREATE INDEX IF NOT EXISTS usage_history_timestamp ON usage_history(timestamp)"
                )
            self._history_db_path = expanded_path

    def begin(self) -> None:
        with self._lock:
            self._active += 1

    def record_rejection(self, reason: str) -> None:
        with self._lock:
            if reason == "memory":
                self._rejected_memory += 1
            else:
                self._rejected_busy += 1

    def finish(
        self,
        *,
        model: str,
        started: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prefill_ms: float = 0.0,
        decode_ms: float = 0.0,
        vision_cache_hits: int | None = None,
        vision_cache_misses: int | None = None,
        failed: bool = False,
    ) -> None:
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        with self._lock:
            self._active = max(0, self._active - 1)
            self._requests += 1
            self._failed += int(failed)
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._prefill_ms += prefill_ms
            self._decode_ms += decode_ms
            if prefill_ms > 0:
                self._prefill_prompt_tokens += prompt_tokens
            if decode_ms > 0:
                self._decode_completion_tokens += completion_tokens
            if vision_cache_hits is not None or vision_cache_misses is not None:
                self._vision_cache_observed = True
                self._vision_cache_hits += vision_cache_hits or 0
                self._vision_cache_misses += vision_cache_misses or 0
            self._recent.appendleft({
                "timestamp": int(time.time()),
                "model": model,
                "status": "failed" if failed else "complete",
                "duration_ms": round(duration_ms, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
            history_db_path = self._history_db_path
        if history_db_path:
            try:
                with sqlite3.connect(history_db_path, timeout=5) as database:
                    database.execute(
                        "INSERT INTO usage_history VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            int(time.time()),
                            model,
                            "failed" if failed else "complete",
                            duration_ms,
                            prompt_tokens,
                            completion_tokens,
                        ),
                    )
                    database.execute(
                        "DELETE FROM usage_history WHERE timestamp < ?",
                        (int(time.time()) - 400 * 24 * 60 * 60,),
                    )
            except sqlite3.Error:
                logging.getLogger("minimlx.api").exception("Could not persist usage history")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at": int(self.started_at),
                "uptime_seconds": int(max(0.0, time.time() - self.started_at)),
                "scope": "current server process",
                "active_requests": self._active,
                "requests": self._requests,
                "failed_requests": self._failed,
                "rejected_busy_requests": self._rejected_busy,
                "rejected_memory_requests": self._rejected_memory,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "prefill_tps": round(self._prefill_prompt_tokens * 1000.0 / self._prefill_ms, 1)
                if self._prefill_ms > 0
                else None,
                "generation_tps": round(self._decode_completion_tokens * 1000.0 / self._decode_ms, 1)
                if self._decode_ms > 0
                else None,
                "vision_cache_hits": self._vision_cache_hits
                if self._vision_cache_observed
                else None,
                "vision_cache_misses": self._vision_cache_misses
                if self._vision_cache_observed
                else None,
                "vision_cache_hit_rate": round(
                    self._vision_cache_hits
                    / max(1, self._vision_cache_hits + self._vision_cache_misses)
                    * 100.0,
                    1,
                )
                if self._vision_cache_observed
                else None,
                "recent_requests": list(self._recent),
            }


def _message_content(value: Any) -> tuple[ContentPart, ...]:
    if isinstance(value, str):
        return (ContentPart(Modality.TEXT, text=value),)
    if not isinstance(value, list):
        raise ValueError("message content must be a string or an array")
    parts = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("message content parts must be objects")
        if item.get("type") == "text":
            parts.append(ContentPart(Modality.TEXT, text=str(item.get("text", ""))))
        elif item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            uri = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(uri, str) or not uri:
                raise ValueError("image_url.url must be a non-empty string")
            parts.append(ContentPart(Modality.IMAGE, uri=uri))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if not isinstance(source, dict) or source.get("type") != "base64":
                raise ValueError("Anthropic images must use a base64 source")
            media_type = str(source.get("media_type", "image/png"))
            data = source.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError("Anthropic image source requires base64 data")
            parts.append(
                ContentPart(
                    Modality.IMAGE,
                    data_base64=data,
                    mime_type=media_type,
                )
            )
        elif item.get("type") in {"video", "video_url", "input_video"}:
            video_value = item.get("video_url", item.get("video"))
            if isinstance(video_value, dict):
                video_value = video_value.get("url")
            if isinstance(video_value, str) and video_value:
                parts.append(ContentPart(Modality.VIDEO, uri=video_value))
                continue
            raw_frames = item.get("frames")
            if not isinstance(raw_frames, list) or not raw_frames:
                raise ValueError("video content requires a local video URL or non-empty frames array")
            frames = []
            for frame in raw_frames:
                if not isinstance(frame, dict):
                    raise ValueError("video frames must be objects")
                image_url = frame.get("image_url", frame.get("url"))
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                timestamp = frame.get("timestamp_seconds", frame.get("timestamp"))
                if not isinstance(image_url, str) or not image_url:
                    raise ValueError("each video frame requires image_url")
                if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
                    raise ValueError("each video frame requires a numeric timestamp_seconds")
                if image_url.startswith("data:") and ";base64," in image_url:
                    header, payload = image_url.split(",", 1)
                    mime_type = header[5:].split(";", 1)[0]
                    frames.append(VideoFrame(timestamp, data_base64=payload, mime_type=mime_type))
                elif image_url.startswith("file://"):
                    frames.append(VideoFrame(timestamp, uri=image_url))
                else:
                    raise ValueError("video frame image_url must be a data URI or file:// URL")
            parts.append(ContentPart(Modality.VIDEO, frames=tuple(frames)))
        else:
            raise ValueError(f"unsupported chat content type: {item.get('type')!r}")
    if not parts:
        raise ValueError("message content array cannot be empty")
    return tuple(parts)


def _infer(
    runtime: Runtime,
    body: dict[str, Any],
    default_max_tokens: int = DEFAULT_API_MAX_TOKENS,
):
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty array")
    messages = tuple(
        InputMessage(str(message.get("role", "user")), _message_content(message.get("content", "")))
        for message in raw_messages
        if isinstance(message, dict)
    )
    options = {
        key: body[key]
        for key in (
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "ignore_eos",
            "context_tokens",
            "max_context_tokens",
            "prefill_step_size",
            "chat_template_kwargs",
        )
        if key in body
    }
    if "max_completion_tokens" in options and "max_tokens" not in options:
        options["max_tokens"] = options.pop("max_completion_tokens")
    if "ignore_eos" in options and not isinstance(options["ignore_eos"], bool):
        raise ValueError("ignore_eos must be a boolean")
    options.setdefault("max_tokens", default_max_tokens)
    request = InferenceRequest(
        task=ModelTask.GENERATION,
        messages=messages,
        options=options,
        stream=bool(body.get("stream", False)),
    )
    if not request.stream:
        return request, runtime.infer(request)
    return request, runtime.infer_stream(request)


def _anthropic_request(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    system = body.get("system")
    converted = []
    if isinstance(system, str) and system:
        converted.append({"role": "system", "content": system})
    elif isinstance(system, list):
        system_text = "\n".join(
            str(block.get("text", ""))
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if system_text:
            converted.append({"role": "system", "content": system_text})
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("messages must contain objects")
        converted.append(
            {"role": str(message.get("role", "user")), "content": message.get("content", "")}
        )
    return {**body, "messages": converted}


def _prometheus_metrics(snapshot: dict[str, Any]) -> str:
    metrics = {
        "minimlx_requests_total": snapshot["requests"],
        "minimlx_requests_active": snapshot["active_requests"],
        "minimlx_requests_failed_total": snapshot["failed_requests"],
        "minimlx_requests_rejected_busy_total": snapshot.get("rejected_busy_requests", 0),
        "minimlx_requests_rejected_memory_total": snapshot.get("rejected_memory_requests", 0),
        "minimlx_prompt_tokens_total": snapshot["prompt_tokens"],
        "minimlx_completion_tokens_total": snapshot["completion_tokens"],
        "minimlx_prefill_tokens_per_second": snapshot["prefill_tps"] or 0,
        "minimlx_generation_tokens_per_second": snapshot["generation_tps"] or 0,
        "minimlx_uptime_seconds": snapshot["uptime_seconds"],
    }
    descriptions = {
        "minimlx_requests_total": "Completed and failed requests handled by this process.",
        "minimlx_requests_active": "Requests currently being processed.",
        "minimlx_requests_failed_total": "Failed API requests handled by this process.",
        "minimlx_requests_rejected_busy_total": "Requests rejected because the inference slot limit was reached.",
        "minimlx_requests_rejected_memory_total": "Requests rejected to preserve the system memory reserve.",
        "minimlx_prompt_tokens_total": "Prompt tokens processed by this process.",
        "minimlx_completion_tokens_total": "Completion tokens generated by this process.",
        "minimlx_prefill_tokens_per_second": "Mean prefill throughput for this process.",
        "minimlx_generation_tokens_per_second": "Mean generation throughput for this process.",
        "minimlx_uptime_seconds": "API process uptime in seconds.",
    }
    counters = {
        "minimlx_requests_total",
        "minimlx_requests_failed_total",
        "minimlx_requests_rejected_busy_total",
        "minimlx_requests_rejected_memory_total",
        "minimlx_prompt_tokens_total",
        "minimlx_completion_tokens_total",
    }
    return "\n".join(
        f"# HELP {name} {descriptions[name]}\n# TYPE {name} {'counter' if name in counters else 'gauge'}\n{name} {value}"
        for name, value in metrics.items()
    ) + "\n"


def _finish_reason(request: InferenceRequest, completion_tokens: int) -> str:
    """Report token-cap exhaustion distinctly from a model-ended response."""
    max_tokens = request.options.get("max_tokens")
    if max_tokens is not None and completion_tokens >= int(max_tokens):
        return "length"
    return "stop"


def _model_id(runtime: Runtime, alias: str | None = None) -> str:
    if alias:
        return alias
    model_dir = getattr(getattr(runtime, "descriptor", None), "model_dir", "minimlx")
    path = Path(model_dir).expanduser()

    # Match oMLX's public model ID: use the model's directory leaf (or an
    # explicit alias), not the publisher-prefixed flattened download folder.
    try:
        repo_id = (path / ".minimlx-repo-id").read_text(encoding="utf-8").strip()
    except OSError:
        repo_id = ""
    if repo_id and "/" in repo_id:
        return repo_id.rsplit("/", 1)[-1]

    # Hugging Face cache snapshots encode the repository as
    # models--publisher--name/snapshots/<revision>.
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        return path.parent.parent.name.removeprefix("models--").split("--", 1)[-1]

    name = path.name or "minimlx"
    # MiniMLX's own downloader flattens publisher/repository IDs with `--`.
    # Keep this fallback for older downloads without the identity marker.
    return name.split("--", 1)[-1] if "--" in name else name


def _is_requested_model(runtime: Runtime, requested: Any, model_id: str, alias: str | None) -> bool:
    if not isinstance(requested, str) or not requested.strip():
        return False
    requested = requested.strip().casefold()
    path = Path(getattr(getattr(runtime, "descriptor", None), "model_dir", ""))
    candidates = {model_id.casefold(), path.name.casefold()}
    if alias:
        candidates.add(alias.casefold())
    try:
        repo_id = (path / ".minimlx-repo-id").read_text(encoding="utf-8").strip()
    except OSError:
        repo_id = ""
    if repo_id:
        candidates.update((repo_id.casefold(), repo_id.rsplit("/", 1)[-1].casefold()))

    # oMLX accepts provider-qualified IDs such as publisher/model by resolving
    # the final path component against its local model ID.
    requested_leaf = requested.rsplit("/", 1)[-1]
    return requested in candidates or requested_leaf in candidates


def _apply_model_defaults(
    body: dict[str, Any],
    default_options: dict[str, Any] | None,
    force_sampling: bool,
    default_max_tokens: int | None = None,
) -> None:
    if force_sampling:
        for option in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            body.pop(option, None)
    if default_options:
        for option, value in default_options.items():
            body.setdefault(option, value)
    configured_context = (default_options or {}).get("max_context_tokens")
    try:
        context_limit = int(configured_context) if configured_context is not None else None
    except (TypeError, ValueError):
        context_limit = None
    if (
        default_max_tokens is not None
        and "max_tokens" not in body
        and "max_completion_tokens" not in body
    ):
        body["max_tokens"] = (
            min(default_max_tokens, context_limit)
            if context_limit is not None and context_limit > 0
            else default_max_tokens
        )
    if context_limit is not None and context_limit > 0:
        for option in ("max_context_tokens", "context_tokens"):
            if option not in body:
                continue
            try:
                requested = int(body[option])
            except (TypeError, ValueError):
                raise ValueError(f"{option} must be a positive integer") from None
            if isinstance(body[option], bool) or requested <= 0:
                raise ValueError(f"{option} must be a positive integer")
            if requested > context_limit:
                raise ValueError(
                    f"{option} ({requested}) exceeds the configured service context limit ({context_limit})"
                )
            body[option] = requested

        requested_output = body.get(
            "max_tokens", body.get("max_completion_tokens", default_max_tokens)
        )
        if requested_output is not None:
            try:
                output_limit = int(requested_output)
            except (TypeError, ValueError):
                raise ValueError("max_tokens must be a positive integer") from None
            if isinstance(requested_output, bool) or output_limit <= 0:
                raise ValueError("max_tokens must be a positive integer")
            if output_limit > context_limit:
                raise ValueError(
                    f"max_tokens ({output_limit}) exceeds the configured service context limit ({context_limit})"
                )
            if "max_tokens" in body:
                body["max_tokens"] = output_limit

    if "prefill_step_size" in body:
        try:
            requested_step = int(body["prefill_step_size"])
        except (TypeError, ValueError):
            raise ValueError("prefill_step_size must be a positive integer") from None
        if isinstance(body["prefill_step_size"], bool) or requested_step <= 0:
            raise ValueError("prefill_step_size must be a positive integer")
        body["prefill_step_size"] = min(
            requested_step, DEFAULT_MAX_PREFILL_STEP_SIZE
        )


def serve_http(
    runtime: Runtime,
    host: str = "127.0.0.1",
    port: int = 8000,
    default_max_tokens: int = DEFAULT_API_MAX_TOKENS,
    api_key: str | None = None,
    default_options: dict[str, Any] | None = None,
    log_level: str = "INFO",
    log_file_path: str | None = None,
    history_db_path: str | None = None,
    model_alias: str | None = None,
    force_sampling: bool = False,
) -> None:
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if not loopback and not api_key:
        raise ValueError("non-loopback API binding requires an API key")
    if history_db_path:
        from pathlib import Path

        history_db_path = str(Path(history_db_path).expanduser())
    model_id = _model_id(runtime, model_alias)
    telemetry = _ApiTelemetry(history_db_path)
    configured_context = (default_options or {}).get("max_context_tokens", 0)
    try:
        configured_context = int(configured_context)
    except (TypeError, ValueError):
        configured_context = 0
    parallel_limit = max_parallel_requests()
    estimated_request_memory = (
        estimate_context_memory_bytes(runtime, configured_context) * parallel_limit
    )
    inference_admission = InferenceAdmission(
        parallel_limit, additional_reserve_bytes=estimated_request_memory
    )
    memory_monitor = SystemMemoryMonitor(
        additional_reserve_bytes=estimated_request_memory
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if log_file_path:
        from pathlib import Path

        log_path = Path(log_file_path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        class _JsonLogFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps(
                    {
                        "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "exception": self.formatException(record.exc_info)
                        if record.exc_info
                        else None,
                    },
                    ensure_ascii=False,
                )

        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path.resolve()
            for handler in root_logger.handlers
        ):
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(_JsonLogFormatter())
            root_logger.addHandler(handler)
    logger = logging.getLogger("minimlx.api")
    logger.info("MiniMLX API started (model=%s)", model_id)

    def log_model_call(
        path: str,
        status: str,
        started: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: str | None = None,
    ) -> None:
        fields = [
            "model_call",
            f"path={path}",
            f"model={model_id}",
            f"status={status}",
            f"duration_ms={(time.perf_counter() - started) * 1000:.1f}",
            f"prompt_tokens={prompt_tokens}",
            f"completion_tokens={completion_tokens}",
        ]
        if error:
            fields.append(f"error={error!r}")
        logger.info("%s", " ".join(fields))

    class Handler(BaseHTTPRequestHandler):
        server_version = "MiniMLX/0.1"

        def log_message(self, _format: str, *args: object) -> None:
            # Health, telemetry, metrics, and model-list polling are internal
            # UI traffic. Model calls are logged explicitly below with useful
            # inference details instead of noisy access-log lines.
            return

        def _json(
            self,
            status: int,
            value: dict[str, Any],
            headers: dict[str, str] | None = None,
        ) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            for name, header_value in (headers or {}).items():
                self.send_header(name, header_value)
            self.end_headers()
            self.wfile.write(payload)

        def _text(self, status: int, content_type: str, value: str) -> None:
            payload = value.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            if not api_key:
                return True
            bearer = self.headers.get("Authorization", "")
            token = bearer.removeprefix("Bearer ").strip()
            if hmac.compare_digest(self.headers.get("x-api-key", ""), api_key) or hmac.compare_digest(token, api_key):
                return True
            self._json(401, {"error": {"message": "invalid or missing API key", "type": "authentication_error"}})
            return False

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"ok": True, "model": model_id})
            elif self.path == "/metrics":
                if not self._authorized():
                    return
                self._text(200, "text/plain; version=0.0.4; charset=utf-8", _prometheus_metrics(telemetry.snapshot()))
            elif self.path == "/telemetry":
                if not self._authorized():
                    return
                snapshot = telemetry.snapshot()
                snapshot["memory"] = request_memory_snapshot(
                    runtime, estimated_request_memory
                )
                snapshot["memory"]["monitor"] = memory_monitor.snapshot()
                snapshot["max_parallel_requests"] = max_parallel_requests()
                self._json(200, snapshot)
            elif self.path == "/usage":
                if not self._authorized():
                    return
                if not history_db_path:
                    self._json(200, {"enabled": False, "hours": []})
                    return
                with sqlite3.connect(history_db_path) as database:
                    rows = database.execute(
                        """SELECT model, (timestamp / 3600) * 3600, count(*),
                                  sum(prompt_tokens), sum(completion_tokens)
                           FROM usage_history WHERE timestamp >= ?
                           GROUP BY model, 2 ORDER BY 2, model""",
                        (int(time.time()) - 400 * 24 * 60 * 60,),
                    ).fetchall()
                self._json(200, {"enabled": True, "hours": [
                    {"model": row[0], "hour": row[1], "requests": row[2], "prompt_tokens": row[3] or 0, "completion_tokens": row[4] or 0}
                    for row in rows
                ]})
            elif self.path == "/v1/models":
                if not self._authorized():
                    return
                self._json(200, {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "minimlx"}]})
            else:
                self._json(404, {"error": {"message": "not found", "type": "not_found"}})

        def do_POST(self) -> None:
            if self.path not in ("/v1/chat/completions", "/v1/completions", "/v1/messages"):
                self._json(404, {"error": {"message": "not found", "type": "not_found"}})
                return
            if not self._authorized():
                return
            admission = inference_admission.acquire()
            if not admission.allowed:
                telemetry.record_rejection(
                    "busy" if admission.code == "server_busy" else "memory"
                )
                if admission.code == "memory_check_unavailable":
                    logging.getLogger("minimlx.api.memory").warning(
                        "Inference was refused because system memory could not be verified"
                    )
                error_type = (
                    "overloaded_error"
                    if self.path == "/v1/messages"
                    else "server_error"
                )
                self._json(
                    503,
                    {"error": {
                        "message": admission.message,
                        "type": error_type,
                        "code": admission.code,
                    }},
                    {"Retry-After": admission.retry_after or "10"},
                )
                return
            started = time.perf_counter()
            telemetry.begin()
            recorded = False
            prompt_tokens = completion_tokens = 0
            prefill_ms = decode_ms = 0.0
            vision_cache_hits = vision_cache_misses = None
            response_started = False
            is_anthropic = False
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 128 * 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 128 MiB")
                body = json.loads(self.rfile.read(content_length))
                requested_model = body.get("model")
                if requested_model is not None and not _is_requested_model(
                    runtime, requested_model, model_id, model_alias
                ):
                    raise ValueError(
                        f"unknown model {requested_model!r}; use the model ID "
                        f"returned by GET /v1/models (currently {model_id!r})"
                    )
                is_anthropic = self.path == "/v1/messages"
                is_completion = self.path == "/v1/completions"
                if is_completion:
                    prompt = body.get("prompt", "")
                    body["messages"] = [{"role": "user", "content": prompt}]
                elif is_anthropic:
                    body = _anthropic_request(body)
                _apply_model_defaults(
                    body, default_options, force_sampling, default_max_tokens
                )
                request, result = _infer(runtime, body, default_max_tokens)
                stream_options = body.get("stream_options", {})
                if not isinstance(stream_options, dict):
                    raise ValueError("stream_options must be an object")
                include_usage = stream_options.get("include_usage", False)
                if not isinstance(include_usage, bool):
                    raise ValueError("stream_options.include_usage must be a boolean")
                response_id = "chatcmpl-" + uuid.uuid4().hex
                response_created = int(time.time())
                if request.stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    response_started = True
                    if not is_anthropic and not is_completion:
                        role_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": response_created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                        }
                        self.wfile.write(("data: " + json.dumps(role_chunk) + "\n\n").encode())
                        self.wfile.flush()
                    if is_anthropic:
                        self.wfile.write(("event: message_start\ndata: " + json.dumps({"type": "message_start", "message": {"id": response_id, "type": "message", "role": "assistant", "model": model_id, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}}) + "\n\n").encode())
                        self.wfile.write(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
                    stream_done = False
                    for event in result:
                        text = event.get("delta") or ""
                        token_ids = event.get("token_ids") or []
                        completion_tokens += len(token_ids)
                        if event.get("event") == "done":
                            stream_done = True
                            usage = event.get("usage") or {}
                            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                            completion_tokens = int(
                                usage.get("completion_tokens", completion_tokens)
                            )
                            metrics = event.get("metrics") or {}
                            prefill_ms = float(metrics.get("prefill_ms", prefill_ms))
                            decode_ms = float(metrics.get("decode_ms", decode_ms))
                            if "hits" in metrics or "misses" in metrics:
                                vision_cache_hits = int(metrics.get("hits", 0))
                                vision_cache_misses = int(metrics.get("misses", 0))
                        if text:
                            if is_anthropic:
                                packet = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
                                self.wfile.write(("event: content_block_delta\ndata: " + json.dumps(packet, ensure_ascii=False) + "\n\n").encode())
                            else:
                                choice = {"index": 0, "text": text, "finish_reason": None} if is_completion else {"index": 0, "delta": {"content": text}, "finish_reason": None}
                                chunk = {"id": response_id, "object": "text_completion" if is_completion else "chat.completion.chunk", "created": response_created, "model": model_id, "choices": [choice]}
                                self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
                            self.wfile.flush()
                    if not stream_done:
                        raise RuntimeError("stream ended before its completion event")
                    finish_reason = _finish_reason(request, completion_tokens)
                    if is_anthropic:
                        stop_reason = "max_tokens" if finish_reason == "length" else "end_turn"
                        self.wfile.write(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
                        packet = {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": completion_tokens}}
                        self.wfile.write(("event: message_delta\ndata: " + json.dumps(packet) + "\n\n").encode())
                        self.wfile.write(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
                        self.wfile.flush()
                        telemetry.finish(model=model_id, started=started, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, prefill_ms=prefill_ms, decode_ms=decode_ms, vision_cache_hits=vision_cache_hits, vision_cache_misses=vision_cache_misses)
                        log_model_call(self.path, "complete", started, prompt_tokens, completion_tokens)
                        recorded = True
                        return
                    final_choice = {"index": 0, "text": "", "finish_reason": finish_reason} if is_completion else {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    final_chunk = {"id": response_id, "object": "text_completion" if is_completion else "chat.completion.chunk", "created": response_created, "model": model_id, "choices": [final_choice]}
                    self.wfile.write(("data: " + json.dumps(final_chunk) + "\n\n").encode())
                    if include_usage:
                        usage_chunk = {
                            "id": response_id,
                            "object": "text_completion" if is_completion else "chat.completion.chunk",
                            "created": response_created,
                            "model": model_id,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        }
                        self.wfile.write(("data: " + json.dumps(usage_chunk) + "\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    telemetry.finish(
                        model=model_id,
                        started=started,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        prefill_ms=prefill_ms,
                        decode_ms=decode_ms,
                        vision_cache_hits=vision_cache_hits,
                        vision_cache_misses=vision_cache_misses,
                    )
                    log_model_call(self.path, "complete", started, prompt_tokens, completion_tokens)
                    recorded = True
                    return
                text = result.payload.text
                prompt_tokens = int(result.usage.get("prompt_tokens", 0))
                completion_tokens = int(result.usage.get("completion_tokens", 0))
                finish_reason = _finish_reason(request, completion_tokens)
                prefill_ms = float(result.metrics.get("prefill_ms", 0.0))
                decode_ms = float(result.metrics.get("decode_ms", 0.0))
                if "hits" in result.metrics or "misses" in result.metrics:
                    vision_cache_hits = int(result.metrics.get("hits", 0))
                    vision_cache_misses = int(result.metrics.get("misses", 0))
                if is_anthropic:
                    self._json(200, {
                        "id": response_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model_id,
                        "content": [{"type": "text", "text": text}],
                        "stop_reason": "max_tokens" if finish_reason == "length" else "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
                    })
                    telemetry.finish(model=model_id, started=started, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, prefill_ms=prefill_ms, decode_ms=decode_ms, vision_cache_hits=vision_cache_hits, vision_cache_misses=vision_cache_misses)
                    log_model_call(self.path, "complete", started, prompt_tokens, completion_tokens)
                    recorded = True
                    return
                choice = (
                    {"index": 0, "text": text, "finish_reason": finish_reason}
                    if is_completion
                    else {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}
                )
                self._json(200, {
                    "id": response_id,
                    "object": "text_completion" if is_completion else "chat.completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [choice],
                    "usage": {"prompt_tokens": result.usage.get("prompt_tokens", 0), "completion_tokens": result.usage.get("completion_tokens", 0), "total_tokens": sum(result.usage.values())},
                })
                telemetry.finish(
                    model=model_id,
                    started=started,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prefill_ms=prefill_ms,
                    decode_ms=decode_ms,
                    vision_cache_hits=vision_cache_hits,
                    vision_cache_misses=vision_cache_misses,
                )
                log_model_call(self.path, "complete", started, prompt_tokens, completion_tokens)
                recorded = True
            except Exception as exc:
                if not isinstance(
                    exc,
                    (SupportResolutionError, ValueError, TypeError, json.JSONDecodeError),
                ):
                    logging.getLogger("minimlx.api").exception(
                        "Unhandled API request error (path=%s, model=%s)",
                        self.path,
                        model_id,
                    )
                log_model_call(self.path, "failed", started, prompt_tokens, completion_tokens, str(exc))
                if response_started:
                    error_code = (
                        "memory_pressure"
                        if "memory" in str(exc).lower()
                        else "inference_interrupted"
                    )
                    error = {
                        "message": str(exc),
                        "type": "server_error",
                        "code": error_code,
                    }
                    try:
                        if is_anthropic:
                            anthropic_error = {
                                "type": "overloaded_error"
                                if error_code == "memory_pressure"
                                else "api_error",
                                "message": str(exc),
                            }
                            packet = {"type": "error", "error": anthropic_error}
                            self.wfile.write(
                                ("event: error\ndata: " + json.dumps(packet, ensure_ascii=False) + "\n\n").encode()
                            )
                        else:
                            error_packet = json.dumps({"error": error}, ensure_ascii=False)
                            # Intentionally omit [DONE] so clients cannot count failure as success.
                            self.wfile.write(("event: error\ndata: " + error_packet + "\n\n").encode())
                        self.wfile.flush()
                        self.close_connection = True
                    except OSError:
                        logging.getLogger("minimlx.api").debug(
                            "Client disconnected while reporting an interrupted stream"
                        )
                else:
                    status = (
                        422
                        if isinstance(exc, SupportResolutionError)
                        else 400
                        if isinstance(exc, (ValueError, TypeError, json.JSONDecodeError))
                        else 500
                    )
                    error_type = "invalid_request_error" if status < 500 else "server_error"
                    self._json(status, {"error": {"message": str(exc), "type": error_type}})
            finally:
                if not recorded:
                    telemetry.finish(model=model_id, started=started, failed=True)
                inference_admission.release()

    server_type = _DualStackHTTPServer if host == "::" else HTTPServer
    server = server_type((host, port), Handler)
    logger.info("MiniMLX API listening at http://%s:%s/v1 (model=%s)", host, port, model_id)
    try:
        memory_monitor.start()
        server.serve_forever()
    finally:
        memory_monitor.stop()
        server.server_close()
