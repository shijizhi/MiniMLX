"""Stable, transport-independent runtime failure taxonomy."""

from __future__ import annotations

from enum import Enum
from typing import Any


class RuntimeErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    MISSING_COMPONENT = "missing_component"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    HARDWARE_BLOCKED = "hardware_blocked"
    MODEL_LOAD_FAILED = "model_load_failed"
    INFERENCE_FAILED = "inference_failed"
    INTERNAL_ERROR = "internal_error"


class SupportResolutionError(RuntimeError):
    """Raised when a structured support decision does not yield an execution plan."""

    def __init__(self, decision) -> None:
        self.decision = decision
        super().__init__(f"model support decision={decision.status.value}: {decision.reason}")


def error_response(command: str | None, error: Exception) -> dict[str, Any]:
    """Encode stable machine-readable diagnostics while retaining legacy text."""
    from .types import SupportStatus

    details: dict[str, Any] = {}
    retryable = False
    if isinstance(error, SupportResolutionError):
        code = RuntimeErrorCode(error.decision.status.value)
        decision = error.decision
        details = {
            "reason": decision.reason,
            "provider": decision.provider,
            "missing_components": list(decision.missing_components),
            "missing_dependencies": list(decision.missing_dependencies),
            "missing_features": list(decision.missing_features),
        }
        retryable = decision.status in {
            SupportStatus.DEPENDENCY_BLOCKED,
            SupportStatus.HARDWARE_BLOCKED,
        }
    elif isinstance(error, (ValueError, TypeError, KeyError)):
        code = RuntimeErrorCode.INVALID_REQUEST
    elif isinstance(error, NotImplementedError):
        code = RuntimeErrorCode.UNSUPPORTED
    elif command == "load":
        code = RuntimeErrorCode.MODEL_LOAD_FAILED
    elif command in {"infer", "generate", "profile_prefill", "profile_projection_fusion"}:
        code = RuntimeErrorCode.INFERENCE_FAILED
    else:
        code = RuntimeErrorCode.INTERNAL_ERROR

    response: dict[str, Any] = {
        "ok": False,
        "error": f"{type(error).__name__}: {error}",
        "error_code": code.value,
        "retryable": retryable,
        "details": details,
    }
    if command is not None:
        response["command"] = command
    return response
