"""Versioned JSON-compatible conversion for the task-oriented runtime contract."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import (
    Acceleration,
    AudioPayload,
    ContentPart,
    EmbeddingPayload,
    GenerationPayload,
    InferenceRequest,
    InferenceResult,
    InputMessage,
    ModelTask,
    Modality,
    RerankingPayload,
    SpeechToSpeechPayload,
    TranscriptionPayload,
    VideoFrame,
)


_PAYLOAD_TYPES = {
    "generation": GenerationPayload,
    "embedding": EmbeddingPayload,
    "reranking": RerankingPayload,
    "speech_to_text": TranscriptionPayload,
    "text_to_speech": AudioPayload,
    "speech_to_speech": SpeechToSpeechPayload,
}
PROTOCOL_VERSION = 1


def inference_request_from_dict(value: dict[str, Any]) -> InferenceRequest:
    """Decode the stable transport DTO without importing an MLX backend."""
    try:
        if not isinstance(value, dict):
            raise TypeError("request root must be an object")
        version = int(value.get("schema_version", PROTOCOL_VERSION))
        if version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported schema_version {version}; expected {PROTOCOL_VERSION}")
        task = ModelTask(value["task"])
        raw_messages = value.get("messages", ())
        if not isinstance(raw_messages, (list, tuple)):
            raise TypeError("messages must be an array")
        raw_options = value.get("options", {})
        if not isinstance(raw_options, dict):
            raise TypeError("options must be an object")
        stream = value.get("stream", False)
        if not isinstance(stream, bool):
            raise TypeError("stream must be a boolean")
        messages = tuple(
            InputMessage(
                role=message["role"],
                content=tuple(
                    ContentPart(
                        modality=Modality(part["modality"]),
                        text=part.get("text"),
                        uri=part.get("uri"),
                        data_base64=part.get("data_base64"),
                        data=part.get("data"),
                        mime_type=part.get("mime_type"),
                        metadata=part.get("metadata", {}),
                        frames=(
                            tuple(
                                VideoFrame(
                                    timestamp_seconds=frame["timestamp_seconds"],
                                    uri=frame.get("uri"),
                                    data_base64=frame.get("data_base64"),
                                    mime_type=frame.get("mime_type"),
                                    metadata=frame.get("metadata", {}),
                                )
                                for frame in part["frames"]
                            )
                            if "frames" in part
                            else None
                        ),
                    )
                    for part in message.get("content", ())
                ),
            )
            for message in raw_messages
        )
        return InferenceRequest(
            task=task,
            messages=messages,
            options=raw_options,
            accelerations=frozenset(
                Acceleration(item) for item in value.get("accelerations", ())
            ),
            stream=stream,
            session_id=value.get("session_id"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid inference request: {exc}") from exc


def inference_result_to_dict(result: InferenceResult) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_VERSION,
        "task": result.task.value,
        "payload_type": result.payload.__class__.__name__,
        "payload": asdict(result.payload),
        "usage": result.usage,
        "metrics": result.metrics,
        "metadata": result.metadata,
    }


def inference_result_from_dict(value: dict[str, Any]) -> InferenceResult:
    """Reconstruct a tagged result and validate task/payload agreement."""
    try:
        if not isinstance(value, dict):
            raise TypeError("result root must be an object")
        version = int(value.get("schema_version", PROTOCOL_VERSION))
        if version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported schema_version {version}; expected {PROTOCOL_VERSION}")
        task = ModelTask(value["task"])
        payload_type = _PAYLOAD_TYPES[task.value]
        declared_payload_type = value.get("payload_type")
        if declared_payload_type is not None and declared_payload_type != payload_type.__name__:
            raise ValueError(
                f"payload_type {declared_payload_type!r} does not match task {task.value!r}"
            )
        raw_payload = value["payload"]
        if not isinstance(raw_payload, dict):
            raise TypeError("result payload must be an object")
        if payload_type is SpeechToSpeechPayload and isinstance(raw_payload.get("audio"), dict):
            raw_payload = {
                **raw_payload,
                "audio": AudioPayload(**raw_payload["audio"]),
            }
        usage = value.get("usage", {})
        metrics = value.get("metrics", {})
        metadata = value.get("metadata", {})
        if not isinstance(usage, dict) or not all(
            isinstance(key, str) and isinstance(item, int) and not isinstance(item, bool)
            for key, item in usage.items()
        ):
            raise TypeError("usage must map string keys to integer counts")
        if not isinstance(metrics, dict) or not all(
            isinstance(key, str) and isinstance(item, (int, float)) and not isinstance(item, bool)
            for key, item in metrics.items()
        ):
            raise TypeError("metrics must map string keys to numeric values")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be an object")
        return InferenceResult(
            task=task,
            payload=payload_type(**raw_payload),
            usage=usage,
            metrics={key: float(item) for key, item in metrics.items()},
            metadata=metadata,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid inference result: {exc}") from exc
