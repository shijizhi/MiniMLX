from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any


class ModelTask(str, Enum):
    """What an inference request asks the runtime to do."""

    GENERATION = "generation"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_SPEECH = "speech_to_speech"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"


class ModelStructure(str, Enum):
    DENSE = "dense"
    MOE = "moe"
    HYBRID_ATTENTION = "hybrid_attention"
    ENCODER_DECODER = "encoder_decoder"
    DIFFUSION = "diffusion"


class Acceleration(str, Enum):
    MTP = "mtp"
    DFLASH = "dflash"
    SPECULATIVE = "speculative"
    QUANTIZED_KV = "quantized_kv"


class ComponentKind(str, Enum):
    CONFIG = "config"
    TOKENIZER = "tokenizer"
    PROCESSOR = "processor"
    TEXT_WEIGHTS = "text_weights"
    VISION_ENCODER = "vision_encoder"
    PROJECTOR = "projector"
    AUDIO_ENCODER = "audio_encoder"
    AUDIO_DECODER = "audio_decoder"
    VOCODER = "vocoder"
    DRAFT_WEIGHTS = "draft_weights"
    ENGRAM_TABLE = "engram_table"
    SPARSE_INDEXER = "sparse_indexer"
    WEIGHTS_SHARD = "weights_shard"
    OTHER = "other"


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    MISSING_COMPONENT = "missing_component"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    HARDWARE_BLOCKED = "hardware_blocked"


@dataclass(frozen=True)
class ArtifactComponent:
    """A checkpoint file/component, independent of any loader implementation."""

    name: str
    kind: ComponentKind
    path: str
    required: bool = True
    present: bool = True


@dataclass(frozen=True)
class DetectionEvidence:
    """Provenance for a discovered model fact; confidence is in [0, 1]."""

    field: str
    value: str
    source: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("evidence confidence must be between 0 and 1")


@dataclass(frozen=True)
class CapabilityProfile:
    """Orthogonal task, modality, structure and acceleration declarations."""

    tasks: frozenset[ModelTask] = frozenset()
    input_modalities: frozenset[Modality] = frozenset()
    output_modalities: frozenset[Modality] = frozenset()
    structures: frozenset[ModelStructure] = frozenset()
    accelerations: frozenset[Acceleration] = frozenset()


@dataclass(frozen=True)
class AccelerationArtifactRequirement:
    """Checkpoint facts required before an optional acceleration can be routed."""

    model_types: frozenset[str] = frozenset()
    architecture_tokens: tuple[str, ...] = ()
    component_kinds: frozenset[ComponentKind] = frozenset()
    matching_metadata_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_types and not self.architecture_tokens:
            raise ValueError("acceleration artifact requirement needs a model or architecture selector")
        if not self.component_kinds:
            raise ValueError("acceleration artifact requirement needs component kinds")


@dataclass(frozen=True)
class VideoFrame:
    """One timestamped, transport-safe video frame."""

    timestamp_seconds: float
    uri: str | None = None
    data_base64: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_seconds, bool)
            or not isinstance(self.timestamp_seconds, (int, float))
            or not isfinite(self.timestamp_seconds)
            or self.timestamp_seconds < 0
        ):
            raise ValueError("video frame timestamp_seconds must be a non-negative number")
        if (self.uri is None) == (self.data_base64 is None):
            raise ValueError("video frame requires exactly one of uri or data_base64")
        if not isinstance(self.metadata, dict):
            raise ValueError("video frame metadata must be an object")


@dataclass(frozen=True)
class ContentPart:
    """Transport-safe input part; binary content crosses IPC as base64 or URI."""

    modality: Modality
    text: str | None = None
    uri: str | None = None
    data_base64: str | None = None
    data: dict[str, Any] | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    frames: tuple[VideoFrame, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise ValueError("content metadata must be an object")
        if self.modality is Modality.TEXT:
            if (
                self.text is None
                or self.uri is not None
                or self.data_base64 is not None
                or self.data is not None
                or self.frames is not None
            ):
                raise ValueError("text content requires text and cannot contain binary references")
        elif self.modality is Modality.STRUCTURED:
            if (
                not isinstance(self.data, dict)
                or self.text is not None
                or self.uri is not None
                or self.data_base64 is not None
                or self.frames is not None
            ):
                raise ValueError("structured content requires an object payload only")
        elif self.modality is Modality.VIDEO:
            has_reference = (self.uri is None) != (self.data_base64 is None)
            has_frames = isinstance(self.frames, tuple) and bool(self.frames)
            if (
                self.text is not None
                or self.data is not None
                or has_reference == has_frames
            ):
                raise ValueError("video content requires either a reference or a non-empty frame list")
            if self.frames is not None and not all(
                isinstance(frame, VideoFrame) for frame in self.frames
            ):
                raise ValueError("video frames must contain VideoFrame values")
        elif (
            self.text is not None
            or self.data is not None
            or self.frames is not None
            or (self.uri is None) == (self.data_base64 is None)
        ):
            raise ValueError("non-text content requires exactly one of uri or data_base64")


@dataclass(frozen=True)
class InputMessage:
    role: str
    content: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, str):
            raise ValueError("message role must be a string")
        if not self.role.strip():
            raise ValueError("message role cannot be empty")
        if not self.content:
            raise ValueError("message content cannot be empty")


@dataclass(frozen=True)
class InferenceRequest:
    task: ModelTask
    messages: tuple[InputMessage, ...]
    options: dict[str, Any] = field(default_factory=dict)
    accelerations: frozenset[Acceleration] = frozenset()
    stream: bool = False
    session_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, ModelTask):
            raise ValueError("request task must be a ModelTask")
        if not isinstance(self.options, dict):
            raise ValueError("request options must be an object")
        if not isinstance(self.accelerations, frozenset) or not all(
            isinstance(item, Acceleration) for item in self.accelerations
        ):
            raise ValueError("request accelerations must be a frozenset of Acceleration values")
        if not isinstance(self.stream, bool):
            raise ValueError("request stream flag must be a boolean")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise ValueError("session_id must be a string or null")
        if not self.messages:
            raise ValueError("inference request must contain at least one message")


@dataclass(frozen=True)
class GenerationPayload:
    text: str
    prompt_tokens: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddingPayload:
    vectors: list[list[float]]


@dataclass(frozen=True)
class RerankingPayload:
    scores: list[float]
    indices: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptionPayload:
    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AudioPayload:
    data_base64: str
    mime_type: str
    sample_rate: int | None = None


@dataclass(frozen=True)
class SpeechToSpeechPayload:
    text: str | None = None
    audio: AudioPayload | None = None


InferencePayload = (
    GenerationPayload
    | EmbeddingPayload
    | RerankingPayload
    | TranscriptionPayload
    | AudioPayload
    | SpeechToSpeechPayload
)


@dataclass(frozen=True)
class InferenceResult:
    task: ModelTask
    payload: InferencePayload
    usage: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = {
            ModelTask.GENERATION: GenerationPayload,
            ModelTask.EMBEDDING: EmbeddingPayload,
            ModelTask.RERANKING: RerankingPayload,
            ModelTask.SPEECH_TO_TEXT: TranscriptionPayload,
            ModelTask.TEXT_TO_SPEECH: AudioPayload,
            ModelTask.SPEECH_TO_SPEECH: SpeechToSpeechPayload,
        }[self.task]
        if not isinstance(self.payload, expected):
            raise ValueError(
                f"task {self.task.value!r} requires {expected.__name__} payload"
            )


@dataclass(frozen=True)
class RuntimeEnvironment:
    backends: frozenset[str] = frozenset()
    backend_versions: dict[str, str] = field(default_factory=dict)
    hardware: str | None = None
    available_memory_bytes: int | None = None
    features: frozenset[str] = frozenset()
    disabled_optimizations: frozenset[str] = frozenset()
    explicit_optimizations: frozenset[str] = frozenset()
    accelerations: frozenset[Acceleration] = frozenset()
    acceleration_components: dict[Acceleration, str] = field(default_factory=dict)
    acceleration_artifacts: dict[Acceleration, ModelDescriptor] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportDecision:
    status: SupportStatus
    reason: str
    provider: str | None = None
    missing_components: tuple[str, ...] = ()
    missing_dependencies: tuple[str, ...] = ()
    missing_features: tuple[str, ...] = ()
    required_memory_bytes: int | None = None
    available_memory_bytes: int | None = None
    evidence: tuple[DetectionEvidence, ...] = ()


@dataclass(frozen=True)
class ModelDescriptor:
    model_dir: str
    model_type: str
    architecture: str
    tokenizer_family: str
    capabilities: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)
    support: CapabilityProfile = field(default_factory=CapabilityProfile)
    components: tuple[ArtifactComponent, ...] = ()
    evidence: tuple[DetectionEvidence, ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    phase: str
    batch_size: int = 1
    use_custom_kernels: bool = True
    use_continuous_batching: bool = False
    use_speculative_decoding: bool = False
    task: ModelTask | None = None
    task_adapter: str | None = None
    family_adapter: str | None = None
    backend_adapter: str | None = None
    component_adapters: dict[ComponentKind, str] = field(default_factory=dict)
    component_paths: dict[ComponentKind, tuple[str, ...]] = field(default_factory=dict)
    fallback_backend: str | None = None
    accelerations: frozenset[Acceleration] = frozenset()
    acceleration_components: dict[Acceleration, str] = field(default_factory=dict)
    acceleration_adapters: dict[Acceleration, str] = field(default_factory=dict)
    acceleration_component_kinds: dict[Acceleration, ComponentKind] = field(default_factory=dict)
    baseline_provider: str | None = None
    optimization_providers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.acceleration_adapters.keys() != self.acceleration_component_kinds.keys():
            raise ValueError(
                "each acceleration component adapter must declare exactly one component kind"
            )
        if any(not name for name in self.acceleration_adapters.values()):
            raise ValueError("acceleration component adapter names cannot be empty")


@dataclass(frozen=True)
class GenerationResult:
    prompt_tokens: list[int]
    generated_tokens: list[int]
    text: str
    prefill_ms: float
    decode_ms: float
    ttft_ms: float | None = None

    @property
    def decode_tokens_per_second(self) -> float:
        if self.decode_ms <= 0 or not self.generated_tokens:
            return 0.0
        return len(self.generated_tokens) * 1000.0 / self.decode_ms
