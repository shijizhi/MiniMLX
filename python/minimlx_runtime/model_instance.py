"""Loaded model and component resource lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .types import ComponentKind, ExecutionPlan, ModelDescriptor


class ModelInstanceState(str, Enum):
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"


@dataclass
class _Resource:
    name: str
    kind: ComponentKind | None
    value: object
    release: Callable[[object], None] | None


@dataclass
class ModelInstance:
    """Owns one loaded family runtime and any separately loaded components.

    Resources are released in reverse attachment order so dependent components
    (for example a processor or drafter) are released before their base model.
    The runtime adapter may still bundle multiple checkpoint files internally;
    only genuinely separate runtime objects should be attached as components.
    """

    descriptor: ModelDescriptor
    plan: ExecutionPlan
    state: ModelInstanceState = ModelInstanceState.LOADING
    _resources: list[_Resource] = field(default_factory=list, repr=False)
    _primary: object | None = field(default=None, repr=False)

    @property
    def primary(self) -> object | None:
        return self._primary

    def attach_primary(
        self,
        name: str,
        value: object,
        release: Callable[[object], None] | None = None,
    ) -> None:
        if self.state is not ModelInstanceState.LOADING:
            raise RuntimeError("primary model can only be attached while loading")
        if self._primary is not None:
            raise RuntimeError("model instance already has a primary runtime")
        self._primary = value
        self._resources.append(_Resource(name, None, value, release))

    def attach_component(
        self,
        kind: ComponentKind,
        name: str,
        value: object,
        release: Callable[[object], None] | None = None,
    ) -> None:
        if self.state is not ModelInstanceState.LOADING:
            raise RuntimeError("components can only be attached while loading")
        if self._primary is None:
            raise RuntimeError("attach the primary model before its components")
        if any(resource.kind is kind and resource.name == name for resource in self._resources):
            raise RuntimeError(f"duplicate model component resource: {kind.value}/{name}")
        self._resources.append(_Resource(name, kind, value, release))

    def components(self, kind: ComponentKind | None = None) -> tuple[object, ...]:
        return tuple(
            resource.value
            for resource in self._resources
            if resource.kind is not None and (kind is None or resource.kind is kind)
        )

    def mark_ready(self) -> None:
        if self.state is not ModelInstanceState.LOADING or self._primary is None:
            raise RuntimeError("a loading instance with a primary runtime is required")
        self.state = ModelInstanceState.READY

    def unload(self) -> None:
        if self.state is ModelInstanceState.UNLOADED:
            return
        if self.state is ModelInstanceState.UNLOADING:
            raise RuntimeError("model instance unload is already in progress")
        self.state = ModelInstanceState.UNLOADING
        errors: list[Exception] = []
        while self._resources:
            resource = self._resources.pop()
            if resource.release is None:
                continue
            try:
                resource.release(resource.value)
            except Exception as exc:
                errors.append(exc)
        self._primary = None
        self.state = ModelInstanceState.UNLOADED
        if errors:
            raise ExceptionGroup("one or more model resources failed to unload", errors)
