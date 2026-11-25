from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

from .model_catalog import STANDARD_MLX_LM_TEXT_FAMILIES
from .types import (
    Acceleration,
    AccelerationArtifactRequirement,
    CapabilityProfile,
    ComponentKind,
    DetectionEvidence,
    ExecutionPlan,
    InferenceRequest,
    ModelDescriptor,
    ModelStructure,
    ModelTask,
    Modality,
    RuntimeEnvironment,
    SupportDecision,
    SupportStatus,
)


_TASK_OUTPUT_MODALITIES = {
    ModelTask.GENERATION: frozenset({Modality.TEXT}),
    ModelTask.SPEECH_TO_TEXT: frozenset({Modality.TEXT}),
    ModelTask.TEXT_TO_SPEECH: frozenset({Modality.AUDIO}),
    ModelTask.SPEECH_TO_SPEECH: frozenset({Modality.TEXT, Modality.AUDIO}),
}


class ModelAdapter(Protocol):
    name: str

    def can_load(self, descriptor: ModelDescriptor) -> bool: ...

    def load(self, descriptor: ModelDescriptor, plan: ExecutionPlan | None = None): ...


class AdapterRole(str, Enum):
    TASK = "task"
    FAMILY = "family"
    BACKEND = "backend"
    COMPONENT = "component"
    OPTIMIZATION = "optimization"


@dataclass(frozen=True)
class AdapterRegistration:
    """Declarative match contract for one or more composable providers."""

    name: str
    roles: frozenset[AdapterRole]
    tasks: frozenset[ModelTask] = frozenset()
    input_modalities: frozenset[Modality] = frozenset()
    output_modalities: frozenset[Modality] = frozenset()
    families: tuple[str, ...] = ()
    compatible_family_adapters: tuple[str, ...] = ()
    required_model_versions: tuple[str, ...] = ()
    excluded_model_versions: tuple[str, ...] = ()
    backend: str | None = None
    component_kind: ComponentKind | None = None
    required_components: frozenset[ComponentKind] = frozenset()
    accelerations: frozenset[Acceleration] = frozenset()
    acceleration_dependencies: tuple[tuple[Acceleration, frozenset[str]], ...] = ()
    acceleration_component_adapters: tuple[
        tuple[Acceleration, ComponentKind, str], ...
    ] = ()
    acceleration_artifact_requirements: tuple[
        tuple[Acceleration, AccelerationArtifactRequirement], ...
    ] = ()
    acceleration_backend_versions: tuple[tuple[Acceleration, str, str], ...] = ()
    required_dependencies: frozenset[str] = frozenset()
    required_backend_versions: tuple[tuple[str, str], ...] = ()
    required_features: frozenset[str] = frozenset()
    minimum_memory_bytes: int | None = None
    supports_streaming: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        is_component = AdapterRole.COMPONENT in self.roles
        if is_component != (self.component_kind is not None):
            raise ValueError(
                "component providers must declare exactly one component_kind; "
                "non-component providers cannot declare one"
            )
        if any(
            acceleration not in self.accelerations
            for acceleration, _ in self.acceleration_artifact_requirements
        ):
            raise ValueError("acceleration artifact requirements must name a declared acceleration")
        if any(
            acceleration not in self.accelerations
            for acceleration, _, _ in self.acceleration_component_adapters
        ):
            raise ValueError("acceleration component adapters must name a declared acceleration")

    def matches(self, descriptor: ModelDescriptor, request: InferenceRequest) -> bool:
        return self.match_failure(descriptor, request) is None

    def match_failure(
        self, descriptor: ModelDescriptor, request: InferenceRequest
    ) -> tuple[str, ...] | None:
        if (
            request.stream
            and AdapterRole.COMPONENT not in self.roles
            and not self.supports_streaming
        ):
            return ()
        if self.tasks and request.task not in self.tasks:
            return ()
        model_facts = f"{descriptor.model_type} {descriptor.architecture}".lower()
        identity = str(
            descriptor.metadata.get("repository_id")
            or Path(descriptor.model_dir).name
        ).lower()
        identity_compact = re.sub(r"[^a-z0-9]+", "", identity)
        if self.required_model_versions and not any(
            re.sub(r"[^a-z0-9]+", "", version.lower()) in identity_compact
            for version in self.required_model_versions
        ):
            return ()
        if any(
            re.sub(r"[^a-z0-9]+", "", version.lower()) in identity_compact
            for version in self.excluded_model_versions
        ):
            return ()
        if self.families and not any(token.lower() in model_facts for token in self.families):
            return ()
        if self.input_modalities and not {
            part.modality for message in request.messages for part in message.content
        }.issubset(self.input_modalities):
            return ()
        required_outputs = _TASK_OUTPUT_MODALITIES.get(request.task, frozenset())
        if self.output_modalities and not required_outputs.issubset(self.output_modalities):
            return ()
        profile = descriptor.support
        if profile.tasks and request.task not in profile.tasks:
            return ()
        requested_modalities = {
            part.modality for message in request.messages for part in message.content
        }
        if profile.input_modalities and not requested_modalities.issubset(
            profile.input_modalities
        ):
            return ()
        if profile.output_modalities and not required_outputs.issubset(
            profile.output_modalities
        ):
            return ()
        present = {part.kind for part in descriptor.components if part.present}
        missing = self.required_components - present
        return tuple(sorted(kind.value for kind in missing)) if missing else None


class ProviderRegistry:
    """Registry for task/family/backend/component providers."""

    def __init__(self, registrations: list[AdapterRegistration] | None = None) -> None:
        self._registrations: dict[str, AdapterRegistration] = {}
        for registration in registrations or ():
            self.register(registration)

    @property
    def registrations(self) -> tuple[AdapterRegistration, ...]:
        return tuple(self._registrations.values())

    def register(self, registration: AdapterRegistration) -> None:
        if registration.name in self._registrations:
            raise ValueError(f"provider already registered: {registration.name}")
        if not registration.roles:
            raise ValueError("provider must declare at least one adapter role")
        self._registrations[registration.name] = registration

    def candidates(
        self,
        role: AdapterRole,
        descriptor: ModelDescriptor,
        request: InferenceRequest,
    ) -> list[AdapterRegistration]:
        return sorted(
            (
                registration
                for registration in self._registrations.values()
                if role in registration.roles
                and (role is not AdapterRole.COMPONENT or registration.component_kind is not None)
                and registration.matches(descriptor, request)
            ),
            key=lambda registration: (-registration.priority, registration.name),
        )


class CapabilityResolver:
    """Resolve a request to compatible providers without loading model weights."""

    REQUIRED_ROLES = (AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND)

    def __init__(self, providers: ProviderRegistry) -> None:
        self.providers = providers

    @staticmethod
    def _base_requirements_met(
        registrations: tuple[AdapterRegistration, ...],
        environment: RuntimeEnvironment,
    ) -> bool:
        for registration in registrations:
            if not registration.required_dependencies.issubset(environment.backends):
                return False
            if any(
                environment.backend_versions.get(package) != expected
                for package, expected in registration.required_backend_versions
            ):
                return False
            if not registration.required_features.issubset(environment.features):
                return False
            if registration.minimum_memory_bytes is not None and (
                environment.available_memory_bytes is None
                or environment.available_memory_bytes < registration.minimum_memory_bytes
            ):
                return False
        return True

    def _component_adapters_available(
        self,
        registrations: tuple[AdapterRegistration, ...],
        descriptor: ModelDescriptor,
        request: InferenceRequest,
    ) -> bool:
        required_kinds = {
            kind for registration in registrations for kind in registration.required_components
        }
        for kind in required_kinds:
            if not any(
                AdapterRole.COMPONENT in candidate.roles
                and candidate.component_kind is kind
                and candidate.matches(descriptor, request)
                for candidate in self.providers.registrations
            ):
                return False
        return True

    @staticmethod
    def _acceleration_artifact_matches(
        requirement: AccelerationArtifactRequirement,
        artifact: ModelDescriptor | None,
        target: ModelDescriptor,
    ) -> bool:
        if artifact is None:
            return False
        model_type_matches = artifact.model_type.lower() in requirement.model_types
        architecture = artifact.architecture.lower()
        architecture_matches = any(
            token.lower() in architecture for token in requirement.architecture_tokens
        )
        if not (model_type_matches or architecture_matches):
            return False
        present_kinds = {
            component.kind for component in artifact.components if component.present
        }
        if not requirement.component_kinds.issubset(present_kinds):
            return False
        return all(
            key in target.metadata
            and key in artifact.metadata
            and target.metadata[key] is not None
            and artifact.metadata[key] is not None
            and target.metadata[key] == artifact.metadata[key]
            for key in requirement.matching_metadata_keys
        )

    def resolve(
        self,
        descriptor: ModelDescriptor,
        request: InferenceRequest,
        environment: RuntimeEnvironment,
    ) -> tuple[SupportDecision, ExecutionPlan | None]:
        candidates_by_role: dict[AdapterRole, list[AdapterRegistration]] = {}
        missing_by_role: list[str] = []
        missing_components: set[str] = set()
        for role in self.REQUIRED_ROLES:
            candidates = self.providers.candidates(role, descriptor, request)
            # Explicit MTP selection may use a compatible MTP provider even
            # when its normal version routing prefers another backend.
            if Acceleration.MTP in environment.accelerations:
                for registration in self.providers.registrations:
                    if (role in registration.roles
                            and Acceleration.MTP in registration.accelerations
                            and registration.excluded_model_versions
                            and registration not in candidates):
                        mtp_candidate = replace(registration, excluded_model_versions=())
                        if mtp_candidate.matches(descriptor, request):
                            candidates.append(mtp_candidate)
                candidates.sort(key=lambda item: (-item.priority, item.name))
            if not candidates:
                missing_by_role.append(role.value)
                for registration in self.providers.registrations:
                    if role not in registration.roles:
                        continue
                    failure = registration.match_failure(descriptor, request)
                    if failure:
                        missing_components.update(failure)
            else:
                candidates_by_role[role] = candidates
        if missing_by_role:
            task_known = bool(descriptor.support.tasks)
            missing_status = (
                SupportStatus.MISSING_COMPONENT
                if missing_components
                else (SupportStatus.UNSUPPORTED if task_known else SupportStatus.AMBIGUOUS)
            )
            return (
                SupportDecision(
                    status=missing_status,
                    reason="no compatible provider for required role(s): "
                    + ", ".join(missing_by_role),
                    missing_components=tuple(sorted(missing_components)),
                    evidence=descriptor.evidence,
                ),
                None,
            )

        # Select by backend group rather than enumerating the Cartesian
        # product of all task/family/backend providers. This keeps resolution
        # bounded as registries grow while still considering lower-ranked
        # candidates when their preferred provider's gates are unavailable.
        role_order = self.REQUIRED_ROLES
        backend_groups = {
            registration.backend
            for candidates in candidates_by_role.values()
            for registration in candidates
        }
        structural: list[tuple[AdapterRegistration, ...]] = []
        viable: list[tuple[AdapterRegistration, ...]] = []
        for backend in sorted(backend_groups, key=lambda value: value or ""):
            candidates_for_group = {
                role: [
                    registration
                    for registration in candidates_by_role[role]
                    if registration.backend is None
                    or (backend is not None and registration.backend == backend)
                ]
                for role in role_order
            }
            if any(not candidates for candidates in candidates_for_group.values()):
                continue
            best_structural = tuple(candidates_for_group[role][0] for role in role_order)
            structural.append(best_structural)
            best_viable = tuple(
                next(
                    (
                        registration
                        for registration in candidates_for_group[role]
                        if self._base_requirements_met((registration,), environment)
                        and self._component_adapters_available(
                            (registration,), descriptor, request
                        )
                    ),
                    None,
                )
                for role in role_order
            )
            if all(registration is not None for registration in best_viable):
                viable.append(best_viable)
        if not structural:
            return (
                SupportDecision(
                    status=SupportStatus.AMBIGUOUS,
                    reason="no compatible task/family/backend provider combination",
                    evidence=descriptor.evidence,
                ),
                None,
            )
        rank = lambda combination: (
                -sum(registration.priority for registration in combination),
                tuple(registration.name for registration in combination),
            )
        selected_tuple = min(viable or structural, key=rank)
        selected = dict(zip(role_order, selected_tuple, strict=True))
        optimization_providers = tuple(
            registration.name
            for registration in self.providers.candidates(
                AdapterRole.OPTIMIZATION, descriptor, request
            )
            if (
                not registration.compatible_family_adapters
                or selected[AdapterRole.FAMILY].name
                in registration.compatible_family_adapters
            )
            if registration.name not in environment.disabled_optimizations
            if (
                not registration.accelerations
                or registration.accelerations & environment.accelerations
            )
            if self._base_requirements_met((registration,), environment)
        )

        required_component_kinds = frozenset(
            kind
            for registration in selected.values()
            for kind in registration.required_components
        )
        selected_components: dict[ComponentKind, AdapterRegistration] = {}
        for component_kind in required_component_kinds:
            candidates = [
                registration
                for registration in self.providers.registrations
                if AdapterRole.COMPONENT in registration.roles
                and registration.component_kind is component_kind
                and registration.matches(descriptor, request)
            ]
            if candidates:
                selected_components[component_kind] = sorted(
                    candidates,
                    key=lambda registration: (-registration.priority, registration.name),
                )[0]
        missing_component_adapters = required_component_kinds - selected_components.keys()
        if missing_component_adapters:
            missing_names = tuple(
                sorted(kind.value for kind in missing_component_adapters)
            )
            return (
                SupportDecision(
                    status=SupportStatus.MISSING_COMPONENT,
                    reason="no component provider is registered for required component(s): "
                    + ", ".join(missing_names),
                    missing_components=missing_names,
                    evidence=descriptor.evidence,
                ),
                None,
            )
        selected_registrations = (
            *selected.values(),
            *selected_components.values(),
            *(
                registration
                for registration in self.providers.registrations
                if registration.name in optimization_providers
            ),
        )

        raw_accelerations = request.accelerations
        legacy_accelerations = request.options.get("accelerations")
        if legacy_accelerations is not None:
            if raw_accelerations:
                return (
                    SupportDecision(
                        status=SupportStatus.UNSUPPORTED,
                        reason="specify accelerations in the typed field, not both typed and legacy options",
                        evidence=descriptor.evidence,
                    ),
                    None,
                )
            raw_accelerations = legacy_accelerations
        if isinstance(raw_accelerations, str):
            raw_accelerations = (raw_accelerations,)
        try:
            requested_accelerations = frozenset(
                value if isinstance(value, Acceleration) else Acceleration(value)
                for value in raw_accelerations
            )
        except (TypeError, ValueError) as exc:
            return (
                SupportDecision(
                    status=SupportStatus.UNSUPPORTED,
                    reason=f"invalid or unknown acceleration request: {exc}",
                    evidence=descriptor.evidence,
                ),
                None,
            )
        supported_accelerations = frozenset(
            acceleration
            for registration in selected_registrations
            for acceleration in registration.accelerations
        )
        unsupported_accelerations = requested_accelerations - supported_accelerations
        if unsupported_accelerations:
            return (
                SupportDecision(
                    status=SupportStatus.UNSUPPORTED,
                    reason="selected providers do not support requested acceleration(s): "
                    + ", ".join(sorted(item.value for item in unsupported_accelerations)),
                    evidence=descriptor.evidence,
                ),
                None,
            )
        unavailable_accelerations = requested_accelerations - environment.accelerations
        if unavailable_accelerations:
            return (
                SupportDecision(
                    status=SupportStatus.DEPENDENCY_BLOCKED,
                    reason="requested acceleration component is not available",
                    missing_dependencies=tuple(
                        f"{item.value} drafter/component"
                        for item in sorted(unavailable_accelerations, key=lambda item: item.value)
                    ),
                    evidence=descriptor.evidence,
                ),
                None,
            )
        artifact_requirements = {
            acceleration: requirement
            for registration in selected_registrations
            for acceleration, requirement in registration.acceleration_artifact_requirements
        }
        compatible_environment_accelerations = frozenset(
            acceleration
            for acceleration in environment.accelerations
            if acceleration not in artifact_requirements
            or self._acceleration_artifact_matches(
                artifact_requirements[acceleration],
                environment.acceleration_artifacts.get(acceleration),
                descriptor,
            )
        )
        incompatible_requested_accelerations = (
            requested_accelerations & environment.accelerations
        ) - compatible_environment_accelerations
        if incompatible_requested_accelerations:
            return (
                SupportDecision(
                    status=SupportStatus.DEPENDENCY_BLOCKED,
                    reason="configured acceleration artifact does not match provider requirements",
                    missing_dependencies=tuple(
                        f"{item.value} artifact identity or required components"
                        for item in sorted(
                            incompatible_requested_accelerations,
                            key=lambda item: item.value,
                        )
                    ),
                    evidence=descriptor.evidence,
                ),
                None,
            )
        selected_accelerations = (
            requested_accelerations
            if requested_accelerations
            else supported_accelerations & compatible_environment_accelerations
        )

        acceleration_adapters: dict[Acceleration, str] = {}
        acceleration_component_kinds: dict[Acceleration, ComponentKind] = {}
        acceleration_component_registrations: dict[
            Acceleration, AdapterRegistration
        ] = {}
        declared_acceleration_bindings: dict[
            Acceleration, set[tuple[ComponentKind, str]]
        ] = {}
        for registration in selected_registrations:
            for acceleration, kind, provider_name in registration.acceleration_component_adapters:
                if acceleration in selected_accelerations:
                    declared_acceleration_bindings.setdefault(acceleration, set()).add(
                        (kind, provider_name)
                    )
        conflicting_accelerations = {
            acceleration: bindings
            for acceleration, bindings in declared_acceleration_bindings.items()
            if len(bindings) > 1
        }
        if conflicting_accelerations:
            details = "; ".join(
                f"{acceleration.value}: "
                + ", ".join(
                    f"{kind.value}/{name}" for kind, name in sorted(bindings, key=lambda item: (item[0].value, item[1]))
                )
                for acceleration, bindings in sorted(
                    conflicting_accelerations.items(), key=lambda item: item[0].value
                )
            )
            return (
                SupportDecision(
                    status=SupportStatus.AMBIGUOUS,
                    reason=f"providers selected conflicting acceleration components: {details}",
                    evidence=descriptor.evidence,
                ),
                None,
            )
        declared_acceleration_adapters = {
            acceleration: next(iter(bindings))
            for acceleration, bindings in declared_acceleration_bindings.items()
        }
        for acceleration, (component_kind, provider_name) in declared_acceleration_adapters.items():
            provider = next(
                (
                    item
                    for item in self.providers.registrations
                    if item.name == provider_name
                    and AdapterRole.COMPONENT in item.roles
                    and item.component_kind is component_kind
                    and item.match_failure(descriptor, request) is None
                ),
                None,
            )
            if provider is None:
                return (
                    SupportDecision(
                        status=SupportStatus.MISSING_COMPONENT,
                        reason=(
                            f"acceleration {acceleration.value!r} selected component provider "
                            f"{provider_name!r}, but it is not registered"
                        ),
                        missing_components=(provider_name,),
                        evidence=descriptor.evidence,
                    ),
                    None,
                )
            acceleration_adapters[acceleration] = provider_name
            acceleration_component_kinds[acceleration] = component_kind
            acceleration_component_registrations[acceleration] = provider
        selected_registrations = (
            *selected_registrations,
            *acceleration_component_registrations.values(),
        )

        dependencies = {
            dependency
            for registration in selected_registrations
            for dependency in registration.required_dependencies
        }
        dependencies.update(
            dependency
            for registration in selected_registrations
            for acceleration, acceleration_dependencies in registration.acceleration_dependencies
            if acceleration in selected_accelerations
            for dependency in acceleration_dependencies
        )
        missing_dependencies = tuple(
            sorted(dependency for dependency in dependencies if dependency not in environment.backends)
        )
        if missing_dependencies:
            return (
                SupportDecision(
                    status=SupportStatus.DEPENDENCY_BLOCKED,
                    reason="required backend dependencies are unavailable",
                    missing_dependencies=missing_dependencies,
                    evidence=descriptor.evidence,
                ),
                None,
            )

        version_requirements = {
            (package, expected)
            for registration in selected_registrations
            for package, expected in registration.required_backend_versions
        }
        version_requirements.update(
            (package, expected)
            for registration in selected_registrations
            for acceleration, package, expected in registration.acceleration_backend_versions
            if acceleration in selected_accelerations
        )
        missing_versions = tuple(
            sorted(
                f"{package}=={expected} (found {environment.backend_versions.get(package, 'not installed')})"
                for package, expected in version_requirements
                if environment.backend_versions.get(package) != expected
            )
        )
        if missing_versions:
            return (
                SupportDecision(
                    status=SupportStatus.DEPENDENCY_BLOCKED,
                    reason="required backend versions are unavailable",
                    missing_dependencies=missing_versions,
                    evidence=descriptor.evidence,
                ),
                None,
            )

        missing_features = {
            feature
            for registration in selected_registrations
            for feature in registration.required_features
            if feature not in environment.features
        }
        required_memory = max(
            (
                registration.minimum_memory_bytes or 0
                for registration in selected_registrations
            ),
            default=0,
        ) or None
        memory_blocked = required_memory is not None and (
            environment.available_memory_bytes is None
            or environment.available_memory_bytes < required_memory
        )
        if missing_features or memory_blocked:
            return (
                SupportDecision(
                    status=SupportStatus.HARDWARE_BLOCKED,
                    reason="runtime environment does not satisfy provider hardware requirements",
                    missing_features=tuple(sorted(missing_features)),
                    required_memory_bytes=required_memory,
                    available_memory_bytes=environment.available_memory_bytes,
                    evidence=descriptor.evidence,
                ),
                None,
            )

        backends = {
            registration.backend
            for registration in selected_registrations
            if registration.backend is not None
        }
        if len(backends) > 1:
            return (
                SupportDecision(
                    status=SupportStatus.AMBIGUOUS,
                    reason="selected providers require incompatible backends",
                    evidence=descriptor.evidence,
                ),
                None,
            )

        component_adapters = {
            kind: registration.name for kind, registration in selected_components.items()
        }

        component_paths: dict[ComponentKind, tuple[str, ...]] = {}
        for component in descriptor.components:
            if component.present:
                component_paths.setdefault(component.kind, ())
                component_paths[component.kind] += (component.path,)
        for acceleration in selected_accelerations:
            component_path = environment.acceleration_components.get(acceleration)
            if component_path:
                component_paths.setdefault(ComponentKind.DRAFT_WEIGHTS, ())
                component_paths[ComponentKind.DRAFT_WEIGHTS] += (component_path,)
        family_provider = selected[AdapterRole.FAMILY].name
        baseline_provider = {
            "qwen35-mlx-lm": "mlx-lm-standard",
            "qwen35-vlm": "mlx-vlm-generation",
        }.get(family_provider, family_provider)
        plan = ExecutionPlan(
            phase="inference",
            task=request.task,
            task_adapter=selected[AdapterRole.TASK].name,
            family_adapter=selected[AdapterRole.FAMILY].name,
            backend_adapter=next(iter(backends), selected[AdapterRole.BACKEND].name),
            component_adapters=component_adapters,
            component_paths=component_paths,
            accelerations=selected_accelerations,
            acceleration_components={
                acceleration: environment.acceleration_components[acceleration]
                for acceleration in selected_accelerations
                if acceleration in environment.acceleration_components
            },
            acceleration_adapters=acceleration_adapters,
            acceleration_component_kinds=acceleration_component_kinds,
            baseline_provider=baseline_provider,
            optimization_providers=optimization_providers,
        )
        provider_names = tuple(
            dict.fromkeys(
                (*[item.name for item in selected.values()], *optimization_providers)
            )
        )
        return (
            SupportDecision(
                status=SupportStatus.SUPPORTED,
                reason="compatible provider plan resolved",
                provider=" + ".join(provider_names),
                evidence=descriptor.evidence,
            ),
            plan,
        )


def builtin_provider_registry() -> ProviderRegistry:
    """Built-in provider declarations, kept independent of backend imports."""
    registrations = [
            AdapterRegistration(
                name="qwen35-tokenizer-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.TOKENIZER,
                families=("qwen3_5", "qwen3.5"),
                excluded_model_versions=("qwen3.8",),
                priority=100,
            ),
            AdapterRegistration(
                name="qwen35-text-weights-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.TEXT_WEIGHTS,
                families=("qwen3_5", "qwen3.5"),
                excluded_model_versions=("qwen3.8",),
                priority=100,
            ),
            AdapterRegistration(
                name="qwen35-processor-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.PROCESSOR,
                families=("qwen3_5", "qwen3.5"),
                excluded_model_versions=("qwen3.8",),
                priority=100,
            ),
            AdapterRegistration(
                name="qwen35-vision-encoder-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.VISION_ENCODER,
                families=("qwen3_5", "qwen3.5"),
                excluded_model_versions=("qwen3.8",),
                priority=100,
            ),
            AdapterRegistration(
                name="mlx-vlm-processor-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.PROCESSOR,
                priority=50,
            ),
            AdapterRegistration(
                name="mlx-vlm-tokenizer-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.TOKENIZER,
                priority=50,
            ),
            AdapterRegistration(
                name="mlx-vlm-text-weights-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.TEXT_WEIGHTS,
                priority=50,
            ),
            AdapterRegistration(
                name="mlx-vlm-vision-encoder-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.VISION_ENCODER,
                priority=50,
            ),
            AdapterRegistration(
                name="deepseek41-engram-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.ENGRAM_TABLE,
                required_model_versions=("deepseek-v4.1",),
                priority=120,
            ),
            AdapterRegistration(
                name="deepseek41-sparse-indexer-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.SPARSE_INDEXER,
                required_model_versions=("deepseek-v4.1",),
                priority=120,
            ),
            AdapterRegistration(
                name="deepseek41-dspark-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.DRAFT_WEIGHTS,
                required_model_versions=("deepseek-v4.1",),
                priority=120,
            ),
            AdapterRegistration(
                name="qwen35-mtp-draft-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.DRAFT_WEIGHTS,
                families=("qwen3_5", "qwen3.5"),
                priority=100,
            ),
            AdapterRegistration(
                name="qwen35-dflash-draft-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.DRAFT_WEIGHTS,
                # Component providers are resolved against the target model;
                # draft identity/compatibility is validated separately from
                # the acceleration artifact descriptor below.
                families=("qwen3_5", "qwen3.5"),
                excluded_model_versions=("qwen3.8",),
                priority=100,
            ),
            AdapterRegistration(
                name="dflash-draft-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.DRAFT_WEIGHTS,
                required_model_versions=("qwen3.8",),
                priority=110,
            ),
            AdapterRegistration(
                name="mlx-lm-tokenizer-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.TOKENIZER,
                families=STANDARD_MLX_LM_TEXT_FAMILIES + ("qwen3_5", "qwen3.5"),
                priority=100,
            ),
            AdapterRegistration(
                name="mlx-lm-text-weights-component",
                roles=frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.TEXT_WEIGHTS,
                families=STANDARD_MLX_LM_TEXT_FAMILIES + ("qwen3_5", "qwen3.5"),
                priority=100,
            ),
        ]
    registrations.extend(
        [
            AdapterRegistration(
                name="mlx-lm-standard",
                roles=frozenset(
                    {AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}
                ),
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT}),
                output_modalities=frozenset({Modality.TEXT}),
                families=STANDARD_MLX_LM_TEXT_FAMILIES + ("qwen3_5", "qwen3.5"),
                backend="mlx-lm-standard-runtime",
                required_components=frozenset(
                    {ComponentKind.TOKENIZER, ComponentKind.TEXT_WEIGHTS}
                ),
                required_dependencies=frozenset({"mlx-lm"}),
                supports_streaming=True,
                priority=80,
            ),
            AdapterRegistration(
                name="qwen38-dflash-acceleration",
                roles=frozenset({AdapterRole.OPTIMIZATION}),
                tasks=frozenset({ModelTask.GENERATION}),
                required_model_versions=("qwen3.8",),
                compatible_family_adapters=("mlx-lm-standard",),
                supports_streaming=True,
                accelerations=frozenset({Acceleration.DFLASH}),
                acceleration_dependencies=((Acceleration.DFLASH, frozenset({"dflash"})),),
                acceleration_artifact_requirements=(
                    (
                        Acceleration.DFLASH,
                        AccelerationArtifactRequirement(
                            model_types=frozenset({"dflash", "dflash2"}),
                            architecture_tokens=("dflashdraftmodel", "dflash2draftmodel"),
                            component_kinds=frozenset(
                                {ComponentKind.DRAFT_WEIGHTS, ComponentKind.WEIGHTS_SHARD}
                            ),
                            matching_metadata_keys=("hidden_size", "vocab_size"),
                        ),
                    ),
                ),
                acceleration_component_adapters=(
                    (Acceleration.DFLASH, ComponentKind.DRAFT_WEIGHTS, "dflash-draft-component"),
                ),
                priority=110,
            ),
            AdapterRegistration(
                name="qwen35-mlx-lm",
                roles=frozenset(
                    {AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}
                ),
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT}),
                output_modalities=frozenset({Modality.TEXT}),
                families=("qwen3_5", "qwen3.5"),
                excluded_model_versions=("qwen3.8",),
                backend="qwen35-runtime",
                required_components=frozenset(
                    {ComponentKind.TOKENIZER, ComponentKind.TEXT_WEIGHTS}
                ),
                required_dependencies=frozenset({"mlx-lm"}),
                accelerations=frozenset({Acceleration.MTP, Acceleration.DFLASH}),
                acceleration_backend_versions=((Acceleration.MTP, "mlx-vlm", "0.7.2"),),
                acceleration_artifact_requirements=(
                    (
                        Acceleration.MTP,
                        AccelerationArtifactRequirement(
                            model_types=frozenset({"qwen3_5_mtp"}),
                            architecture_tokens=("qwen3_5mtp",),
                            component_kinds=frozenset(
                                {ComponentKind.DRAFT_WEIGHTS, ComponentKind.WEIGHTS_SHARD}
                            ),
                            matching_metadata_keys=("hidden_size", "vocab_size"),
                        ),
                    ),
                    (
                        Acceleration.DFLASH,
                        AccelerationArtifactRequirement(
                            model_types=frozenset({"dflash", "dflash2"}),
                            architecture_tokens=("dflashdraftmodel", "dflash2draftmodel"),
                            component_kinds=frozenset(
                                {ComponentKind.DRAFT_WEIGHTS, ComponentKind.WEIGHTS_SHARD}
                            ),
                            matching_metadata_keys=("hidden_size", "vocab_size"),
                        ),
                    ),
                ),
                # DFlash uses MLX-LM directly and does not depend on the
                # separate MLX-VLM MTP implementation.
                acceleration_dependencies=(
                    (Acceleration.MTP, frozenset({"mlx-vlm"})),
                    (Acceleration.DFLASH, frozenset({"dflash"})),
                ),
                acceleration_component_adapters=(
                    (
                        Acceleration.MTP,
                        ComponentKind.DRAFT_WEIGHTS,
                        "qwen35-mtp-draft-component",
                    ),
                    (
                        Acceleration.DFLASH,
                        ComponentKind.DRAFT_WEIGHTS,
                        "qwen35-dflash-draft-component",
                    ),
                ),
                supports_streaming=True,
                # Text-only requests must win over the vision-capable
                # provider. Qwen3.x checkpoints advertise vision metadata
                # even when no media is present; Rapid-MLX uses the same
                # explicit text-only lane to avoid MLLM overhead.
                priority=120,
            ),
            AdapterRegistration(
                name="qwen35-performance-overlay",
                roles=frozenset({AdapterRole.OPTIMIZATION}),
                tasks=frozenset({ModelTask.GENERATION}),
                families=("qwen3_5", "qwen3.5"),
                excluded_model_versions=("qwen3.8",),
                supports_streaming=True,
                priority=100,
            ),
            # Vision-language execution is a separate provider so the
            # resolver can require both the VLM dependency and the checkpoint
            # vision artifacts without weakening the text-only route.
            AdapterRegistration(
                name="qwen35-vlm",
                roles=frozenset(
                    {AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}
                ),
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
                output_modalities=frozenset({Modality.TEXT}),
                families=("qwen3_5", "qwen3.5"),
                excluded_model_versions=("qwen3.8",),
                backend="qwen35-vlm",
                required_components=frozenset(
                    {
                        ComponentKind.TOKENIZER,
                        ComponentKind.TEXT_WEIGHTS,
                        ComponentKind.PROCESSOR,
                        ComponentKind.VISION_ENCODER,
                    }
                ),
                required_dependencies=frozenset({"mlx-vlm"}),
                supports_streaming=True,
                priority=110,
            ),
            # Family-neutral VLM route. mlx-vlm owns architecture-specific
            # loading/preprocessing; this registration only requires the
            # checkpoint's actual vision components and shared backend. It
            # is a fallback for checkpoints without a compatible text-only
            # provider. Text-only requests should use the standard MLX-LM
            # route when available; image/video requests match this provider
            # because the text-only route does not accept those modalities.
            AdapterRegistration(
                name="mlx-vlm-generation",
                roles=frozenset(
                    {AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}
                ),
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
                output_modalities=frozenset({Modality.TEXT}),
                backend="mlx-vlm-runtime",
                required_components=frozenset(
                    {
                        ComponentKind.TOKENIZER,
                        ComponentKind.TEXT_WEIGHTS,
                        ComponentKind.PROCESSOR,
                        ComponentKind.VISION_ENCODER,
                    }
                ),
                required_dependencies=frozenset({"mlx-vlm"}),
                excluded_model_versions=("inkling-2-mini",),
                supports_streaming=True,
                priority=70,
            ),
            AdapterRegistration(
                name="qwen38-vlm",
                roles=frozenset(
                    {AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}
                ),
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
                output_modalities=frozenset({Modality.TEXT}),
                families=("qwen4_exp",),
                backend="mlx-vlm-runtime",
                required_components=frozenset(
                    {
                        ComponentKind.TOKENIZER,
                        ComponentKind.TEXT_WEIGHTS,
                        ComponentKind.PROCESSOR,
                        ComponentKind.VISION_ENCODER,
                    }
                ),
                required_dependencies=frozenset({"mlx-vlm"}),
                supports_streaming=True,
                priority=130,
            ),
            AdapterRegistration(
                name="deepseek41-mlx-vlm",
                roles=frozenset(
                    {AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}
                ),
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
                output_modalities=frozenset({Modality.TEXT}),
                required_model_versions=("deepseek-v4.1",),
                backend="deepseek41-mlx-vlm-runtime",
                required_components=frozenset(
                    {
                        ComponentKind.TOKENIZER,
                        ComponentKind.TEXT_WEIGHTS,
                        ComponentKind.PROCESSOR,
                        ComponentKind.VISION_ENCODER,
                        ComponentKind.ENGRAM_TABLE,
                        ComponentKind.SPARSE_INDEXER,
                        ComponentKind.DRAFT_WEIGHTS,
                    }
                ),
                required_dependencies=frozenset({"mlx-vlm"}),
                supports_streaming=True,
                priority=140,
            ),
        ]
    )
    return ProviderRegistry(registrations)


def describe_model(model_dir: str | Path) -> ModelDescriptor:
    path = Path(model_dir).expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"model config must be a JSON object: {config_path}")
    model_type = str(config.get("model_type", "unknown"))
    architectures = config.get("architectures", [])
    if isinstance(architectures, str):
        architectures = [architectures]
    architecture = str(architectures[0] if architectures else model_type)
    identity_marker = path / ".minimlx-repo-id"
    repository_id = (
        identity_marker.read_text().strip()
        if identity_marker.is_file()
        else path.name
    ) or path.name
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = {}
    capabilities: set[str] = set()
    num_experts = text_config.get("num_experts", config.get("num_experts"))
    experts_per_token = text_config.get(
        "num_experts_per_tok", config.get("num_experts_per_tok")
    )
    is_moe = isinstance(num_experts, int) and not isinstance(num_experts, bool) and num_experts > 1
    if is_moe:
        capabilities.add("moe")
    family_facts = f"{model_type} {architecture}".lower()
    is_qwen35 = re.search(r"qwen3[._-]5", family_facts) is not None
    # Qwen3.8-Flash-Next checkpoints have appeared with both the canonical
    # qwen3_next model type and repository names containing qwen3.8.  Keep
    # this fact separate from the Qwen3.5 route so the offload planner can
    # select the correct execution lane without making the generic loader
    # claim support for an unverified architecture.
    is_qwen38_flash_next = (
        re.search(r"qwen3[._-]8", family_facts) is not None
        or "qwen3_next" in family_facts
        or "qwen4_exp" in family_facts
        or "flash-next" in repository_id.lower()
    )
    is_deepseek41_flash = (
        re.search(r"deepseek[._-]?v4[._-]?1", family_facts) is not None
        or "deepseek-v4.1-flash" in repository_id.lower()
    )
    if is_qwen35:
        capabilities.update(
            {"prefill", "decode", "single_request", "hybrid_attention", "recurrent_state"}
        )
    if is_qwen38_flash_next:
        capabilities.update(
            {
                "prefill",
                "decode",
                "single_request",
                "hybrid_attention",
                "recurrent_state",
                "moe",
                "ngram_embedding",
                "ssd_offload_candidate",
            }
        )
    if is_deepseek41_flash:
        capabilities.update(
            {
                "prefill",
                "decode",
                "single_request",
                "causal_encoder_decoder",
                "compressed_sparse_attention",
                "engram_memory",
                "hyper_connections",
                "dspark_mtp",
                "ssd_offload_candidate",
            }
        )
    vision_config_keys = (
        "vision_config", "vit_config", "mm_vision_tower", "vision_tower",
        "vision_model", "visual", "image_encoder", "image_processor",
    )
    has_vision = any(
        value not in (None, False, "", {}, [])
        for key in vision_config_keys
        for value in (config.get(key),)
    )
    if has_vision:
        capabilities.add("vision_language")
    has_video = (path / "video_preprocessor_config.json").is_file()
    if has_video:
        capabilities.add("video_language")
    input_modalities = {Modality.TEXT}
    if has_vision:
        input_modalities.add(Modality.IMAGE)
    if has_video:
        input_modalities.add(Modality.VIDEO)
    profile = CapabilityProfile(
        tasks=frozenset({ModelTask.GENERATION})
        if is_qwen35 or is_qwen38_flash_next or is_deepseek41_flash or has_vision
        else frozenset(),
        input_modalities=frozenset(input_modalities)
        if is_qwen35 or is_qwen38_flash_next or is_deepseek41_flash or has_vision
        else frozenset(),
        output_modalities=frozenset({Modality.TEXT})
        if is_qwen35 or is_qwen38_flash_next or is_deepseek41_flash or has_vision
        else frozenset(),
        structures=frozenset(
            ({ModelStructure.HYBRID_ATTENTION} if is_qwen35 else set())
            | ({ModelStructure.MOE} if is_moe else set())
        ),
        accelerations=frozenset(),
    )
    evidence = [
        DetectionEvidence("model_type", model_type, "config.json:model_type"),
        DetectionEvidence(
            "architecture", architecture, "config.json:architectures[0]"
        ),
        DetectionEvidence(
            "repository_id",
            repository_id,
            ".minimlx-repo-id or checkpoint directory name",
        ),
    ]
    if has_vision:
        evidence.append(
            DetectionEvidence("vision", "present", "config.json:vision_subconfig")
        )
    components = _describe_components(path, config)
    component_kinds = {component.kind for component in components if component.present}
    return ModelDescriptor(
        model_dir=str(path),
        model_type=model_type,
        architecture=architecture,
        tokenizer_family="qwen" if "qwen" in model_type.lower() else "huggingface",
        capabilities=frozenset(capabilities),
        metadata={
            "repository_id": repository_id,
            "has_vision": has_vision,
            "has_video": has_video,
            "has_audio": bool(config.get("audio_config") or config.get("audio")),
            "quantization": config.get("quantization"),
            "dtype": config.get("dtype", text_config.get("dtype")),
            "num_hidden_layers": config.get(
                "num_hidden_layers", text_config.get("num_hidden_layers")
            ),
            "num_attention_heads": config.get(
                "num_attention_heads", text_config.get("num_attention_heads")
            ),
            "num_key_value_heads": config.get(
                "num_key_value_heads", text_config.get("num_key_value_heads")
            ),
            "intermediate_size": config.get(
                "intermediate_size", text_config.get("intermediate_size")
            ),
            "max_position_embeddings": config.get(
                "max_position_embeddings", text_config.get("max_position_embeddings")
            ),
            "hidden_size": config.get("hidden_size", text_config.get("hidden_size")),
            "vocab_size": config.get("vocab_size", text_config.get("vocab_size")),
            "num_experts": num_experts,
            "num_experts_per_tok": experts_per_token,
            "moe_intermediate_size": text_config.get(
                "moe_intermediate_size", config.get("moe_intermediate_size")
            ),
            "shared_expert_intermediate_size": text_config.get(
                "shared_expert_intermediate_size",
                config.get("shared_expert_intermediate_size"),
            ),
            "is_qwen38_flash_next": is_qwen38_flash_next,
            "is_deepseek41_flash": is_deepseek41_flash,
            "has_engram": ComponentKind.ENGRAM_TABLE in component_kinds,
            "has_sparse_indexer": ComponentKind.SPARSE_INDEXER in component_kinds,
            "has_dspark": ComponentKind.DRAFT_WEIGHTS in component_kinds,
            "total_weight_bytes": _checkpoint_weight_bytes(path),
        },
        support=profile,
        components=components,
        evidence=tuple(evidence),
    )


def _describe_components(path: Path, config: dict) -> tuple:
    """Describe checkpoint artifacts from files and the safetensors index."""
    from .types import ArtifactComponent

    components: list[ArtifactComponent] = [
        ArtifactComponent("model config", ComponentKind.CONFIG, "config.json")
    ]
    architecture_config = config.get("architectures", ())
    if isinstance(architecture_config, str):
        architecture_config = (architecture_config,)
    declared_draft_artifact = any(
        marker in str(value).lower()
        for value in (config.get("model_type", ""), *architecture_config)
        for marker in ("mtp", "dflash", "draft", "drafter")
    )
    for name, kind in (
        ("tokenizer.json", ComponentKind.TOKENIZER),
        ("tokenizer_config.json", ComponentKind.TOKENIZER),
        ("special_tokens_map.json", ComponentKind.TOKENIZER),
        ("vocab.json", ComponentKind.TOKENIZER),
        ("vocab.model", ComponentKind.TOKENIZER),
        ("merges.txt", ComponentKind.TOKENIZER),
        ("preprocessor_config.json", ComponentKind.PROCESSOR),
        ("processor_config.json", ComponentKind.PROCESSOR),
        ("processor.json", ComponentKind.PROCESSOR),
        ("image_processor_config.json", ComponentKind.PROCESSOR),
        ("video_preprocessor_config.json", ComponentKind.PROCESSOR),
    ):
        if (path / name).is_file():
            components.append(ArtifactComponent(name, kind, name, required=False))

    weight_files: dict[str, set[str]] = {}
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file() and not (path / "offload_index.json").is_file():
        try:
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
            if not isinstance(weight_map, dict):
                raise ValueError("weight_map must be an object")
            for tensor_name, filename in weight_map.items():
                if not isinstance(tensor_name, str) or not isinstance(filename, str):
                    raise ValueError("weight_map keys and values must be strings")
                shard = Path(filename)
                resolved_shard = (path / shard).resolve()
                if shard.is_absolute() or not resolved_shard.is_relative_to(path.resolve()):
                    raise ValueError(f"weight shard escapes model directory: {filename!r}")
                weight_files.setdefault(filename, set()).add(tensor_name.lower())
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid safetensors index {index_path}: {exc}") from exc
    if not weight_files:
        weight_files.update((file.name, set()) for file in path.glob("*.safetensors"))
        # Monolithic checkpoints often have no index. Read only the
        # safetensors key metadata so vision/text components can still be
        # classified without materializing model weights.
        try:
            from safetensors import safe_open

            for filename in tuple(weight_files):
                resolved = (path / filename).resolve()
                with safe_open(str(resolved), framework="numpy") as handle:
                    weight_files[filename] = {key.lower() for key in handle.keys()}
        except Exception:
            # Discovery must remain metadata-only and tolerant of placeholder,
            # incomplete, or non-safetensors test files. The file remains a
            # weight shard; only optional tensor-role classification is lost.
            pass
    for filename, tensor_names in sorted(weight_files.items()):
        resolved = (path / filename).resolve()
        present = resolved.is_file()
        components.append(
            ArtifactComponent(filename, ComponentKind.WEIGHTS_SHARD, filename, present=present)
        )
        names = " ".join(tensor_names)
        roles: list[ComponentKind] = []
        filename_lower = filename.lower()
        if "engram" in filename_lower or "ngram" in filename_lower:
            roles.append(ComponentKind.ENGRAM_TABLE)
        if "indexer" in filename_lower or "sparse_attention" in filename_lower:
            roles.append(ComponentKind.SPARSE_INDEXER)
        if (
            declared_draft_artifact
            or "mtp" in filename_lower
            or "draft" in filename_lower
            or "dspark" in filename_lower
            or any(
                token in names
                for token in ("mtp.", "mtp_", "drafter.", "draft.")
            )
        ):
            roles.append(ComponentKind.DRAFT_WEIGHTS)
        if any(
            token in names
            for token in (
                "visual", "vision.", "vision_", "vision_model", "image_encoder",
                "image_tower", "patch_embed",
            )
        ):
            roles.append(ComponentKind.VISION_ENCODER)
        if any(token in names for token in ("projector", "merger", "multi_modal_projector")):
            roles.append(ComponentKind.PROJECTOR)
        if any(token in names for token in ("audio_tower", "audio_encoder", "audio.", "vocoder")):
            roles.append(ComponentKind.AUDIO_ENCODER)
        if any(token in names for token in ("language_model.", "embed_tokens", "model.layers.")):
            roles.append(ComponentKind.TEXT_WEIGHTS)
        if not roles:
            roles.append(ComponentKind.TEXT_WEIGHTS)
        for role in roles:
            components.append(
                ArtifactComponent(filename, role, filename, required=role is ComponentKind.TEXT_WEIGHTS, present=present)
            )
    present_kinds = {component.kind for component in components if component.present}
    if any(
        config.get(key)
        for key in (
            "vision_config", "vit_config", "mm_vision_tower", "vision_tower",
            "vision_model", "visual", "image_encoder", "image_processor",
        )
    ):
        if ComponentKind.VISION_ENCODER not in present_kinds:
            components.append(
                ArtifactComponent(
                    "vision encoder weights",
                    ComponentKind.VISION_ENCODER,
                    "<not found in checkpoint index>",
                    present=False,
                )
            )
        if ComponentKind.PROCESSOR not in present_kinds:
            components.append(
                ArtifactComponent(
                    "vision processor metadata",
                    ComponentKind.PROCESSOR,
                    "preprocessor_config.json|processor_config.json",
                    present=False,
                )
            )
    if config.get("audio_config") or config.get("audio"):
        if ComponentKind.AUDIO_ENCODER not in present_kinds:
            components.append(
                ArtifactComponent(
                    "audio encoder weights",
                    ComponentKind.AUDIO_ENCODER,
                    "<not found in checkpoint index>",
                    present=False,
                )
            )
    return tuple(components)


def _checkpoint_weight_bytes(path: Path) -> int:
    """Return the on-disk weight size without opening tensors.

    The value is an accounting hint for offload admission, not an estimate of
    peak runtime memory.  Ignore unrelated files and tolerate incomplete
    downloads so discovery remains useful for model-management screens.
    """
    total = 0
    for pattern in ("*.safetensors", "*.bin", "*.gguf"):
        for weight_path in path.glob(pattern):
            try:
                total += weight_path.stat().st_size
            except OSError:
                continue
    return total


class ModelRegistry:
    def __init__(self, adapters: list[ModelAdapter] | None = None) -> None:
        self.adapters = adapters or []
        self._factories: dict[str, Callable[[], ModelAdapter]] = {}
        for adapter in self.adapters:
            self._factories[adapter.name] = type(adapter)

    def register(
        self,
        adapter: ModelAdapter,
        *,
        factory: Callable[[], ModelAdapter] | None = None,
    ) -> None:
        if any(existing.name == adapter.name for existing in self.adapters):
            raise ValueError(f"model adapter already registered: {adapter.name}")
        self.adapters.append(adapter)
        self._factories[adapter.name] = factory or type(adapter)

    def resolve(self, descriptor: ModelDescriptor) -> ModelAdapter:
        for adapter in self.adapters:
            if adapter.can_load(descriptor):
                return adapter
        names = ", ".join(adapter.name for adapter in self.adapters) or "none"
        raise ValueError(
            f"no adapter can load model_type={descriptor.model_type!r}; "
            f"registered adapters: {names}"
        )

    def create(self, descriptor: ModelDescriptor) -> ModelAdapter:
        """Create an isolated model instance using its registered factory."""
        prototype = self.resolve(descriptor)
        instance = self._factories[prototype.name]()
        if instance.name != prototype.name or not instance.can_load(descriptor):
            raise RuntimeError(
                f"factory for {prototype.name!r} returned an incompatible adapter instance"
            )
        return instance

    def create_named(self, adapter_name: str, descriptor: ModelDescriptor) -> ModelAdapter:
        """Create the loader selected by an already-resolved execution plan."""
        factory = self._factories.get(adapter_name)
        if factory is None:
            raise ValueError(f"no model adapter factory is registered as {adapter_name!r}")
        instance = factory()
        if instance.name != adapter_name or not instance.can_load(descriptor):
            raise RuntimeError(
                f"adapter factory {adapter_name!r} cannot load this checkpoint"
            )
        return instance
