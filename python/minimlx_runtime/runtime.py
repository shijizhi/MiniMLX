from __future__ import annotations

import importlib.metadata
import inspect
import os
from dataclasses import replace
from pathlib import Path

from .registry import (
    CapabilityResolver,
    ModelRegistry,
    builtin_provider_registry,
    describe_model,
)
from .model_instance import ModelInstance
from .errors import SupportResolutionError
from .types import (
    Acceleration,
    ComponentKind,
    ContentPart,
    GenerationPayload,
    GenerationResult,
    InferenceRequest,
    InferenceResult,
    InputMessage,
    ModelTask,
    Modality,
    RuntimeEnvironment,
    SupportStatus,
    SupportDecision,
)


def _call_with_optional_kwargs(method, *args, **kwargs):
    """Keep older third-party adapters callable while forwarding supported options."""
    try:
        parameters = inspect.signature(method).parameters.values()
        accepts_arbitrary = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        accepted = {parameter.name for parameter in parameters}
    except (TypeError, ValueError):
        return method(*args, **kwargs)
    if not accepts_arbitrary:
        kwargs = {name: value for name, value in kwargs.items() if name in accepted}
    return method(*args, **kwargs)


class Runtime:
    """Long-lived model runtime used by the future Rust supervisor."""

    def __init__(self) -> None:
        from .adapters import MLXVLMAdapter, Qwen2Adapter, Qwen35Adapter

        qwen35 = Qwen35Adapter()
        mlx_vlm = MLXVLMAdapter()
        qwen2 = Qwen2Adapter()
        self.registry = ModelRegistry([qwen35, mlx_vlm, qwen2])
        self.adapters = {qwen35.name: qwen35}
        self.backend_adapters = {
            "qwen35-runtime": qwen35,
            "mlx-vlm-runtime": mlx_vlm,
            "qwen2-runtime": qwen2,
        }
        # The text and VLM providers share one family implementation and one
        # loaded model instance; the provider names remain distinct so the
        # resolver can enforce VLM-specific dependencies and components.
        self.family_provider_aliases = {
            qwen35.name: frozenset({qwen35.name, "qwen35-vlm"}),
            mlx_vlm.name: frozenset({mlx_vlm.name, "mlx-vlm-generation"}),
            qwen2.name: frozenset({qwen2.name}),
        }
        self.component_loaders = {
            "qwen35-tokenizer-component": lambda adapter, _descriptor, _plan: adapter.tokenizer,
            "qwen35-text-weights-component": lambda adapter, _descriptor, _plan: adapter.model,
            "qwen35-processor-component": lambda adapter, _descriptor, _plan: adapter.processor,
            "qwen35-vision-encoder-component": lambda adapter, _descriptor, _plan: adapter.model,
            "qwen35-mtp-draft-component": lambda adapter, _descriptor, _plan: adapter._mtp_draft_model,
            "qwen35-dflash-draft-component": lambda adapter, _descriptor, _plan: adapter._dflash_draft_model,
            "qwen2-tokenizer-component": lambda adapter, _descriptor, _plan: adapter.tokenizer,
            "qwen2-text-weights-component": lambda adapter, _descriptor, _plan: adapter.model,
            "mlx-vlm-tokenizer-component": lambda adapter, _descriptor, _plan: adapter.tokenizer,
            "mlx-vlm-text-weights-component": lambda adapter, _descriptor, _plan: adapter.model,
            "mlx-vlm-processor-component": lambda adapter, _descriptor, _plan: adapter.processor,
            "mlx-vlm-vision-encoder-component": lambda adapter, _descriptor, _plan: adapter.model,
        }
        self.task_executors = {
            "qwen35-mlx-lm": self._execute_generation,
            "qwen35-vlm": self._execute_generation,
            "qwen2-mlx-lm": self._execute_generation,
            "mlx-vlm-generation": self._execute_generation,
        }
        self.providers = builtin_provider_registry()
        self.resolver = CapabilityResolver(self.providers)
        self.adapter = None
        self._instance: ModelInstance | None = None
        self._runtime_environment: RuntimeEnvironment | None = None
        self.descriptor = None
        self.execution_plan = None

    @staticmethod
    def _release_adapter(adapter) -> None:
        """Release optional family-specific state without masking primary errors."""
        unload = getattr(adapter, "unload", None)
        if unload is not None:
            unload()

    @staticmethod
    def _environment() -> RuntimeEnvironment:
        backends = set()
        versions = {}
        for package in ("mlx-lm", "mlx-vlm", "mlx-audio", "mlx-embeddings", "dflash"):
            try:
                versions[package] = importlib.metadata.version(package)
                backends.add(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        mtp_draft_path = os.environ.get("MINIMLX_MTP_DRAFT_MODEL")
        dflash_draft_path = os.environ.get("MINIMLX_DFLASH_DRAFT_MODEL")
        if mtp_draft_path and dflash_draft_path:
            raise ValueError("configure either an MTP draft or a DFlash draft, not both")
        mtp_draft_dir = None
        mtp_draft_descriptor = None
        dflash_draft_dir = None
        dflash_draft_descriptor = None
        for draft_path, acceleration in (
            (mtp_draft_path, Acceleration.MTP),
            (dflash_draft_path, Acceleration.DFLASH),
        ):
            if not draft_path:
                continue
            candidate = Path(draft_path).expanduser()
            if candidate.is_dir():
                try:
                    draft_descriptor = describe_model(candidate)
                    weight_shards = [
                        component
                        for component in draft_descriptor.components
                        if component.kind is ComponentKind.WEIGHTS_SHARD
                    ]
                    has_draft_weights = any(
                        component.kind is ComponentKind.DRAFT_WEIGHTS
                        and component.present
                        for component in draft_descriptor.components
                    )
                    complete = (
                        weight_shards
                        and all(item.present for item in weight_shards)
                        and has_draft_weights
                    )
                    if complete and acceleration is Acceleration.MTP:
                        mtp_draft_dir, mtp_draft_descriptor = str(candidate.resolve()), draft_descriptor
                    elif complete and acceleration is Acceleration.DFLASH:
                        dflash_draft_dir, dflash_draft_descriptor = str(candidate.resolve()), draft_descriptor
                except (OSError, ValueError):
                    # A configured path is not an available acceleration unless
                    # its model descriptor and complete weight manifest parse.
                    pass
        return RuntimeEnvironment(
            backends=frozenset(backends),
            backend_versions=versions,
            accelerations=frozenset(
                acceleration
                for acceleration, path in (
                    (Acceleration.MTP, mtp_draft_dir),
                    (Acceleration.DFLASH, dflash_draft_dir),
                )
                if path
            ),
            acceleration_components={
                acceleration: path
                for acceleration, path in (
                    (Acceleration.MTP, mtp_draft_dir),
                    (Acceleration.DFLASH, dflash_draft_dir),
                )
                if path
            },
            acceleration_artifacts={
                acceleration: artifact
                for acceleration, artifact in (
                    (Acceleration.MTP, mtp_draft_descriptor),
                    (Acceleration.DFLASH, dflash_draft_descriptor),
                )
                if artifact is not None
            },
        )

    def load(self, model_dir: str | Path) -> None:
        descriptor = describe_model(model_dir)
        loader = self.registry.resolve(descriptor)
        request = InferenceRequest(
            task=ModelTask.GENERATION,
            messages=(
                InputMessage(
                    role="user", content=(ContentPart(Modality.TEXT, text=""),)
                ),
            ),
            # Supplying a draft checkpoint is an explicit request for MTP, not
            # an opportunistic acceleration. If initialization fails, surface
            # the actual MTP error instead of falling back after partially
            # importing MLX-VLM and leaving two MLX binding states behind.
            accelerations=(
                frozenset(
                    {Acceleration.MTP}
                    if os.environ.get("MINIMLX_MTP_DRAFT_MODEL")
                    else {Acceleration.DFLASH}
                )
                if os.environ.get("MINIMLX_MTP_DRAFT_MODEL")
                or os.environ.get("MINIMLX_DFLASH_DRAFT_MODEL")
                else frozenset()
            ),
        )
        runtime_environment = self._environment()
        decision, plan = self.resolver.resolve(descriptor, request, runtime_environment)
        if decision.status is not SupportStatus.SUPPORTED or plan is None:
            raise SupportResolutionError(decision)
        family_provider_aliases = getattr(self, "family_provider_aliases", {})
        if plan.family_adapter not in family_provider_aliases.get(
            loader.name, frozenset({loader.name})
        ):
            raise RuntimeError(
                "resolved family adapter does not match the registered model loader"
            )
        candidate = self.registry.create(descriptor)
        try:
            loaded = candidate.load(descriptor, plan)
            if loaded is None:
                raise RuntimeError(
                    f"family adapter {loader.name!r} returned no loaded instance"
                )
        except Exception as acceleration_error:
            try:
                self._release_adapter(candidate)
            except Exception:
                pass
            # Accelerations discovered from the environment are opportunistic
            # at load time. If one cannot initialize, resolve a fresh plan
            # with those optional components removed and retry the safe base
            # route. Explicitly requested accelerations are never downgraded.
            if not plan.accelerations or request.accelerations:
                raise
            fallback_environment = replace(
                runtime_environment,
                accelerations=runtime_environment.accelerations - plan.accelerations,
                acceleration_components={
                    key: value
                    for key, value in runtime_environment.acceleration_components.items()
                    if key not in plan.accelerations
                },
                acceleration_artifacts={
                    key: value
                    for key, value in runtime_environment.acceleration_artifacts.items()
                    if key not in plan.accelerations
                },
            )
            fallback_decision, fallback_plan = self.resolver.resolve(
                descriptor, request, fallback_environment
            )
            if (
                fallback_decision.status is not SupportStatus.SUPPORTED
                or fallback_plan is None
                or fallback_plan.accelerations
            ):
                raise RuntimeError(
                    "optional acceleration failed to load and no unaccelerated "
                    f"fallback plan is available: {acceleration_error}; "
                    f"fallback resolution: {fallback_decision.reason}"
                ) from acceleration_error
            candidate = self.registry.create(descriptor)
            plan = fallback_plan
            runtime_environment = fallback_environment
            try:
                loaded = candidate.load(descriptor, plan)
                if loaded is None:
                    raise RuntimeError(
                        f"family adapter {loader.name!r} returned no loaded instance"
                    )
            except Exception as fallback_error:
                try:
                    self._release_adapter(candidate)
                except Exception:
                    pass
                raise RuntimeError(
                    "optional acceleration failed and the base backend fallback "
                    f"also failed: acceleration={acceleration_error}; fallback={fallback_error}"
                ) from fallback_error
        instance = ModelInstance(descriptor=descriptor, plan=plan)
        try:
            instance.attach_primary(
                plan.family_adapter,
                loaded,
                release=lambda value: self._release_adapter(value),
            )
            for kind, provider_name in plan.component_adapters.items():
                component_loader = getattr(self, "component_loaders", {}).get(provider_name)
                if component_loader is None:
                    raise RuntimeError(
                        f"execution plan selected component provider without a loader: {provider_name}"
                    )
                component = component_loader(loaded, descriptor, plan)
                if component is None:
                    raise RuntimeError(
                        f"component provider {provider_name!r} did not load {kind.value}"
                    )
                # These Qwen components are views into the family-loaded model;
                # their shared owner is the primary adapter, released last.
                instance.attach_component(kind, provider_name, component)
            for acceleration, provider_name in plan.acceleration_adapters.items():
                component_loader = getattr(self, "component_loaders", {}).get(provider_name)
                if component_loader is None:
                    raise RuntimeError(
                        "execution plan selected acceleration component provider "
                        f"without a loader: {provider_name}"
                    )
                component = component_loader(loaded, descriptor, plan)
                if component is None:
                    raise RuntimeError(
                        f"acceleration provider {provider_name!r} did not load "
                        f"{acceleration.value} component"
                    )
                instance.attach_component(
                    plan.acceleration_component_kinds[acceleration],
                    provider_name,
                    component,
                )
            instance.mark_ready()
        except Exception:
            try:
                if instance.primary is not None:
                    instance.unload()
                else:
                    self._release_adapter(loaded)
            except Exception:
                pass
            raise
        old_adapter = self.adapter
        old_instance = getattr(self, "_instance", None)
        self._instance = instance
        self.adapter = loaded
        self._runtime_environment = runtime_environment
        self.descriptor = descriptor
        self.execution_plan = plan
        self.adapters[plan.family_adapter] = loaded
        self.backend_adapters[plan.backend_adapter] = loaded
        if old_adapter is not None and old_adapter is not loaded:
            if old_instance is not None:
                old_instance.unload()
            else:
                self._release_adapter(old_adapter)

    def unload(self) -> None:
        """Unload the active model while keeping the runtime reusable."""
        old_adapter = self.adapter
        old_instance = getattr(self, "_instance", None)
        self._instance = None
        self.adapter = None
        self._runtime_environment = None
        self.descriptor = None
        self.execution_plan = None
        self.adapters.clear()
        self.backend_adapters.clear()
        if old_instance is not None:
            old_instance.unload()
        elif old_adapter is not None:
            self._release_adapter(old_adapter)

    def resolve_request(self, request: InferenceRequest) -> SupportDecision:
        if self.descriptor is None:
            raise RuntimeError("load() must be called before resolve_request()")
        decision, _ = self.resolver.resolve(
            self.descriptor, request, self._active_environment()
        )
        return decision

    def _active_environment(self) -> RuntimeEnvironment:
        """Keep route resolution tied to the environment used to load the model."""
        return getattr(self, "_runtime_environment", None) or self._environment()

    @staticmethod
    def inspect_model(
        model_dir: str | Path,
        request: InferenceRequest,
        environment: RuntimeEnvironment | None = None,
    ) -> dict:
        """Resolve declared support without loading weights or importing MLX."""
        descriptor = describe_model(model_dir)
        decision, plan = CapabilityResolver(builtin_provider_registry()).resolve(
            descriptor, request, environment or Runtime._environment()
        )
        return {
            "schema_version": 1,
            "descriptor": {
                "model_dir": descriptor.model_dir,
                "model_type": descriptor.model_type,
                "architecture": descriptor.architecture,
                "tokenizer_family": descriptor.tokenizer_family,
                "capabilities": sorted(descriptor.capabilities),
                "metadata": descriptor.metadata,
                "support": {
                    "tasks": sorted(task.value for task in descriptor.support.tasks),
                    "input_modalities": sorted(
                        modality.value for modality in descriptor.support.input_modalities
                    ),
                    "output_modalities": sorted(
                        modality.value for modality in descriptor.support.output_modalities
                    ),
                    "structures": sorted(
                        structure.value for structure in descriptor.support.structures
                    ),
                    "accelerations": sorted(
                        acceleration.value
                        for acceleration in descriptor.support.accelerations
                    ),
                },
                "components": [
                    {
                        "name": item.name,
                        "kind": item.kind.value,
                        "path": item.path,
                        "required": item.required,
                        "present": item.present,
                    }
                    for item in descriptor.components
                ],
                "evidence": [
                    {
                        "field": item.field,
                        "value": item.value,
                        "source": item.source,
                        "confidence": item.confidence,
                    }
                    for item in descriptor.evidence
                ],
            },
            "decision": {
                "status": decision.status.value,
                "reason": decision.reason,
                "provider": decision.provider,
                "missing_components": list(decision.missing_components),
                "missing_dependencies": list(decision.missing_dependencies),
                "missing_features": list(decision.missing_features),
                "required_memory_bytes": decision.required_memory_bytes,
                "available_memory_bytes": decision.available_memory_bytes,
            },
            "plan": (
                {
                    "phase": plan.phase,
                    "task": plan.task.value if plan.task else None,
                    "task_adapter": plan.task_adapter,
                    "family_adapter": plan.family_adapter,
                    "backend_adapter": plan.backend_adapter,
                    "component_adapters": {
                        kind.value: name
                        for kind, name in plan.component_adapters.items()
                    },
                    "fallback_backend": plan.fallback_backend,
                    "accelerations": sorted(
                        acceleration.value for acceleration in plan.accelerations
                    ),
                    "acceleration_components": {
                        acceleration.value: path
                        for acceleration, path in plan.acceleration_components.items()
                    },
                    "acceleration_adapters": {
                        acceleration.value: name
                        for acceleration, name in plan.acceleration_adapters.items()
                    },
                    "acceleration_component_kinds": {
                        acceleration.value: kind.value
                        for acceleration, kind in plan.acceleration_component_kinds.items()
                    },
                    "component_paths": {
                        kind.value: list(paths)
                        for kind, paths in plan.component_paths.items()
                    },
                }
                if plan
                else None
            ),
        }

    @staticmethod
    def _execute_generation(adapter, request: InferenceRequest) -> InferenceResult:
        if request.task is not ModelTask.GENERATION:
            raise NotImplementedError(
                f"task {request.task.value!r} has a contract but no execution provider yet"
            )
        parts = [
            part
            for message in request.messages
            for part in message.content
        ]
        image_parts = [part for part in parts if part.modality is Modality.IMAGE]
        video_parts = [part for part in parts if part.modality is Modality.VIDEO]
        if image_parts or video_parts:
            if any(part.modality not in (Modality.TEXT, Modality.IMAGE, Modality.VIDEO) for part in parts):
                raise NotImplementedError(
                    "VLM generation currently accepts text, image, and video parts only"
                )
            if image_parts and video_parts and getattr(
                getattr(getattr(adapter, "model", None), "config", None),
                "model_type",
                None,
            ) == "qwen3_5":
                raise NotImplementedError(
                    "the mlx-vlm Qwen3.5 backend does not currently support mixing image and video inputs in one request"
                )
            images = image_parts
            max_images = max(1, int(os.environ.get("MINIMLX_MAX_IMAGE_COUNT", "16")))
            if len(images) > max_images:
                raise ValueError(f"request contains {len(images)} images; maximum is {max_images}")
            max_videos = max(1, int(os.environ.get("MINIMLX_MAX_VIDEO_COUNT", "4")))
            if len(video_parts) > max_videos:
                raise ValueError(f"request contains {len(video_parts)} videos; maximum is {max_videos}")
            max_frames = max(1, int(os.environ.get("MINIMLX_MAX_VIDEO_FRAMES", "64")))
            if any(part.frames is not None and len(part.frames) > max_frames for part in video_parts):
                raise ValueError(f"video exceeds MINIMLX_MAX_VIDEO_FRAMES ({max_frames})")
            template_kwargs = request.options.get("chat_template_kwargs", {})
            if not isinstance(template_kwargs, dict):
                raise ValueError("chat_template_kwargs must be an object")
            if request.stream:
                raise NotImplementedError("streaming image generation is not supported yet")
            prompt = Runtime._vlm_messages(request)
            result = _call_with_optional_kwargs(adapter.generate_multimodal,
                prompt,
                images,
                int(request.options.get("max_tokens", 16)),
                request.options.get("prefill_step_size", 2048),
                request.session_id,
                temperature=float(request.options.get("temperature", 0.0)),
                top_p=float(request.options.get("top_p", 1.0)),
                top_k=int(request.options.get("top_k", 0)),
                min_p=float(request.options.get("min_p", 0.0)),
                repetition_penalty=float(request.options.get("repetition_penalty", 1.0)),
                presence_penalty=float(request.options.get("presence_penalty", 0.0)),
                frequency_penalty=float(request.options.get("frequency_penalty", 0.0)),
                video_parts=video_parts,
                chat_template_kwargs=template_kwargs,
            )
            prompt_tokens, generated, text, prefill_ms, decode_ms = result
        else:
            if any(part.modality is not Modality.TEXT for part in parts):
                raise NotImplementedError(
                    "generation provider currently accepts text or image parts only"
                )
            prompt = "\n".join(part.text or "" for part in parts)
            format_messages = getattr(adapter, "format_messages", None)
            if callable(format_messages):
                formatted_prompt = _call_with_optional_kwargs(
                    format_messages,
                    request.messages,
                    chat_template_kwargs=request.options.get("chat_template_kwargs", {}),
                )
                if formatted_prompt is not None:
                    prompt = formatted_prompt
            if getattr(adapter, "_vlm_enabled", False):
                prompt_tokens, generated, text, prefill_ms, decode_ms = (
                    _call_with_optional_kwargs(adapter.generate_multimodal,
                        Runtime._vlm_messages(request),
                        [],
                        int(request.options.get("max_tokens", 16)),
                        int(request.options.get("prefill_step_size", 2048)),
                        request.session_id,
                        request.options.get("context_tokens"),
                        temperature=float(request.options.get("temperature", 0.0)),
                        top_p=float(request.options.get("top_p", 1.0)),
                        top_k=int(request.options.get("top_k", 0)),
                        min_p=float(request.options.get("min_p", 0.0)),
                        repetition_penalty=float(request.options.get("repetition_penalty", 1.0)),
                        presence_penalty=float(request.options.get("presence_penalty", 0.0)),
                        frequency_penalty=float(request.options.get("frequency_penalty", 0.0)),
                        max_context_tokens=request.options.get("max_context_tokens"),
                        chat_template_kwargs=request.options.get("chat_template_kwargs", {}),
                    )
                )
            else:
                prompt_tokens, generated, text, prefill_ms, decode_ms = _call_with_optional_kwargs(adapter.generate,
                    prompt,
                    int(request.options.get("max_tokens", 16)),
                    request.options.get("context_tokens"),
                    int(request.options.get("prefill_step_size", 2048)),
                    request.session_id,
                    temperature=float(request.options.get("temperature", 0.0)),
                    top_p=float(request.options.get("top_p", 1.0)),
                    top_k=int(request.options.get("top_k", 0)),
                    min_p=float(request.options.get("min_p", 0.0)),
                    repetition_penalty=float(request.options.get("repetition_penalty", 1.0)),
                    presence_penalty=float(request.options.get("presence_penalty", 0.0)),
                    frequency_penalty=float(request.options.get("frequency_penalty", 0.0)),
                    max_context_tokens=request.options.get("max_context_tokens"),
                )
        metrics = {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
        }
        cache_metrics = getattr(adapter, "inference_cache_metrics", None)
        if callable(cache_metrics):
            metrics.update(cache_metrics())
        if any(part.modality is Modality.IMAGE for part in parts):
            vision_metrics = getattr(adapter, "vision_feature_cache_metrics", None)
            if callable(vision_metrics):
                metrics.update(vision_metrics())

        return InferenceResult(
            task=request.task,
            payload=GenerationPayload(
                text=text,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated,
            ),
            usage={"prompt_tokens": len(prompt_tokens), "completion_tokens": len(generated)},
            metrics=metrics,
        )

    @staticmethod
    def _vlm_messages(request: InferenceRequest) -> list[dict]:
        """Preserve OpenAI-style message/content ordering for mlx-vlm."""
        messages = []
        for message in request.messages:
            content = []
            for part in message.content:
                if part.modality is Modality.TEXT:
                    content.append({"type": "text", "text": part.text or ""})
                elif part.modality is Modality.IMAGE:
                    # The native checkpoint chat template owns the exact
                    # image-token expansion; do not synthesize <image> text.
                    content.append({"type": "image"})
                elif part.modality is Modality.VIDEO:
                    content.append({"type": "video"})
            if content:
                messages.append({"role": message.role, "content": content})
        return messages

    def infer(self, request: InferenceRequest) -> InferenceResult:
        if self.adapter is None:
            raise RuntimeError("load() must be called before infer()")
        decision, plan = self.resolver.resolve(
            self.descriptor, request, self._active_environment()
        )
        if decision.status is not SupportStatus.SUPPORTED or plan is None:
            raise SupportResolutionError(decision)
        if plan.family_adapter not in self.adapters:
            raise RuntimeError(f"execution plan selected unknown family adapter: {plan.family_adapter}")
        if plan.backend_adapter not in self.backend_adapters:
            raise RuntimeError(f"execution plan selected unknown backend adapter: {plan.backend_adapter}")
        executor = self.task_executors.get(plan.task_adapter)
        if executor is None:
            raise RuntimeError(f"execution plan selected unknown task adapter: {plan.task_adapter}")
        backend = self.backend_adapters[plan.backend_adapter]
        if backend is not self.adapter or backend is not self.adapters[plan.family_adapter]:
            raise RuntimeError("execution plan adapters do not refer to the loaded model instance")
        self.execution_plan = plan
        return executor(backend, request)

    def infer_stream(self, request: InferenceRequest):
        """Stream image-generation deltas through the Python API boundary."""
        if not request.stream:
            raise ValueError("infer_stream requires request.stream=true")
        if request.task is not ModelTask.GENERATION:
            raise NotImplementedError("streaming is only supported for generation")
        if self.adapter is None:
            raise RuntimeError("load() must be called before infer_stream")
        decision, plan = self.resolver.resolve(
            self.descriptor, request, self._active_environment()
        )
        if decision.status is not SupportStatus.SUPPORTED or plan is None:
            raise SupportResolutionError(decision)
        if not any(
            part.modality in (Modality.IMAGE, Modality.VIDEO)
            for message in request.messages
            for part in message.content
        ):
            raise NotImplementedError("streaming text generation is not implemented")
        backend = self.backend_adapters.get(plan.backend_adapter)
        if backend is None or backend is not self.adapter:
            raise RuntimeError("streaming plan selected an unknown backend")
        images = [
            part
            for message in request.messages
            for part in message.content
            if part.modality is Modality.IMAGE
        ]
        videos = [
            part
            for message in request.messages
            for part in message.content
            if part.modality is Modality.VIDEO
        ]
        if len(images) > max(1, int(os.environ.get("MINIMLX_MAX_IMAGE_COUNT", "16"))):
            raise ValueError("image request exceeds MINIMLX_MAX_IMAGE_COUNT")
        text = []
        template_kwargs = request.options.get("chat_template_kwargs", {})
        if not isinstance(template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object")
        max_videos = max(1, int(os.environ.get("MINIMLX_MAX_VIDEO_COUNT", "4")))
        if len(videos) > max_videos:
            raise ValueError(f"request contains {len(videos)} videos; maximum is {max_videos}")
        max_frames = max(1, int(os.environ.get("MINIMLX_MAX_VIDEO_FRAMES", "64")))
        if any(part.frames is not None and len(part.frames) > max_frames for part in videos):
            raise ValueError(f"video exceeds MINIMLX_MAX_VIDEO_FRAMES ({max_frames})")
        for chunk in _call_with_optional_kwargs(backend.stream_multimodal,
            Runtime._vlm_messages(request),
            images,
            int(request.options.get("max_tokens", 16)),
            int(request.options.get("prefill_step_size", 2048)),
            request.session_id,
            temperature=float(request.options.get("temperature", 0.0)),
            top_p=float(request.options.get("top_p", 1.0)),
            top_k=int(request.options.get("top_k", 0)),
            min_p=float(request.options.get("min_p", 0.0)),
            repetition_penalty=float(request.options.get("repetition_penalty", 1.0)),
            presence_penalty=float(request.options.get("presence_penalty", 0.0)),
            frequency_penalty=float(request.options.get("frequency_penalty", 0.0)),
            video_parts=videos,
            chat_template_kwargs=template_kwargs,
        ):
            delta = chunk["text"]
            text.append(delta)
            yield {
                "event": "delta",
                "delta": delta,
                "token_ids": chunk["token_ids"],
            }
        yield {
            "event": "done",
            "text": "".join(text),
            "metrics": backend.vision_feature_cache_metrics(),
        }

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        context_tokens: int | None = None,
        prefill_step_size: int = 2048,
        session_id: str | None = None,
    ) -> GenerationResult:
        result = self.infer(
            InferenceRequest(
                task=ModelTask.GENERATION,
                messages=(
                    InputMessage(
                        role="user", content=(ContentPart(Modality.TEXT, text=prompt),)
                    ),
                ),
                options={
                    "max_tokens": max_tokens,
                    "context_tokens": context_tokens,
                    "prefill_step_size": prefill_step_size,
                },
                session_id=session_id,
            )
        )
        payload = result.payload
        assert isinstance(payload, GenerationPayload)
        return GenerationResult(
            prompt_tokens=payload.prompt_tokens,
            generated_tokens=payload.generated_tokens,
            text=payload.text,
            prefill_ms=result.metrics["prefill_ms"],
            decode_ms=result.metrics["decode_ms"],
        )

    def clear_session(self, session_id: str) -> None:
        if self.adapter is None:
            raise RuntimeError("load() must be called before clear_session")
        self.adapter.clear_session(session_id)
