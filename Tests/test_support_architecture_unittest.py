"""Headless contract tests; these do not import MLX or require Metal."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from minimlx_runtime.inference_protocol import (
    inference_request_from_dict,
    inference_result_from_dict,
    inference_result_to_dict,
)
from minimlx_runtime.registry import (
    AdapterRegistration,
    AdapterRole,
    CapabilityResolver,
    ProviderRegistry,
    builtin_provider_registry,
    describe_model,
)
from minimlx_runtime.types import (
    Acceleration,
    AccelerationArtifactRequirement,
    ArtifactComponent,
    CapabilityProfile,
    ComponentKind,
    ContentPart,
    EmbeddingPayload,
    ExecutionPlan,
    GenerationPayload,
    GenerationResult,
    InferenceRequest,
    InferenceResult,
    InputMessage,
    ModelDescriptor,
    ModelStructure,
    ModelTask,
    Modality,
    RuntimeEnvironment,
    SupportDecision,
    SupportStatus,
)
from minimlx_runtime.vision_cache import VisionInputCache


def text_request(task: ModelTask = ModelTask.GENERATION) -> InferenceRequest:
    return InferenceRequest(
        task=task,
        messages=(
            InputMessage("user", (ContentPart(Modality.TEXT, text="hello"),)),
        ),
    )


def descriptor(*, components=(), profile=None, model_type="example_text", architecture="ExampleForCausalLM", metadata=None) -> ModelDescriptor:
    return ModelDescriptor(
        model_dir="/models/example",
        model_type=model_type,
        architecture=architecture,
        tokenizer_family="example",
        metadata=metadata or {},
        components=components,
        support=profile or CapabilityProfile(),
    )


def mtp_draft_descriptor(*, model_type="qwen3_5_mtp", architecture="Qwen3_5MTPForCausalLM"):
    return descriptor(
        model_type=model_type,
        architecture=architecture,
        metadata={"hidden_size": 2560, "vocab_size": 248320},
        components=(
            ArtifactComponent("draft.safetensors", ComponentKind.DRAFT_WEIGHTS, "draft.safetensors"),
            ArtifactComponent("draft.safetensors", ComponentKind.WEIGHTS_SHARD, "draft.safetensors"),
        ),
    )


def dflash_draft_descriptor():
    return descriptor(
        model_type="qwen3",
        architecture="DFlash2DraftModel",
        metadata={"hidden_size": 5120, "vocab_size": 248320},
        components=(
            ArtifactComponent("model.safetensors", ComponentKind.DRAFT_WEIGHTS, "model.safetensors"),
            ArtifactComponent("model.safetensors", ComponentKind.WEIGHTS_SHARD, "model.safetensors"),
        ),
    )


class SupportArchitectureTests(unittest.TestCase):
    def test_qwen35_mtp_stream_uses_specialized_overlay_without_baseline_adapter(self):
        from minimlx_runtime.adapters.qwen35 import Qwen35Adapter

        class FakeOverlay:
            def generate(self, *args, on_token=None, **kwargs):
                on_token("first", [101])
                on_token(" second", [102])
                return [1, 2], [101, 102], "first second", 5.0, 4.0

        adapter = Qwen35Adapter.__new__(Qwen35Adapter)
        adapter._mtp_enabled = True
        adapter._dflash_enabled = False
        adapter._optimization_active = False
        adapter._baseline_adapter = None
        adapter._optimization_overlay = FakeOverlay()
        adapter._tokenizer = None
        events = list(adapter.stream_generate("prompt", max_tokens=2))

        self.assertEqual([event["delta"] for event in events if event["event"] == "delta"], ["first", " second"])
        self.assertEqual(events[-1]["usage"], {"prompt_tokens": 2, "completion_tokens": 2})

    def test_qwen35_baseline_stream_forwards_events(self):
        from minimlx_runtime.adapters.qwen35 import Qwen35Adapter

        class FakeBaseline:
            def stream_generate(self, *args, **kwargs):
                yield {"event": "delta", "delta": "hello", "token_ids": [101]}
                yield {"event": "done", "text": "hello", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        adapter = Qwen35Adapter.__new__(Qwen35Adapter)
        adapter._dflash_enabled = False
        adapter._mtp_enabled = False
        adapter._optimization_active = False
        adapter._baseline_adapter = FakeBaseline()
        adapter._optimization_overlay = None
        events = list(adapter.stream_generate("prompt", max_tokens=1))
        self.assertEqual([event["delta"] for event in events if event["event"] == "delta"], ["hello"])
        self.assertEqual(events[-1]["event"], "done")

    def test_failed_mtp_cleanup_synchronizes_resets_and_drops_request_caches(self):
        from minimlx_runtime.adapters.qwen35 import _cleanup_failed_mtp_request

        events = []

        class FakeMLX:
            @staticmethod
            def synchronize(stream):
                events.append(("synchronize", stream))

            @staticmethod
            def clear_cache():
                events.append(("clear_cache",))

        class FakeDrafter:
            @staticmethod
            def reset(target):
                events.append(("reset", target))

        stream = object()
        target = object()
        target_cache = [object(), object()]
        shared_kv = {"key": object()}
        _cleanup_failed_mtp_request(
            FakeMLX, stream, FakeDrafter.reset, target, target_cache, shared_kv
        )

        self.assertEqual(events, [
            ("synchronize", stream),
            ("reset", target),
            ("clear_cache",),
        ])
        self.assertEqual(target_cache, [])
        self.assertEqual(shared_kv, {})

    def test_generic_vlm_provider_routes_unrelated_architectures_and_keeps_qwen35_priority(self):
        from minimlx_runtime.adapters.mlx_vlm import MLXVLMAdapter
        from minimlx_runtime.adapters.qwen35 import Qwen35Adapter
        from minimlx_runtime.registry import ModelRegistry

        components = (
            ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
            ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
            ArtifactComponent("processor", ComponentKind.PROCESSOR, "preprocessor_config.json"),
            ArtifactComponent("vision", ComponentKind.VISION_ENCODER, "model.safetensors"),
        )
        environment = RuntimeEnvironment(backends=frozenset({"mlx-vlm"}))
        resolver = CapabilityResolver(builtin_provider_registry())
        image_request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (ContentPart(Modality.TEXT, text="Read this"),
                                    ContentPart(Modality.IMAGE, uri="file:///fixture.png"))),),
        )
        for model_type, architecture in (
            ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
            ("llava", "LlavaForConditionalGeneration"),
            ("gemma4", "Gemma4ForConditionalGeneration"),
        ):
            with self.subTest(model_type=model_type):
                model = descriptor(
                    model_type=model_type,
                    architecture=architecture,
                    components=components,
                    metadata={"has_vision": True},
                    profile=CapabilityProfile(
                        tasks=frozenset({ModelTask.GENERATION}),
                        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
                        output_modalities=frozenset({Modality.TEXT}),
                    ),
                )
                decision, plan = resolver.resolve(model, image_request, environment)
                self.assertEqual(decision.status, SupportStatus.SUPPORTED)
                self.assertEqual(plan.task_adapter, "mlx-vlm-generation")
                self.assertEqual(plan.family_adapter, "mlx-vlm-generation")
                self.assertEqual(plan.backend_adapter, "mlx-vlm-runtime")
                self.assertTrue(MLXVLMAdapter.can_load(model))
                self.assertEqual(
                    ModelRegistry([Qwen35Adapter(), MLXVLMAdapter()]).resolve(model).name,
                    "mlx-vlm-model",
                )

        qwen35 = descriptor(
            model_type="qwen3_5", architecture="Qwen3_5ForConditionalGeneration",
            components=components, metadata={"has_vision": True},
            profile=CapabilityProfile(
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
                output_modalities=frozenset({Modality.TEXT}),
            ),
        )
        decision, plan = resolver.resolve(qwen35, image_request, environment)
        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.family_adapter, "qwen35-vlm")
        self.assertEqual(plan.backend_adapter, "qwen35-vlm")

    def test_qwen35_and_generic_vlm_share_loader_base_and_image_validation(self):
        from minimlx_runtime.adapters.mlx_vlm import MLXVLMAdapter
        from minimlx_runtime.adapters.qwen35 import Qwen35Adapter

        self.assertTrue(issubclass(Qwen35Adapter, MLXVLMAdapter))
        self.assertIs(Qwen35Adapter._image_input, MLXVLMAdapter._image_input)

    def test_qwen35_optimization_algorithms_are_owned_by_overlay(self):
        from minimlx_runtime.adapters.qwen35 import Qwen35Adapter
        from minimlx_runtime.adapters.qwen35_overlay import Qwen35PerformanceOverlay

        self.assertFalse(hasattr(Qwen35Adapter, "_generate_optimized"))
        self.assertFalse(hasattr(Qwen35Adapter, "_install_q4_native_prefill"))
        self.assertFalse(hasattr(Qwen35Adapter, "_qwen_cached_image_features"))
        self.assertTrue(hasattr(Qwen35PerformanceOverlay, "_generate_optimized"))
        self.assertTrue(hasattr(Qwen35PerformanceOverlay, "_install_q4_native_prefill"))
        self.assertTrue(hasattr(Qwen35PerformanceOverlay, "_qwen_cached_image_features"))

        host = types.SimpleNamespace(model=object())
        overlay = Qwen35PerformanceOverlay(host)
        overlay._greedy_skip_logsumexp = True
        self.assertIs(overlay.model, host.model)
        self.assertTrue(host._greedy_skip_logsumexp)

    def test_qwen35_vision_cache_is_a_hook_on_shared_generation_path(self):
        from unittest.mock import patch
        from minimlx_runtime.adapters.qwen35 import Qwen35Adapter

        seen = {}
        cache = object()
        image = object()

        class InputIds:
            shape = (1, 2)

            def reshape(self, *_shape):
                return self

            def tolist(self):
                return [10, 11]

        def fake_generate(_model, _processor, _prompt, **kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(
                text="ok", token_ids=[12], generation_tokens=1,
                prompt_tokens=2, prompt_tps=10.0, generation_tps=5.0,
            )

        fake_generate_module = types.ModuleType("mlx_vlm.generate")
        fake_generate_module.PromptCacheState = type("PromptCacheState", (), {})
        fake_generate_module.generate = fake_generate
        adapter = Qwen35Adapter()
        adapter.model = object()
        adapter.processor = object()
        adapter._vlm_enabled = True
        adapter._vision_feature_cache = cache
        adapter._apc_manager = cache
        adapter._prepare = lambda _messages, _parts: (
            "prompt", [image], InputIds(), None, None, {"image_grid_thw": object()}
        )
        with patch.dict(sys.modules, {"mlx_vlm.generate": fake_generate_module}):
            result = adapter.generate_multimodal([], [image], max_tokens=1)

        self.assertEqual(result[:3], ([10, 11], [12], "ok"))
        self.assertIs(seen["vision_cache"], cache)
        self.assertIs(seen["apc_manager"], cache)
        self.assertEqual(seen["_image_key"], [image])

    def test_shared_vlm_apc_is_opt_in_and_memory_only(self):
        from unittest.mock import patch
        from minimlx_runtime.adapters.mlx_vlm import MLXVLMAdapter

        seen = {}
        manager = object()
        apc_module = types.ModuleType("mlx_vlm.apc")

        def from_env(**kwargs):
            seen.update(kwargs)
            return manager

        apc_module.from_env = from_env
        descriptor = types.SimpleNamespace(model_dir="/models/example-vlm")
        with patch.dict(os.environ, {"MINIMLX_ENABLE_VLM_APC": "0"}, clear=False):
            self.assertIsNone(MLXVLMAdapter._create_apc_manager(descriptor))
        with patch.dict(
            os.environ,
            {
                "MINIMLX_ENABLE_VLM_APC": "1",
                "MINIMLX_VLM_APC_MEMORY_GB": "0.75",
            },
            clear=False,
        ), patch.dict(sys.modules, {"mlx_vlm.apc": apc_module}):
            self.assertIs(MLXVLMAdapter._create_apc_manager(descriptor), manager)

        self.assertTrue(seen["overrides"]["enabled"])
        self.assertFalse(seen["overrides"]["disk_enabled"])
        self.assertEqual(seen["overrides"]["memory_max_gb"], 0.75)
        self.assertTrue(seen["model_namespace"].startswith("minimlx-"))

    def test_shared_vlm_apc_reads_upstream_lookup_counter_names(self):
        from minimlx_runtime.adapters.mlx_vlm import MLXVLMAdapter

        adapter = MLXVLMAdapter()
        adapter._apc_manager = types.SimpleNamespace(
            stats_snapshot=lambda: {
                "lookups_hit": 1,
                "lookups_miss": 1,
                "matched_tokens": 2159,
                "resident_bytes": 583598080,
            }
        )

        self.assertEqual(
            adapter.inference_cache_metrics(),
            {
                "prefix_cache_hits": 1,
                "prefix_cache_misses": 1,
                "prefix_cache_matched_tokens": 2159,
                "prefix_cache_resident_bytes": 583598080,
            },
        )

    def test_generic_vlm_route_reports_missing_artifacts_instead_of_unknown_family(self):
        from minimlx_runtime.adapters.mlx_vlm import MLXVLMAdapter
        from minimlx_runtime.registry import ModelRegistry

        model = descriptor(
            model_type="llava_next",
            architecture="LlavaNextForConditionalGeneration",
            components=(
                ArtifactComponent("config", ComponentKind.CONFIG, "config.json"),
                ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
            ),
            metadata={"has_vision": True},
            profile=CapabilityProfile(
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
                output_modalities=frozenset({Modality.TEXT}),
            ),
        )
        self.assertTrue(MLXVLMAdapter.can_load(model))
        self.assertEqual(ModelRegistry([MLXVLMAdapter()]).resolve(model).name, "mlx-vlm-model")
        decision, _ = CapabilityResolver(builtin_provider_registry()).resolve(
            model,
            InferenceRequest(
                ModelTask.GENERATION,
                (InputMessage("user", (ContentPart(Modality.IMAGE, uri="file:///fixture.png"),)),),
            ),
            RuntimeEnvironment(backends=frozenset({"mlx-vlm"})),
        )
        self.assertEqual(decision.status, SupportStatus.MISSING_COMPONENT)

    def test_standard_text_adapter_and_provider_route_supports_qwen2(self):
        from minimlx_runtime.registry import ModelRegistry
        from minimlx_runtime.adapters.mlx_lm_standard import StandardTextAdapter

        components = (
            ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
            ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
        )
        model = descriptor(
            model_type="qwen2",
            architecture="Qwen2ForCausalLM",
            components=components,
            metadata={"quantization": {"bits": 4, "group_size": 64}},
        )
        self.assertTrue(StandardTextAdapter.can_load(model))
        self.assertTrue(StandardTextAdapter.can_load(descriptor(model_type="qwen3_5")))
        registry = ModelRegistry([StandardTextAdapter()])
        self.assertEqual(registry.resolve(model).name, "mlx-lm-standard")
        self.assertEqual(registry.create(model).name, "mlx-lm-standard")
        decision, plan = CapabilityResolver(builtin_provider_registry()).resolve(
            model,
            text_request(),
            RuntimeEnvironment(backends=frozenset({"mlx-lm"})),
        )
        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.family_adapter, "mlx-lm-standard")
        self.assertEqual(plan.backend_adapter, "mlx-lm-standard-runtime")
        self.assertEqual(
            plan.component_adapters[ComponentKind.TOKENIZER],
            "mlx-lm-tokenizer-component",
        )
        stream_request = InferenceRequest(
            ModelTask.GENERATION,
            text_request().messages,
            stream=True,
        )
        stream_decision, stream_plan = CapabilityResolver(
            builtin_provider_registry()
        ).resolve(model, stream_request, RuntimeEnvironment(backends=frozenset({"mlx-lm"})))
        self.assertEqual(stream_decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(stream_plan.family_adapter, plan.family_adapter)
        self.assertEqual(stream_plan.backend_adapter, plan.backend_adapter)

    def test_qwen2_adapter_stream_generate_contract(self):
        from unittest.mock import patch
        from minimlx_runtime.adapters.mlx_lm_standard import StandardTextAdapter

        seen = {}

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

        class Response:
            text = "OK"
            token = 7
            prompt_tps = 10.0
            prompt_tokens = 2
            generation_tps = 5.0

        fake_module = types.SimpleNamespace(
            stream_generate=lambda model, tokenizer, prompt, **kwargs: (
                seen.update(prompt=prompt, kwargs=kwargs) or iter((Response(),))
            )
        )
        adapter = StandardTextAdapter()
        adapter.model = object()
        adapter.tokenizer = Tokenizer()
        with patch.dict(sys.modules, {"mlx_lm": fake_module}):
            result = adapter.generate("hello", max_context_tokens=2)
        self.assertEqual(seen["prompt"], [ord("l"), ord("o")])
        self.assertEqual(result[:3], ([ord("l"), ord("o")], [7], "OK"))
        self.assertEqual(result[3], 200.0)

    def test_qwen2_adapter_uses_checkpoint_chat_template(self):
        from minimlx_runtime.adapters.mlx_lm_standard import StandardTextAdapter

        tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: (
                f"<{messages[0]['role']}>{messages[0]['content']}"
                f"|{kwargs['add_generation_prompt']}"
            )
        )
        adapter = StandardTextAdapter()
        adapter.tokenizer = tokenizer
        message = InputMessage(
            "user", (ContentPart(Modality.TEXT, text="hello"),)
        )
        self.assertEqual(adapter.format_messages((message,)), "<user>hello|True")

    def test_qwen2_adapter_forwards_model_chat_template_parameters(self):
        from minimlx_runtime.adapters.mlx_lm_standard import StandardTextAdapter

        seen = {}
        tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: seen.update(kwargs) or "prompt"
        )
        adapter = StandardTextAdapter()
        adapter.tokenizer = tokenizer
        message = InputMessage("user", (ContentPart(Modality.TEXT, text="hello"),))
        self.assertEqual(
            adapter.format_messages((message,), {"enable_thinking": False}), "prompt"
        )
        self.assertFalse(seen["enable_thinking"])
        self.assertFalse(seen["tokenize"])

    def test_all_official_qwen35_sizes_resolve_dense_or_moe_providers(self):
        resolver = CapabilityResolver(builtin_provider_registry())
        configurations = (
            ("Qwen3.5-0.8B", "qwen3_5", "Qwen3_5ForConditionalGeneration"),
            ("Qwen3.5-2B", "qwen3_5", "Qwen3_5ForConditionalGeneration"),
            ("Qwen3.5-4B", "qwen3_5", "Qwen3_5ForConditionalGeneration"),
            ("Qwen3.5-9B", "qwen3_5", "Qwen3_5ForConditionalGeneration"),
            ("Qwen3.5-27B", "qwen3_5", "Qwen3_5ForConditionalGeneration"),
            ("Qwen3.5-35B-A3B", "qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
            ("Qwen3.5-122B-A10B", "qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
            ("Qwen3.5-397B-A17B", "qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
        )
        vision_components = (
            ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
            ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
            ArtifactComponent("processor", ComponentKind.PROCESSOR, "processor.json"),
            ArtifactComponent("vision", ComponentKind.VISION_ENCODER, "model.safetensors"),
        )
        environment = RuntimeEnvironment(backends=frozenset({"mlx-lm", "mlx-vlm"}))

        for name, model_type, architecture in configurations:
            with self.subTest(model=name):
                model = ModelDescriptor(
                    model_dir=f"/fixtures/{name}",
                    model_type=model_type,
                    architecture=architecture,
                    tokenizer_family="qwen",
                    support=CapabilityProfile(
                        tasks=frozenset({ModelTask.GENERATION}),
                        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
                        output_modalities=frozenset({Modality.TEXT}),
                    ),
                    components=vision_components,
                )
                text_request = InferenceRequest(
                    ModelTask.GENERATION,
                    (InputMessage("user", (ContentPart(Modality.TEXT, text="hello"),)),),
                )
                image_request = InferenceRequest(
                    ModelTask.GENERATION,
                    (InputMessage("user", (ContentPart(Modality.IMAGE, uri="file:///fixture.png"),)),),
                )
                text_decision, text_plan = resolver.resolve(model, text_request, environment)
                stream_request = InferenceRequest(
                    ModelTask.GENERATION,
                    text_request.messages,
                    stream=True,
                )
                stream_decision, stream_plan = resolver.resolve(
                    model, stream_request, environment
                )
                image_decision, image_plan = resolver.resolve(model, image_request, environment)
                self.assertEqual(text_decision.status, SupportStatus.SUPPORTED)
                self.assertEqual(stream_decision.status, SupportStatus.SUPPORTED)
                # Qwen3.5 checkpoints advertise vision metadata, but a
                # text-only request must stay on the lighter mlx-lm lane.
                # The VLM provider is reserved for requests that actually
                # contain image/video input.
                self.assertEqual(text_plan.family_adapter, "qwen35-mlx-lm")
                self.assertEqual(stream_plan.family_adapter, text_plan.family_adapter)
                self.assertEqual(stream_plan.backend_adapter, text_plan.backend_adapter)
                self.assertEqual(text_plan.baseline_provider, "mlx-lm-standard")
                self.assertEqual(
                    text_plan.optimization_providers,
                    ("qwen35-performance-overlay",),
                )
                self.assertEqual(image_decision.status, SupportStatus.SUPPORTED)
                self.assertEqual(image_plan.family_adapter, "qwen35-vlm")
                self.assertEqual(image_plan.baseline_provider, "mlx-vlm-generation")
                self.assertEqual(
                    image_plan.optimization_providers,
                    ("qwen35-performance-overlay",),
                )
                disabled_environment = RuntimeEnvironment(
                    backends=frozenset({"mlx-lm", "mlx-vlm"}),
                    disabled_optimizations=frozenset(
                        {"qwen35-performance-overlay"}
                    ),
                )
                _, baseline_only_plan = resolver.resolve(
                    model, text_request, disabled_environment
                )
                self.assertEqual(
                    baseline_only_plan.baseline_provider, "mlx-lm-standard"
                )
                self.assertFalse(baseline_only_plan.optimization_providers)

    def test_qwen38_uses_generic_providers_and_is_excluded_from_qwen35_route(self):
        components = (
            ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
            ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
            ArtifactComponent("processor", ComponentKind.PROCESSOR, "processor.json"),
            ArtifactComponent("vision", ComponentKind.VISION_ENCODER, "model.safetensors"),
        )
        model = ModelDescriptor(
            model_dir="/models/mlx-community--Qwen3.8-27B-4bit",
            model_type="qwen3_5",
            architecture="Qwen3_5ForConditionalGeneration",
            tokenizer_family="qwen",
            metadata={"repository_id": "mlx-community/Qwen3.8-27B-4bit"},
            support=CapabilityProfile(
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
                output_modalities=frozenset({Modality.TEXT}),
            ),
            components=components,
        )
        resolver = CapabilityResolver(builtin_provider_registry())
        environment = RuntimeEnvironment(
            backends=frozenset({"mlx-lm", "mlx-vlm"})
        )
        text_decision, text_plan = resolver.resolve(model, text_request(), environment)
        image_request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (ContentPart(Modality.IMAGE, uri="file:///fixture.png"),)),),
        )
        image_decision, image_plan = resolver.resolve(model, image_request, environment)

        self.assertEqual(text_decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(text_plan.family_adapter, "mlx-lm-standard")
        self.assertEqual(text_plan.baseline_provider, "mlx-lm-standard")
        self.assertFalse(text_plan.optimization_providers)
        self.assertEqual(text_plan.backend_adapter, "mlx-lm-standard-runtime")
        self.assertEqual(image_decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(image_plan.family_adapter, "mlx-vlm-generation")
        self.assertEqual(image_plan.baseline_provider, "mlx-vlm-generation")
        self.assertFalse(image_plan.optimization_providers)
        self.assertEqual(image_plan.backend_adapter, "mlx-vlm-runtime")
        self.assertFalse(image_plan.accelerations)

    def test_qwen35_text_route_survives_missing_vision_processor(self):
        model = ModelDescriptor(
            model_dir="/fixtures/Qwen3.5-35B-A3B",
            model_type="qwen3_5_moe",
            architecture="Qwen3_5MoeForConditionalGeneration",
            tokenizer_family="qwen",
            support=CapabilityProfile(
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
                output_modalities=frozenset({Modality.TEXT}),
            ),
            components=(
                ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
                ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
            ),
        )
        decision, plan = CapabilityResolver(builtin_provider_registry()).resolve(
            model,
            InferenceRequest(
                ModelTask.GENERATION,
                (InputMessage("user", (ContentPart(Modality.TEXT, text="hello"),)),),
            ),
            RuntimeEnvironment(backends=frozenset({"mlx-lm", "mlx-vlm"})),
        )
        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.backend_adapter, "qwen35-runtime")

    def test_qwen35_discovery_preserves_mixed_quantization_metadata(self):
        quantization = {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "overrides": {
                "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 64},
                "model.layers.1.mlp.gate_proj": {"bits": 4, "group_size": 32},
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                        "quantization": quantization,
                        "text_config": {"num_experts": 64},
                    }
                )
            )
            discovered = describe_model(path)

        self.assertEqual(discovered.metadata["quantization"], quantization)
        self.assertIn(ModelStructure.MOE, discovered.support.structures)

    def test_vision_input_cache_is_bounded_lru_and_reports_stats(self):
        class Tensor:
            nbytes = 4

        cache = VisionInputCache(max_bytes=4)
        cache.put("a", Tensor())
        self.assertIsNotNone(cache.get("a"))
        self.assertIsNone(cache.get("b"))
        cache.put("b", Tensor())
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.stats(), (1, 2, 2, 4))

    def test_qwen35_vision_request_selects_vlm_provider_and_components(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "vision_config": {"hidden_size": 16},
                    }
                )
            )
            (path / "tokenizer.json").touch()
            (path / "preprocessor_config.json").touch()
            (path / "video_preprocessor_config.json").write_text("{}")
            (path / "model.safetensors").touch()
            (path / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "model.safetensors",
                            "visual.0.weight": "model.safetensors",
                            "multi_modal_projector.weight": "model.safetensors",
                        }
                    }
                )
            )
            model = describe_model(path)

        request = InferenceRequest(
            ModelTask.GENERATION,
            (
                InputMessage(
                    "user",
                    (
                        ContentPart(Modality.IMAGE, uri="file:///tmp/image.png"),
                        ContentPart(Modality.TEXT, text="describe it"),
                    ),
                ),
            ),
        )
        decision, plan = CapabilityResolver(builtin_provider_registry()).resolve(
            model,
            request,
            RuntimeEnvironment(backends=frozenset({"mlx-vlm"})),
        )
        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.backend_adapter, "qwen35-vlm")
        self.assertEqual(plan.task_adapter, "qwen35-vlm")
        self.assertTrue(model.metadata["has_video"])
        self.assertEqual(
            set(plan.component_adapters),
            {
                ComponentKind.TOKENIZER,
                ComponentKind.TEXT_WEIGHTS,
                ComponentKind.PROCESSOR,
                ComponentKind.VISION_ENCODER,
            },
        )
        video_request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (ContentPart(Modality.VIDEO, uri="file:///tmp/clip.mp4"),)),),
        )
        decision, video_plan = CapabilityResolver(builtin_provider_registry()).resolve(
            model,
            video_request,
            RuntimeEnvironment(backends=frozenset({"mlx-vlm"})),
        )
        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(video_plan.backend_adapter, "qwen35-vlm")

    def test_qwen35_vision_request_reports_missing_vlm_dependency(self):
        model = ModelDescriptor(
            model_dir="/models/qwen-vl",
            model_type="qwen3_5",
            architecture="Qwen3_5ForConditionalGeneration",
            tokenizer_family="qwen",
            support=CapabilityProfile(
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
                output_modalities=frozenset({Modality.TEXT}),
            ),
            components=(
                ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
                ArtifactComponent("text", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
                ArtifactComponent("processor", ComponentKind.PROCESSOR, "processor.json"),
                ArtifactComponent("vision", ComponentKind.VISION_ENCODER, "model.safetensors"),
            ),
        )
        request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (ContentPart(Modality.IMAGE, uri="file:///tmp/image.png"),)),),
        )
        decision, _ = CapabilityResolver(builtin_provider_registry()).resolve(
            model, request, RuntimeEnvironment()
        )
        self.assertEqual(decision.status, SupportStatus.DEPENDENCY_BLOCKED)
        self.assertEqual(decision.missing_dependencies, ("mlx-vlm",))

    def test_importing_discovery_is_headless_and_qwen_discovery_has_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                        "num_experts": 64,
                        "text_config": {"num_experts_per_tok": 4},
                    }
                )
            )
            (path / "tokenizer.json").write_text("{}")
            (path / "model.safetensors").touch()
            discovered = describe_model(path)

        self.assertEqual(discovered.support.tasks, {ModelTask.GENERATION})
        self.assertIn(ModelStructure.MOE, discovered.support.structures)
        self.assertNotIn("mtp", {task.value for task in discovered.support.tasks})
        self.assertTrue(any(item.field == "architecture" for item in discovered.evidence))
        self.assertTrue(any(item.kind is ComponentKind.TEXT_WEIGHTS for item in discovered.components))

    def test_runtime_construction_does_not_import_mlx_or_require_metal(self):
        import os
        import subprocess
        import sys

        project_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project_root / "python")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from minimlx_runtime.runtime import Runtime; "
                "Runtime(); assert 'mlx.core' not in sys.modules",
            ],
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_inspect_resolves_model_without_loading_mlx(self):
        import os
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                    }
                )
            )
            (model / "tokenizer.json").write_text("{}")
            (model / "model.safetensors").touch()
            project_root = Path(__file__).resolve().parents[1]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root / "python")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.argv=['minimlx-runtime','inspect',sys.argv[1]]; "
                    "from minimlx_runtime.cli import main; main(); "
                    "assert 'mlx.core' not in sys.modules",
                    str(model),
                ],
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["descriptor"]["model_type"], "qwen3_5")
        self.assertIn(report["decision"]["status"], {status.value for status in SupportStatus})

    def test_discovery_does_not_infer_execution_capabilities_for_unknown_or_qwen3_models(self):
        for model_type, architecture in (
            ("example_unknown", "ExampleForCausalLM"),
            ("qwen3", "Qwen3ForCausalLM"),
        ):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                (path / "config.json").write_text(
                    json.dumps(
                        {"model_type": model_type, "architectures": [architecture]}
                    )
                )
                discovered = describe_model(path)

            self.assertEqual(discovered.capabilities, frozenset())
            self.assertEqual(discovered.support.tasks, frozenset())
            self.assertNotIn(ModelStructure.HYBRID_ATTENTION, discovered.support.structures)

    def test_discovery_requires_canonical_qwen35_and_numeric_moe_metadata(self):
        cases = (
            ("qwen3_vl5", "Qwen3VL5ForConditionalGeneration", "64", False, False),
            ("qwen3_5", "Qwen3_5ForConditionalGeneration", "0", True, False),
            ("qwen3_5", "Qwen3_5ForConditionalGeneration", True, True, False),
        )
        for model_type, architecture, num_experts, is_qwen35, is_moe in cases:
            with self.subTest(model_type=model_type, num_experts=num_experts), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                (path / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": model_type,
                            "architectures": [architecture],
                            "num_experts": num_experts,
                        }
                    )
                )
                discovered = describe_model(path)

            self.assertEqual(discovered.support.tasks == {ModelTask.GENERATION}, is_qwen35)
            self.assertEqual(ModelStructure.MOE in discovered.support.structures, is_moe)

    def test_discovery_marks_declared_mtp_checkpoint_as_draft_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5_mtp", "mtp_num_hidden_layers": 1})
            )
            (path / "model.safetensors").touch()
            discovered = describe_model(path)

        self.assertIn(
            ComponentKind.DRAFT_WEIGHTS,
            {component.kind for component in discovered.components if component.present},
        )

    def test_resolver_composes_independent_task_family_and_backend_providers(self):
        registry = ProviderRegistry(
            [
                AdapterRegistration(
                    "task-generation",
                    frozenset({AdapterRole.TASK}),
                    tasks=frozenset({ModelTask.GENERATION}),
                ),
                AdapterRegistration(
                    "family-example",
                    frozenset({AdapterRole.FAMILY}),
                    families=("example",),
                ),
                AdapterRegistration(
                    "backend-test",
                    frozenset({AdapterRole.BACKEND}),
                    backend="test-backend",
                    required_dependencies=frozenset({"test-backend"}),
                ),
            ]
        )
        decision, plan = CapabilityResolver(registry).resolve(
            descriptor(), text_request(), RuntimeEnvironment(backends=frozenset({"test-backend"}))
        )

        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.task_adapter, "task-generation")
        self.assertEqual(plan.family_adapter, "family-example")
        self.assertEqual(plan.backend_adapter, "test-backend")

    def test_resolver_falls_back_to_lower_priority_compatible_provider_combination(self):
        registry = ProviderRegistry(
            [
                AdapterRegistration(
                    "task-fast-but-incompatible",
                    frozenset({AdapterRole.TASK}),
                    tasks=frozenset({ModelTask.GENERATION}),
                    backend="mlx-fast",
                    priority=100,
                ),
                AdapterRegistration(
                    "task-compatible",
                    frozenset({AdapterRole.TASK}),
                    tasks=frozenset({ModelTask.GENERATION}),
                    backend="mlx-compatible",
                    priority=10,
                ),
                AdapterRegistration(
                    "family-fast-but-incompatible",
                    frozenset({AdapterRole.FAMILY}),
                    families=("example",),
                    backend="mlx-other",
                    priority=100,
                ),
                AdapterRegistration(
                    "family-compatible",
                    frozenset({AdapterRole.FAMILY}),
                    families=("example",),
                    backend="mlx-compatible",
                    priority=10,
                ),
                AdapterRegistration(
                    "backend-fast-but-incompatible",
                    frozenset({AdapterRole.BACKEND}),
                    backend="mlx-fast",
                    priority=100,
                ),
                AdapterRegistration(
                    "backend-compatible",
                    frozenset({AdapterRole.BACKEND}),
                    backend="mlx-compatible",
                    priority=10,
                ),
            ]
        )

        decision, plan = CapabilityResolver(registry).resolve(
            descriptor(), text_request(), RuntimeEnvironment()
        )

        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.task_adapter, "task-compatible")
        self.assertEqual(plan.family_adapter, "family-compatible")
        self.assertEqual(plan.backend_adapter, "mlx-compatible")

    def test_resolver_falls_back_when_top_ranked_backend_dependency_is_missing(self):
        registrations = []
        for role, label in (
            (AdapterRole.TASK, "task"),
            (AdapterRole.FAMILY, "family"),
            (AdapterRole.BACKEND, "backend"),
        ):
            kwargs = (
                {"tasks": frozenset({ModelTask.GENERATION})}
                if role is AdapterRole.TASK
                else {"families": ("example",)}
                if role is AdapterRole.FAMILY
                else {}
            )
            registrations.extend(
                (
                    AdapterRegistration(
                        f"{label}-preferred",
                        frozenset({role}),
                        backend="mlx-unavailable",
                        required_dependencies=frozenset({"missing-package"}),
                        priority=100,
                        **kwargs,
                    ),
                    AdapterRegistration(
                        f"{label}-fallback",
                        frozenset({role}),
                        backend="mlx-available",
                        required_dependencies=frozenset({"mlx-lm"}),
                        priority=10,
                        **kwargs,
                    ),
                )
            )
        decision, plan = CapabilityResolver(ProviderRegistry(registrations)).resolve(
            descriptor(), text_request(), RuntimeEnvironment(backends=frozenset({"mlx-lm"}))
        )

        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.task_adapter, "task-fallback")
        self.assertEqual(plan.family_adapter, "family-fallback")
        self.assertEqual(plan.backend_adapter, "mlx-available")

    def test_resolver_falls_back_when_top_ranked_route_lacks_component_adapter(self):
        registry = ProviderRegistry(
            [
                AdapterRegistration(
                    "task-with-unavailable-vision-path",
                    frozenset({AdapterRole.TASK}),
                    tasks=frozenset({ModelTask.GENERATION}),
                    required_components=frozenset({ComponentKind.VISION_ENCODER}),
                    backend="mlx-vlm",
                    priority=100,
                ),
                AdapterRegistration(
                    "task-text-fallback",
                    frozenset({AdapterRole.TASK}),
                    tasks=frozenset({ModelTask.GENERATION}),
                    backend="mlx-lm",
                    priority=10,
                ),
                AdapterRegistration(
                    "family-vlm",
                    frozenset({AdapterRole.FAMILY}),
                    families=("example",),
                    backend="mlx-vlm",
                    priority=100,
                ),
                AdapterRegistration(
                    "family-text",
                    frozenset({AdapterRole.FAMILY}),
                    families=("example",),
                    backend="mlx-lm",
                    priority=10,
                ),
                AdapterRegistration(
                    "backend-vlm",
                    frozenset({AdapterRole.BACKEND}),
                    backend="mlx-vlm",
                    priority=100,
                ),
                AdapterRegistration(
                    "backend-mlx-lm",
                    frozenset({AdapterRole.BACKEND}),
                    backend="mlx-lm",
                    priority=10,
                ),
            ]
        )

        decision, plan = CapabilityResolver(registry).resolve(
            descriptor(), text_request(), RuntimeEnvironment()
        )

        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.task_adapter, "task-text-fallback")
        self.assertEqual(plan.family_adapter, "family-text")
        self.assertEqual(plan.backend_adapter, "mlx-lm")

    def test_resolver_selects_required_component_provider_into_execution_plan(self):
        component_registry = ProviderRegistry(
            [
                AdapterRegistration(
                    "vision-generation",
                    frozenset({AdapterRole.TASK}),
                    tasks=frozenset({ModelTask.GENERATION}),
                    input_modalities=frozenset({Modality.IMAGE}),
                ),
                AdapterRegistration(
                    "example-family",
                    frozenset({AdapterRole.FAMILY}),
                    families=("example",),
                    required_components=frozenset({ComponentKind.VISION_ENCODER}),
                ),
                AdapterRegistration(
                    "mlx-vlm",
                    frozenset({AdapterRole.BACKEND}),
                    backend="mlx-vlm",
                    required_dependencies=frozenset({"mlx-vlm"}),
                ),
                AdapterRegistration(
                    "vision-loader",
                    frozenset({AdapterRole.COMPONENT}),
                    component_kind=ComponentKind.VISION_ENCODER,
                    priority=5,
                ),
            ]
        )
        model = descriptor(
            model_type="example_text",
            components=(
                ArtifactComponent("vision", ComponentKind.VISION_ENCODER, "vision.safetensors"),
            ),
            profile=CapabilityProfile(
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.IMAGE}),
            ),
        )
        image_request = InferenceRequest(
            ModelTask.GENERATION,
            (
                InputMessage(
                    "user",
                    (ContentPart(Modality.IMAGE, uri="file:///image.png"),),
                ),
            ),
        )
        decision, plan = CapabilityResolver(component_registry).resolve(
            model, image_request, RuntimeEnvironment(backends=frozenset({"mlx-vlm"}))
        )

        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(
            plan.component_adapters,
            {ComponentKind.VISION_ENCODER: "vision-loader"},
        )

    def test_component_provider_registration_requires_a_component_kind(self):
        with self.assertRaisesRegex(ValueError, "declare exactly one component_kind"):
            AdapterRegistration("broken-component", frozenset({AdapterRole.COMPONENT}))

    def test_selected_component_provider_dependencies_are_checked(self):
        registry = ProviderRegistry(
            [
                AdapterRegistration(
                    "task",
                    frozenset({AdapterRole.TASK}),
                    tasks=frozenset({ModelTask.GENERATION}),
                ),
                AdapterRegistration(
                    "family",
                    frozenset({AdapterRole.FAMILY}),
                    families=("example",),
                    required_components=frozenset({ComponentKind.VISION_ENCODER}),
                ),
                AdapterRegistration(
                    "backend",
                    frozenset({AdapterRole.BACKEND}),
                    backend="vision-backend",
                ),
                AdapterRegistration(
                    "vision-component",
                    frozenset({AdapterRole.COMPONENT}),
                    component_kind=ComponentKind.VISION_ENCODER,
                    required_dependencies=frozenset({"vision-processor"}),
                ),
            ]
        )
        model = descriptor(
            components=(
                ArtifactComponent("vision", ComponentKind.VISION_ENCODER, "vision.safetensors"),
            )
        )
        decision, plan = CapabilityResolver(registry).resolve(
            model, text_request(), RuntimeEnvironment()
        )
        self.assertEqual(decision.status, SupportStatus.DEPENDENCY_BLOCKED)
        self.assertEqual(decision.missing_dependencies, ("vision-processor",))
        self.assertIsNone(plan)

    def test_required_component_without_loader_is_not_reported_supported(self):
        registry = ProviderRegistry(
            [
                AdapterRegistration(
                    "generation",
                    frozenset({AdapterRole.TASK}),
                    tasks=frozenset({ModelTask.GENERATION}),
                ),
                AdapterRegistration(
                    "vision-family",
                    frozenset({AdapterRole.FAMILY}),
                    families=("example",),
                    required_components=frozenset({ComponentKind.VISION_ENCODER}),
                ),
                AdapterRegistration(
                    "vision-backend",
                    frozenset({AdapterRole.BACKEND}),
                    backend="vision-backend",
                ),
            ]
        )
        model = descriptor(
            components=(
                ArtifactComponent("vision", ComponentKind.VISION_ENCODER, "vision.safetensors"),
            )
        )

        decision, plan = CapabilityResolver(registry).resolve(
            model, text_request(), RuntimeEnvironment()
        )

        self.assertEqual(decision.status, SupportStatus.MISSING_COMPONENT)
        self.assertEqual(decision.missing_components, ("vision_encoder",))
        self.assertIsNone(plan)

    def test_qwen35_is_the_first_complete_builtin_route(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                    }
                )
            )
            (path / "tokenizer.json").write_text("{}")
            (path / "model.safetensors").touch()
            qwen_descriptor = describe_model(path)

        decision, plan = CapabilityResolver(builtin_provider_registry()).resolve(
            qwen_descriptor,
            text_request(),
            RuntimeEnvironment(backends=frozenset({"mlx-lm"})),
        )
        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.task_adapter, "qwen35-mlx-lm")
        self.assertEqual(plan.family_adapter, "qwen35-mlx-lm")

    def test_execution_plan_rejects_untyped_acceleration_component_binding(self):
        with self.assertRaisesRegex(ValueError, "exactly one component kind"):
            ExecutionPlan(
                phase="inference",
                accelerations=frozenset({Acceleration.MTP}),
                acceleration_adapters={Acceleration.MTP: "draft-provider"},
            )

    def test_resolver_rejects_conflicting_acceleration_component_providers(self):
        registrations = [
            AdapterRegistration(
                "task",
                frozenset({AdapterRole.TASK}),
                tasks=frozenset({ModelTask.GENERATION}),
                families=("example",),
                backend="test-backend",
                accelerations=frozenset({Acceleration.MTP}),
                acceleration_component_adapters=(
                    (Acceleration.MTP, ComponentKind.DRAFT_WEIGHTS, "draft-a"),
                ),
            ),
            AdapterRegistration(
                "family",
                frozenset({AdapterRole.FAMILY}),
                families=("example",),
                backend="test-backend",
                accelerations=frozenset({Acceleration.MTP}),
            ),
            AdapterRegistration(
                "backend",
                frozenset({AdapterRole.BACKEND}),
                families=("example",),
                backend="test-backend",
                accelerations=frozenset({Acceleration.MTP}),
                acceleration_component_adapters=(
                    (Acceleration.MTP, ComponentKind.DRAFT_WEIGHTS, "draft-b"),
                ),
            ),
            AdapterRegistration(
                "draft-a",
                frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.DRAFT_WEIGHTS,
                families=("example",),
            ),
            AdapterRegistration(
                "draft-b",
                frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.DRAFT_WEIGHTS,
                families=("example",),
            ),
        ]
        decision, plan = CapabilityResolver(ProviderRegistry(registrations)).resolve(
            descriptor(model_type="example", architecture="ExampleForCausalLM"),
            text_request(),
            RuntimeEnvironment(accelerations=frozenset({Acceleration.MTP})),
        )
        self.assertEqual(decision.status, SupportStatus.AMBIGUOUS)
        self.assertIn("conflicting acceleration components", decision.reason)
        self.assertIsNone(plan)

    def test_model_instance_releases_components_before_primary_and_is_idempotent(self):
        from minimlx_runtime.model_instance import ModelInstance, ModelInstanceState

        released = []
        instance = ModelInstance(
            descriptor=descriptor(), plan=ExecutionPlan(phase="inference")
        )
        instance.attach_primary("family-runtime", object(), lambda _: released.append("primary"))
        processor = object()
        instance.attach_component(
            ComponentKind.PROCESSOR,
            "processor",
            processor,
            lambda _: released.append("processor"),
        )
        instance.mark_ready()
        self.assertEqual(instance.state, ModelInstanceState.READY)
        self.assertEqual(instance.components(ComponentKind.PROCESSOR), (processor,))

        instance.unload()
        instance.unload()

        self.assertEqual(released, ["processor", "primary"])
        self.assertEqual(instance.state, ModelInstanceState.UNLOADED)
        self.assertEqual(instance.components(), ())

    def test_model_instance_unload_continues_after_a_component_release_error(self):
        from minimlx_runtime.model_instance import ModelInstance, ModelInstanceState

        released = []

        def fail_release(_):
            released.append("component")
            raise RuntimeError("component cleanup failure")

        instance = ModelInstance(
            descriptor=descriptor(), plan=ExecutionPlan(phase="inference")
        )
        instance.attach_primary("family-runtime", object(), lambda _: released.append("primary"))
        instance.attach_component(
            ComponentKind.DRAFT_WEIGHTS, "drafter", object(), fail_release
        )
        instance.mark_ready()

        with self.assertRaises(ExceptionGroup):
            instance.unload()

        self.assertEqual(released, ["component", "primary"])
        self.assertEqual(instance.state, ModelInstanceState.UNLOADED)

    def test_model_registry_uses_explicit_factory_for_isolated_instances(self):
        from minimlx_runtime.registry import ModelRegistry

        instances = []

        class AdapterStub:
            name = "example"

            def can_load(self, model):
                return model.model_type == "example_text"

            def load(self, model):
                return self

        prototype = AdapterStub()

        def factory():
            instance = AdapterStub()
            instances.append(instance)
            return instance

        registry = ModelRegistry()
        registry.register(prototype, factory=factory)
        first = registry.create(descriptor())
        second = registry.create(descriptor())

        self.assertIsNot(first, second)
        self.assertIsNot(first, prototype)
        self.assertEqual(instances, [first, second])

    def test_runtime_model_inspection_is_weight_load_free_and_serializable(self):
        from minimlx_runtime.runtime import Runtime

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]})
            )
            (path / "tokenizer.json").write_text("{}")
            (path / "model.safetensors").touch()
            report = Runtime.inspect_model(
                path,
                text_request(),
                RuntimeEnvironment(backends=frozenset({"mlx-lm"})),
            )

        self.assertEqual(report["decision"]["status"], "supported")
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["plan"]["task"], "generation")
        self.assertEqual(report["plan"]["backend_adapter"], "qwen35-runtime")
        self.assertTrue(report["descriptor"]["components"])

    def test_worker_inspect_command_uses_typed_request_without_loading_runtime(self):
        from minimlx_runtime.api.worker import Worker

        class InspectionOnlyRuntime:
            @staticmethod
            def inspect_model(model_dir, request):
                return {
                    "descriptor": {"model_dir": model_dir},
                    "decision": {"status": "unsupported", "task": request.task.value},
                    "plan": None,
                }

        response = Worker(InspectionOnlyRuntime()).handle(
            {
                "command": "inspect_model",
                "model_dir": "/models/example",
                "task": "embedding",
                "messages": [
                    {"role": "user", "content": [{"modality": "text", "text": "query"}]}
                ],
            }
        )
        self.assertEqual(response["command"], "inspect_model")
        self.assertEqual(response["decision"]["status"], "unsupported")
        self.assertIsNone(response["plan"])

    def test_worker_infer_command_round_trips_versioned_typed_payload(self):
        from minimlx_runtime.api.worker import Worker

        class InferenceOnlyRuntime:
            @staticmethod
            def infer(request):
                self.assertEqual(request.task, ModelTask.GENERATION)
                self.assertEqual(request.messages[0].content[0].text, "hello")
                self.assertEqual(request.accelerations, frozenset({Acceleration.MTP}))
                return InferenceResult(
                    ModelTask.GENERATION,
                    GenerationPayload("world", [11], [22]),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                    metrics={"decode_ms": 2.0},
                )

        response = Worker(InferenceOnlyRuntime()).handle(
            {
                "command": "infer",
                "schema_version": 1,
                "task": "generation",
                "messages": [
                    {"role": "user", "content": [{"modality": "text", "text": "hello"}]}
                ],
                "accelerations": ["mtp"],
                "options": {"max_tokens": 1},
            }
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["schema_version"], 1)
        self.assertEqual(response["task"], "generation")
        self.assertEqual(response["payload"]["text"], "world")

    def test_inference_request_accelerations_are_typed_with_legacy_compatibility(self):
        from minimlx_runtime.inference_protocol import inference_request_from_dict

        base = {
            "schema_version": 1,
            "task": "generation",
            "messages": [
                {"role": "user", "content": [{"modality": "text", "text": "hello"}]}
            ],
        }
        typed = inference_request_from_dict({**base, "accelerations": ["mtp"]})
        self.assertEqual(typed.accelerations, frozenset({Acceleration.MTP}))

        legacy = inference_request_from_dict(
            {**base, "options": {"accelerations": ["mtp"]}}
        )
        # Legacy payloads are normalized by the resolver, while new callers use
        # the typed top-level field.
        self.assertEqual(legacy.accelerations, frozenset())
        self.assertEqual(legacy.options["accelerations"], ["mtp"])

        with self.assertRaisesRegex(ValueError, "invalid inference request"):
            inference_request_from_dict({**base, "accelerations": ["unknown"]})

    def test_structured_task_content_is_typed_validated_and_round_trips(self):
        raw = {
            "schema_version": 1,
            "task": "reranking",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "modality": "structured",
                            "data": {"query": "blue sky", "documents": ["one", "two"]},
                        }
                    ],
                }
            ],
        }
        request = inference_request_from_dict(raw)
        part = request.messages[0].content[0]

        self.assertIs(part.modality, Modality.STRUCTURED)
        self.assertEqual(part.data, {"query": "blue sky", "documents": ["one", "two"]})
        with self.assertRaisesRegex(ValueError, "structured content requires an object"):
            inference_request_from_dict(
                {
                    **raw,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"modality": "structured", "data": ["not", "object"]}],
                        }
                    ],
                }
            )

    def test_video_content_accepts_timestamped_frame_lists(self):
        request = inference_request_from_dict(
            {
                "schema_version": 1,
                "task": "generation",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "modality": "video",
                                "frames": [
                                    {
                                        "timestamp_seconds": 1.25,
                                        "uri": "file:///frame.png",
                                        "mime_type": "image/png",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        frame = request.messages[0].content[0].frames[0]
        self.assertEqual(frame.timestamp_seconds, 1.25)
        self.assertEqual(frame.uri, "file:///frame.png")

        with self.assertRaisesRegex(ValueError, "either a reference or a non-empty frame list"):
            ContentPart(
                Modality.VIDEO,
                uri="file:///video.mp4",
                frames=(frame,),
            )

    def test_worker_unload_command_releases_runtime_and_returns_ack(self):
        from minimlx_runtime.api.worker import Worker

        class RuntimeStub:
            unloaded = False

            def unload(self):
                self.unloaded = True

        runtime = RuntimeStub()
        response = Worker(runtime).handle({"command": "unload"})
        self.assertTrue(runtime.unloaded)
        self.assertEqual(response, {"ok": True, "command": "unload", "released": True})

    def test_transport_error_response_preserves_support_decision_fields(self):
        from minimlx_runtime.errors import SupportResolutionError, error_response

        decision = SupportDecision(
            SupportStatus.DEPENDENCY_BLOCKED,
            "required backend dependencies are unavailable",
            missing_dependencies=("mlx-vlm==0.7.2 (found not installed)",),
        )
        response = error_response("infer", SupportResolutionError(decision))

        self.assertFalse(response["ok"])
        self.assertEqual(response["command"], "infer")
        self.assertEqual(response["error_code"], "dependency_blocked")
        self.assertTrue(response["retryable"])
        self.assertEqual(
            response["details"]["missing_dependencies"],
            ["mlx-vlm==0.7.2 (found not installed)"],
        )

    def test_transport_error_response_classifies_invalid_requests(self):
        from minimlx_runtime.errors import error_response

        response = error_response("infer", ValueError("bad task"))

        self.assertEqual(response["error_code"], "invalid_request")
        self.assertFalse(response["retryable"])

    def test_runtime_infer_raises_support_error_with_original_decision(self):
        from minimlx_runtime.errors import SupportResolutionError
        from minimlx_runtime.runtime import Runtime

        decision = SupportDecision(
            SupportStatus.MISSING_COMPONENT,
            "vision encoder is missing",
            missing_components=("vision_encoder",),
        )

        class ResolverStub:
            def resolve(self, descriptor, request, environment):
                return decision, None

        runtime = Runtime.__new__(Runtime)
        runtime.adapter = object()
        runtime.descriptor = descriptor()
        runtime.resolver = ResolverStub()
        runtime._environment = lambda: RuntimeEnvironment()

        with self.assertRaises(SupportResolutionError) as raised:
            runtime.infer(text_request())

        self.assertIs(raised.exception.decision, decision)

    def test_runtime_unload_clears_route_references_before_releasing_adapter(self):
        from minimlx_runtime.runtime import Runtime
        from minimlx_runtime.model_instance import ModelInstance

        class AdapterStub:
            unloaded = False

            def unload(self):
                self.unloaded = True

        adapter = AdapterStub()
        instance = ModelInstance(
            descriptor=descriptor(), plan=ExecutionPlan(phase="inference")
        )
        instance.attach_primary("family", adapter, lambda value: Runtime._release_adapter(value))
        instance.mark_ready()
        runtime = Runtime.__new__(Runtime)
        runtime._instance = instance
        runtime.adapter = adapter
        runtime.descriptor = object()
        runtime.execution_plan = object()
        runtime._runtime_environment = RuntimeEnvironment()
        runtime.adapters = {"family": adapter}
        runtime.backend_adapters = {"backend": adapter}

        runtime.unload()

        self.assertTrue(adapter.unloaded)
        self.assertIsNone(instance.primary)
        self.assertIsNone(runtime.adapter)
        self.assertIsNone(runtime.descriptor)
        self.assertIsNone(runtime.execution_plan)
        self.assertIsNone(runtime._runtime_environment)
        self.assertEqual(runtime.adapters, {})
        self.assertEqual(runtime.backend_adapters, {})

    def test_optional_adapter_cleanup_supports_adapters_without_unload_hook(self):
        from minimlx_runtime.runtime import Runtime

        Runtime._release_adapter(object())

    def test_failed_model_replacement_preserves_current_adapter_and_root_error(self):
        from unittest.mock import patch

        from minimlx_runtime import runtime as runtime_module
        from minimlx_runtime.runtime import Runtime

        class Candidate:
            name = "example"
            cleaned = False

            def load(self, model, plan=None):
                raise ValueError("candidate load failed")

            def unload(self):
                self.cleaned = True
                raise RuntimeError("cleanup error must not mask load error")

        candidate = Candidate()
        prototype = Candidate()

        class RegistryStub:
            def resolve(self, model):
                return prototype

            def create(self, model):
                return candidate

        class ResolverStub:
            def resolve(self, model, request, environment):
                return SupportDecision(SupportStatus.SUPPORTED, "ok"), ExecutionPlan(
                    phase="inference",
                    task=ModelTask.GENERATION,
                    task_adapter="task",
                    family_adapter="example",
                    backend_adapter="backend",
                )

        old = object()
        runtime = Runtime.__new__(Runtime)
        runtime.registry = RegistryStub()
        runtime.resolver = ResolverStub()
        runtime._environment = lambda: RuntimeEnvironment()
        runtime.adapter = old
        runtime.descriptor = object()
        runtime.execution_plan = object()
        runtime.adapters = {"example": old}
        runtime.backend_adapters = {"backend": old}

        with patch.object(runtime_module, "describe_model", return_value=descriptor()):
            with self.assertRaisesRegex(ValueError, "candidate load failed"):
                runtime.load("/models/new")

        self.assertTrue(candidate.cleaned)
        self.assertIs(runtime.adapter, old)
        self.assertEqual(runtime.adapters, {"example": old})
        self.assertEqual(runtime.backend_adapters, {"backend": old})

    def test_successful_model_replacement_publishes_candidate_and_plan(self):
        from unittest.mock import patch

        from minimlx_runtime import runtime as runtime_module
        from minimlx_runtime.runtime import Runtime

        released = []

        class AdapterStub:
            name = "example"

            def __init__(self, label):
                self.label = label
                self.tokenizer = object()
                self.drafter = object()

            def can_load(self, model):
                return True

            def load(self, model, plan=None):
                self.descriptor = model
                self.plan = plan
                return self

            def unload(self):
                released.append(self.label)

        candidate = AdapterStub("candidate")
        prototype = AdapterStub("prototype")
        old = AdapterStub("old")
        plan = ExecutionPlan(
            phase="inference",
            task=ModelTask.GENERATION,
            task_adapter="task",
            family_adapter="example",
            backend_adapter="backend",
            component_adapters={
                ComponentKind.TOKENIZER: "tokenizer-provider",
            },
            accelerations=frozenset({Acceleration.MTP}),
            acceleration_adapters={Acceleration.MTP: "draft-provider"},
            acceleration_component_kinds={
                Acceleration.MTP: ComponentKind.DRAFT_WEIGHTS
            },
        )

        class RegistryStub:
            def resolve(self, model):
                return prototype

            def create(self, model):
                return candidate

        class ResolverStub:
            def resolve(self, model, request, environment):
                return SupportDecision(SupportStatus.SUPPORTED, "ok"), plan

        runtime = Runtime.__new__(Runtime)
        runtime.registry = RegistryStub()
        runtime.resolver = ResolverStub()
        runtime._environment = lambda: RuntimeEnvironment()
        runtime.adapter = old
        runtime.descriptor = object()
        runtime.execution_plan = None
        runtime.adapters = {"example": old}
        runtime.backend_adapters = {"backend": old}
        runtime.component_loaders = {
            "tokenizer-provider": lambda adapter, model, resolved_plan: adapter.tokenizer,
            "draft-provider": lambda adapter, model, resolved_plan: adapter.drafter,
        }
        new_descriptor = descriptor()

        with patch.object(runtime_module, "describe_model", return_value=new_descriptor):
            runtime.load("/models/new")

        self.assertIs(runtime.adapter, candidate)
        self.assertIs(runtime.descriptor, new_descriptor)
        self.assertIs(runtime.execution_plan, plan)
        self.assertIs(runtime.adapters["example"], candidate)
        self.assertIs(runtime.backend_adapters["backend"], candidate)
        self.assertEqual(runtime._instance.state.value, "ready")
        self.assertIs(runtime._instance.primary, candidate)
        self.assertIs(candidate.plan, plan)
        self.assertIs(
            runtime._instance.components(ComponentKind.TOKENIZER)[0], candidate.tokenizer
        )
        self.assertIs(
            runtime._instance.components(ComponentKind.DRAFT_WEIGHTS)[0], candidate.drafter
        )
        self.assertEqual(released, ["old"])

    def test_optional_acceleration_and_overlay_failure_re_resolves_to_base_plan(self):
        from unittest.mock import patch

        from minimlx_runtime import runtime as runtime_module
        from minimlx_runtime.runtime import Runtime

        released = []

        class AdapterStub:
            name = "qwen35-mlx-lm"

            def __init__(self, fail=False):
                self.fail = fail
                self.tokenizer = object()
                self.model = object()

            def can_load(self, _descriptor):
                return True

            def load(self, _descriptor, plan=None):
                if self.fail and (plan.accelerations or plan.optimization_providers):
                    raise RuntimeError("optional execution layer initialization failed")
                self.plan = plan
                return self

            def unload(self):
                released.append(self)

        accelerated_plan = ExecutionPlan(
            phase="inference",
            task=ModelTask.GENERATION,
            task_adapter="qwen35-mlx-lm",
            family_adapter="qwen35-mlx-lm",
            backend_adapter="qwen35-runtime",
            accelerations=frozenset({Acceleration.MTP}),
            baseline_provider="mlx-lm-standard",
            optimization_providers=("qwen35-performance-overlay",),
        )
        base_plan = ExecutionPlan(
            phase="inference",
            task=ModelTask.GENERATION,
            task_adapter="qwen35-mlx-lm",
            family_adapter="qwen35-mlx-lm",
            backend_adapter="qwen35-runtime",
            baseline_provider="mlx-lm-standard",
        )

        class RegistryStub:
            def __init__(self):
                self.created = []

            def resolve(self, _descriptor):
                return AdapterStub()

            def create(self, _descriptor):
                adapter = AdapterStub(fail=not self.created)
                self.created.append(adapter)
                return adapter

        class ResolverStub:
            def __init__(self):
                self.environments = []

            def resolve(self, _descriptor, _request, environment):
                self.environments.append(environment)
                plan = (
                    accelerated_plan
                    if environment.accelerations
                    and "qwen35-performance-overlay"
                    not in environment.disabled_optimizations
                    else base_plan
                )
                return SupportDecision(SupportStatus.SUPPORTED, "ok"), plan

        runtime = Runtime.__new__(Runtime)
        runtime.registry = RegistryStub()
        runtime.resolver = ResolverStub()
        runtime._environment = lambda: RuntimeEnvironment(
            accelerations=frozenset({Acceleration.MTP}),
            acceleration_components={Acceleration.MTP: "/draft"},
        )
        runtime.adapter = None
        runtime._instance = None
        runtime.adapters = {}
        runtime.backend_adapters = {}
        runtime.component_loaders = {}

        with patch.object(runtime_module, "describe_model", return_value=descriptor()):
            runtime.load("/models/qwen")

        self.assertEqual(len(runtime.registry.created), 2)
        self.assertEqual(runtime.execution_plan, base_plan)
        self.assertEqual(runtime._runtime_environment.accelerations, frozenset())
        self.assertEqual(
            runtime._runtime_environment.acceleration_components, {}
        )
        self.assertIn(
            "qwen35-performance-overlay",
            runtime._runtime_environment.disabled_optimizations,
        )
        self.assertIs(runtime.adapter, runtime.registry.created[1])
        self.assertIn(runtime.registry.created[0], released)

    def test_qwen35_loader_uses_only_resolved_mtp_plan(self):
        import os
        from unittest.mock import patch

        from minimlx_runtime.adapters.qwen35 import Qwen35Adapter
        from minimlx_runtime.types import Acceleration

        adapter = Qwen35Adapter()
        model = descriptor(model_type="qwen3_5", architecture="Qwen3_5ForConditionalGeneration")
        plan = ExecutionPlan(
            phase="inference",
            accelerations=frozenset({Acceleration.MTP}),
            acceleration_components={Acceleration.MTP: "/resolved/draft"},
        )
        with (
            patch.dict(os.environ, {"MINIMLX_MTP_DRAFT_MODEL": "/unresolved/env-draft"}),
            patch.object(adapter, "_load_mtp_backend", return_value=adapter) as mtp_loader,
        ):
            self.assertIs(adapter.load(model, plan), adapter)
        mtp_loader.assert_called_once_with(model, "/resolved/draft")
        self.assertIsNone(Qwen35Adapter._mtp_draft_path(None))
        self.assertIsNone(
            Qwen35Adapter._mtp_draft_path(ExecutionPlan(phase="inference"))
        )

    def test_runtime_dispatches_through_resolved_task_family_and_backend_plan(self):
        from minimlx_runtime.runtime import Runtime

        model = object()
        request = text_request()
        chosen = ExecutionPlan(
            phase="inference",
            task=ModelTask.GENERATION,
            task_adapter="task-provider",
            family_adapter="family-provider",
            backend_adapter="backend-provider",
        )

        class ResolverStub:
            def resolve(self, descriptor, actual_request, environment):
                self.inputs = (descriptor, actual_request, environment)
                return SupportDecision(SupportStatus.SUPPORTED, "test route"), chosen

        runtime = Runtime.__new__(Runtime)
        runtime.adapter = model
        runtime.descriptor = object()
        runtime.resolver = ResolverStub()
        runtime.adapters = {"family-provider": model}
        runtime.backend_adapters = {"backend-provider": model}
        runtime.execution_plan = None
        load_environment = RuntimeEnvironment(backends=frozenset({"load-time"}))
        runtime._runtime_environment = load_environment
        runtime._environment = lambda: RuntimeEnvironment(
            backends=frozenset({"changed-later"})
        )
        called = []

        def execute(selected_model, actual_request):
            called.append((selected_model, actual_request))
            return InferenceResult(
                ModelTask.GENERATION,
                GenerationPayload("ok", [1], [2]),
            )

        runtime.task_executors = {"task-provider": execute}
        result = runtime.infer(request)
        self.assertEqual(result.payload.text, "ok")
        self.assertEqual(called, [(model, request)])
        self.assertEqual(runtime.execution_plan, chosen)
        self.assertIs(runtime.resolver.inputs[2], load_environment)

    def test_qwen_generation_executor_uses_selected_adapter_instance(self):
        from minimlx_runtime.runtime import Runtime

        class AdapterStub:
            def generate(self, prompt, max_tokens, context_tokens, prefill_step_size, session_id):
                self.args = (prompt, max_tokens, context_tokens, prefill_step_size, session_id)
                return [1], [2], "answer", 3.0, 4.0

        adapter = AdapterStub()
        request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (ContentPart(Modality.TEXT, text="question"),)),),
            options={"max_tokens": 9, "prefill_step_size": 512, "context_tokens": 100},
            session_id="session-a",
        )
        result = Runtime._execute_generation(adapter, request)
        self.assertEqual(adapter.args, ("question", 9, 100, 512, "session-a"))
        self.assertEqual(result.payload.text, "answer")
        self.assertEqual(result.metrics, {"prefill_ms": 3.0, "decode_ms": 4.0})

    def test_vlm_text_generation_forwards_fixed_context_length(self):
        from minimlx_runtime.runtime import Runtime

        class VlmAdapterStub:
            _vlm_enabled = True

            def inference_cache_metrics(self):
                return {"prefix_cache_hits": 2, "prefix_cache_matched_tokens": 512}

            def generate_multimodal(
                self,
                prompt,
                images,
                max_tokens,
                prefill_step_size,
                session_id,
                context_tokens=None,
            ):
                self.args = (
                    prompt,
                    images,
                    max_tokens,
                    prefill_step_size,
                    session_id,
                    context_tokens,
                )
                return [1] * context_tokens, [2], "answer", 3.0, 4.0

            def vision_feature_cache_metrics(self):
                return {"hits": 0, "misses": 0, "stores": 0, "entries": 0}

        adapter = VlmAdapterStub()
        request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (ContentPart(Modality.TEXT, text="question"),)),),
            options={"max_tokens": 9, "prefill_step_size": 512, "context_tokens": 1024},
            session_id="context-test",
        )

        result = Runtime._execute_generation(adapter, request)

        self.assertEqual(
            adapter.args,
            (
                [{"role": "user", "content": [{"type": "text", "text": "question"}]}],
                [],
                9,
                512,
                "context-test",
                1024,
            ),
        )
        self.assertEqual(len(result.payload.prompt_tokens), 1024)
        self.assertEqual(result.metrics["prefix_cache_hits"], 2)
        self.assertEqual(result.metrics["prefix_cache_matched_tokens"], 512)

    def test_vlm_phase_timings_derive_ttft_and_decode_from_backend_rates(self):
        from minimlx_runtime.adapters.qwen35 import _vlm_phase_timings

        self.assertEqual(
            _vlm_phase_timings(
                prompt_tokens=1024,
                generated_tokens=128,
                prompt_tokens_per_second=512.0,
                generation_tokens_per_second=64.0,
                elapsed_ms=2500.0,
            ),
            (2000.0, 2000.0),
        )

    def test_runtime_generate_runs_complete_resolver_and_executor_chain(self):
        from minimlx_runtime.registry import CapabilityResolver, builtin_provider_registry
        from minimlx_runtime.runtime import Runtime

        class AdapterStub:
            def generate(self, prompt, max_tokens, context_tokens, prefill_step_size, session_id):
                self.args = (prompt, max_tokens, context_tokens, prefill_step_size, session_id)
                return [1], [2], "answer", 3.0, 4.0

        model = AdapterStub()
        model_descriptor = descriptor(
            model_type="qwen3_5",
            architecture="Qwen3_5ForConditionalGeneration",
            profile=CapabilityProfile(
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT}),
                output_modalities=frozenset({Modality.TEXT}),
            ),
            components=(
                ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
                ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
            ),
        )
        runtime = Runtime.__new__(Runtime)
        runtime.adapter = model
        runtime.descriptor = model_descriptor
        runtime.execution_plan = None
        runtime.resolver = CapabilityResolver(builtin_provider_registry())
        runtime._environment = lambda: RuntimeEnvironment(
            backends=frozenset({"mlx-lm"})
        )
        runtime.adapters = {"qwen35-mlx-lm": model}
        runtime.backend_adapters = {"qwen35-runtime": model}
        runtime.task_executors = {"qwen35-mlx-lm": Runtime._execute_generation}

        result = runtime.generate("question", max_tokens=7, prefill_step_size=512)

        self.assertEqual(model.args, ("question", 7, None, 512, None))
        self.assertEqual(result.text, "answer")
        self.assertEqual(result.generated_tokens, [2])

    def test_compatibility_report_checks_resolved_inference_route(self):
        from tools.verify_model_support import _execution_route_summary

        plan = ExecutionPlan(
            phase="inference",
            task=ModelTask.GENERATION,
            task_adapter="qwen35-mlx-lm",
            family_adapter="qwen35-mlx-lm",
            backend_adapter="qwen35-runtime",
            accelerations=frozenset({Acceleration.MTP}),
            acceleration_adapters={Acceleration.MTP: "qwen35-mtp-draft-component"},
            acceleration_component_kinds={
                Acceleration.MTP: ComponentKind.DRAFT_WEIGHTS
            },
        )
        self.assertEqual(
            _execution_route_summary(plan, mtp_enabled=True),
            {
                "task": "generation",
                "task_adapter": "qwen35-mlx-lm",
                "family_adapter": "qwen35-mlx-lm",
                "backend_adapter": "qwen35-runtime",
                "accelerations": ["mtp"],
            },
        )
        plan_without_mtp = ExecutionPlan(
            phase="inference",
            task=ModelTask.GENERATION,
            task_adapter="qwen35-mlx-lm",
            family_adapter="qwen35-mlx-lm",
            backend_adapter="qwen35-runtime",
        )
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            _execution_route_summary(plan_without_mtp, mtp_enabled=True)

    def test_legacy_generate_api_wraps_the_typed_inference_contract(self):
        from minimlx_runtime.runtime import Runtime

        runtime = Runtime.__new__(Runtime)

        def infer(request):
            self.assertEqual(request.task, ModelTask.GENERATION)
            self.assertEqual(request.messages[0].content[0].text, "prompt")
            self.assertEqual(
                request.options,
                {"max_tokens": 5, "context_tokens": 4096, "prefill_step_size": 512},
            )
            self.assertEqual(request.session_id, "session-1")
            return InferenceResult(
                ModelTask.GENERATION,
                GenerationPayload("answer", [1, 2], [3, 4]),
                metrics={"prefill_ms": 12.0, "decode_ms": 34.0},
            )

        runtime.infer = infer
        result = runtime.generate(
            "prompt",
            max_tokens=5,
            context_tokens=4096,
            prefill_step_size=512,
            session_id="session-1",
        )
        self.assertEqual(result, GenerationResult([1, 2], [3, 4], "answer", 12.0, 34.0))

    def test_framework_contract_enumerates_all_declared_tasks_and_modalities(self):
        self.assertEqual(
            {task.value for task in ModelTask},
            {
                "generation",
                "embedding",
                "reranking",
                "speech_to_text",
                "text_to_speech",
                "speech_to_speech",
            },
        )
        self.assertEqual(
            {modality.value for modality in Modality},
            {"text", "image", "audio", "video", "structured"},
        )
        for modality in (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO):
            request = InferenceRequest(
                ModelTask.GENERATION,
                (InputMessage("user", (ContentPart(modality, uri="file:///payload"),)),),
            )
            self.assertIs(request.messages[0].content[0].modality, modality)
        structured = InferenceRequest(
            ModelTask.RERANKING,
            (
                InputMessage(
                    "user",
                    (ContentPart(Modality.STRUCTURED, data={"query": "q", "documents": []}),),
                ),
            ),
        )
        self.assertEqual(structured.messages[0].content[0].data, {"query": "q", "documents": []})

    def test_support_record_schema_tracks_framework_task_modality_and_states(self):
        schema_path = Path(__file__).resolve().parents[1] / "design" / "model_support_record.schema.json"
        schema = json.loads(schema_path.read_text())
        key_properties = schema["properties"]["key"]["properties"]
        task_schema = key_properties["task"]["enum"]
        modality_schema = schema["$defs"]["modalities"]["items"]["enum"]
        state_schema = schema["properties"]["state"]["enum"]

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(set(task_schema), {task.value for task in ModelTask})
        self.assertEqual(set(modality_schema), {modality.value for modality in Modality})
        self.assertTrue(
            {"discovered", "classified", "loadable", "smoke_pass", "task_correct",
             "context_or_shape_pass", "benchmarked"}.issubset(state_schema)
        )
        self.assertTrue(
            {"unsupported", "ambiguous", "missing_component", "dependency_blocked",
             "hardware_blocked"}.issubset(state_schema)
        )
        self.assertIn("evidence", schema["required"])

    def test_registry_can_compose_routes_for_each_declared_task_type(self):
        task_modalities = {
            ModelTask.GENERATION: Modality.TEXT,
            ModelTask.EMBEDDING: Modality.TEXT,
            ModelTask.RERANKING: Modality.TEXT,
            ModelTask.SPEECH_TO_TEXT: Modality.AUDIO,
            ModelTask.TEXT_TO_SPEECH: Modality.TEXT,
            ModelTask.SPEECH_TO_SPEECH: Modality.AUDIO,
        }
        registrations = []
        for task, modality in task_modalities.items():
            registrations.append(
                AdapterRegistration(
                    f"task-{task.value}",
                    frozenset({AdapterRole.TASK}),
                    tasks=frozenset({task}),
                    input_modalities=frozenset({modality}),
                )
            )
        registrations.extend(
            (
                AdapterRegistration(
                    "family-example", frozenset({AdapterRole.FAMILY}), families=("example",)
                ),
                AdapterRegistration(
                    "backend-example", frozenset({AdapterRole.BACKEND}), backend="mlx-test"
                ),
            )
        )
        resolver = CapabilityResolver(ProviderRegistry(registrations))
        all_tasks = frozenset(task_modalities)
        all_modalities = frozenset(Modality)
        model = descriptor(
            profile=CapabilityProfile(tasks=all_tasks, input_modalities=all_modalities)
        )
        for task, modality in task_modalities.items():
            request = InferenceRequest(
                task,
                (InputMessage("user", (ContentPart(modality, text="x") if modality is Modality.TEXT else ContentPart(modality, uri="file:///x"),)),),
            )
            decision, plan = resolver.resolve(model, request, RuntimeEnvironment())
            self.assertEqual(decision.status, SupportStatus.SUPPORTED, task.value)
            self.assertEqual(plan.task_adapter, f"task-{task.value}")

    def test_resolver_task_by_payload_kind_matrix_accepts_only_declared_pairs(self):
        supported_pairs = {
            ModelTask.GENERATION: {Modality.TEXT, Modality.IMAGE, Modality.VIDEO},
            ModelTask.EMBEDDING: {Modality.TEXT, Modality.IMAGE},
            ModelTask.RERANKING: {Modality.TEXT, Modality.STRUCTURED},
            ModelTask.SPEECH_TO_TEXT: {Modality.AUDIO},
            ModelTask.TEXT_TO_SPEECH: {Modality.TEXT},
            ModelTask.SPEECH_TO_SPEECH: {Modality.TEXT, Modality.AUDIO},
        }
        all_tasks = frozenset(supported_pairs)
        all_modalities = frozenset(Modality)
        output_modalities = frozenset({Modality.TEXT, Modality.AUDIO})
        required_by_task = {
            ModelTask.GENERATION: frozenset(
                {ComponentKind.TOKENIZER, ComponentKind.TEXT_WEIGHTS}
            ),
            ModelTask.EMBEDDING: frozenset({ComponentKind.TEXT_WEIGHTS}),
            ModelTask.RERANKING: frozenset(
                {ComponentKind.TOKENIZER, ComponentKind.TEXT_WEIGHTS}
            ),
            ModelTask.SPEECH_TO_TEXT: frozenset(
                {ComponentKind.PROCESSOR, ComponentKind.AUDIO_ENCODER}
            ),
            ModelTask.TEXT_TO_SPEECH: frozenset(
                {ComponentKind.AUDIO_DECODER, ComponentKind.VOCODER}
            ),
            ModelTask.SPEECH_TO_SPEECH: frozenset(
                {ComponentKind.AUDIO_ENCODER, ComponentKind.AUDIO_DECODER, ComponentKind.VOCODER}
            ),
        }
        component_kinds = frozenset(
            kind for kinds in required_by_task.values() for kind in kinds
        )
        complete_components = tuple(
            ArtifactComponent(kind.value, kind, f"{kind.value}.safetensors")
            for kind in component_kinds
        )
        registrations = [
            AdapterRegistration(
                f"task-{task.value}",
                frozenset({AdapterRole.TASK}),
                tasks=frozenset({task}),
                input_modalities=frozenset(modalities),
                output_modalities=output_modalities,
                backend=f"backend-{task.value}",
                required_components=required_by_task[task],
            )
            for task, modalities in supported_pairs.items()
        ]
        registrations.extend(
            (
                AdapterRegistration(
                    "family-matrix",
                    frozenset({AdapterRole.FAMILY}),
                    families=("example",),
                    output_modalities=output_modalities,
                ),
                *(
                    AdapterRegistration(
                        f"backend-{task.value}",
                        frozenset({AdapterRole.BACKEND}),
                        backend=f"backend-{task.value}",
                        output_modalities=output_modalities,
                    )
                    for task in supported_pairs
                ),
            )
        )
        registrations.extend(
            AdapterRegistration(
                f"component-{kind.value}",
                frozenset({AdapterRole.COMPONENT}),
                component_kind=kind,
            )
            for kind in component_kinds
        )
        model = descriptor(
            components=complete_components,
            profile=CapabilityProfile(
                tasks=all_tasks,
                input_modalities=all_modalities,
                output_modalities=output_modalities,
            )
        )
        resolver = CapabilityResolver(ProviderRegistry(registrations))

        for task, allowed_modalities in supported_pairs.items():
            for modality in Modality:
                if modality is Modality.TEXT:
                    content = ContentPart(modality, text="payload")
                elif modality is Modality.STRUCTURED:
                    content = ContentPart(modality, data={"query": "q"})
                else:
                    content = ContentPart(modality, uri=f"file:///payload-{modality.value}")
                request = InferenceRequest(task, (InputMessage("user", (content,)),))
                decision, plan = resolver.resolve(model, request, RuntimeEnvironment())
                expected = modality in allowed_modalities
                with self.subTest(task=task.value, modality=modality.value):
                    self.assertEqual(
                        decision.status,
                        SupportStatus.SUPPORTED if expected else SupportStatus.UNSUPPORTED,
                    )
                    self.assertEqual(plan is not None, expected)
                    if expected:
                        self.assertEqual(plan.backend_adapter, f"backend-{task.value}")
                        self.assertEqual(
                            set(plan.component_adapters), required_by_task[task]
                        )

        for component_kind in component_kinds:
            task = next(
                task
                for task, kinds in required_by_task.items()
                if component_kind in kinds
            )
            modality = next(iter(supported_pairs[task]))
            content = (
                ContentPart(modality, text="payload")
                if modality is Modality.TEXT
                else ContentPart(modality, uri="file:///payload")
            )
            incomplete_model = descriptor(
                components=tuple(
                    component
                    for component in complete_components
                    if component.kind is not component_kind
                ),
                profile=model.support,
            )
            missing, plan = resolver.resolve(
                incomplete_model,
                InferenceRequest(task, (InputMessage("user", (content,)),)),
                RuntimeEnvironment(),
            )
            with self.subTest(missing_component=component_kind.value):
                self.assertEqual(missing.status, SupportStatus.MISSING_COMPONENT)
                self.assertIn(component_kind.value, missing.missing_components)
                self.assertIsNone(plan)

    def test_discovery_reports_missing_indexed_shards_and_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.layers.0.weight": "missing-00001.safetensors"}})
            )
            components = describe_model(path).components
            missing = [item for item in components if item.kind is ComponentKind.WEIGHTS_SHARD]
            self.assertEqual(len(missing), 1)
            self.assertFalse(missing[0].present)

            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.layers.0.weight": "../escape.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "escapes model directory"):
                describe_model(path)

    def test_multimodal_config_without_encoder_or_processor_reports_missing_parts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "example_vlm", "vision_config": {"hidden_size": 64}})
            )
            found = describe_model(path)
        missing = {item.kind for item in found.components if not item.present}
        self.assertIn(ComponentKind.VISION_ENCODER, missing)
        self.assertIn(ComponentKind.PROCESSOR, missing)

    def test_hardware_block_is_distinct_from_missing_dependencies(self):
        registration = AdapterRegistration(
            "needs-neural-engine",
            frozenset({AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}),
            tasks=frozenset({ModelTask.GENERATION}),
            families=("example",),
            required_features=frozenset({"neural_engine"}),
            minimum_memory_bytes=1024,
        )
        decision, plan = CapabilityResolver(ProviderRegistry([registration])).resolve(
            descriptor(), text_request(), RuntimeEnvironment(available_memory_bytes=512)
        )
        self.assertEqual(decision.status, SupportStatus.HARDWARE_BLOCKED)
        self.assertEqual(decision.missing_features, ("neural_engine",))
        self.assertEqual(decision.required_memory_bytes, 1024)
        self.assertEqual(decision.missing_dependencies, ())
        self.assertIsNone(plan)

    def test_acceleration_is_negotiated_separately_from_model_family(self):
        request = InferenceRequest(
            ModelTask.GENERATION,
            text_request().messages,
            accelerations=frozenset({Acceleration.MTP}),
        )
        resolver = CapabilityResolver(builtin_provider_registry())
        missing_drafter, _ = resolver.resolve(
            descriptor(
                components=(
                    ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
                    ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
                ),
                profile=CapabilityProfile(tasks=frozenset({ModelTask.GENERATION})),
                model_type="qwen3_5",
                architecture="Qwen3_5ForConditionalGeneration",
                metadata={"hidden_size": 2560, "vocab_size": 248320},
            ),
            request,
            RuntimeEnvironment(backends=frozenset({"mlx-lm"})),
        )
        self.assertEqual(missing_drafter.status, SupportStatus.DEPENDENCY_BLOCKED)
        self.assertIn("mtp drafter/component", missing_drafter.missing_dependencies)

        resolved, plan = resolver.resolve(
            descriptor(
                components=(
                    ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
                    ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
                ),
                profile=CapabilityProfile(tasks=frozenset({ModelTask.GENERATION})),
                model_type="qwen3_5",
                architecture="Qwen3_5ForConditionalGeneration",
                metadata={"hidden_size": 2560, "vocab_size": 248320},
            ),
            request,
            RuntimeEnvironment(
                backends=frozenset({"mlx-lm", "mlx-vlm"}),
                backend_versions={"mlx-vlm": "0.7.2"},
                accelerations=frozenset({Acceleration.MTP}),
                acceleration_components={Acceleration.MTP: "/models/draft"},
                acceleration_artifacts={Acceleration.MTP: mtp_draft_descriptor()},
            ),
        )
        self.assertEqual(resolved.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.accelerations, {Acceleration.MTP})
        self.assertEqual(
            plan.acceleration_adapters,
            {Acceleration.MTP: "qwen35-mtp-draft-component"},
        )
        self.assertEqual(
            plan.acceleration_component_kinds,
            {Acceleration.MTP: ComponentKind.DRAFT_WEIGHTS},
        )
        self.assertEqual(
            plan.acceleration_components,
            {Acceleration.MTP: "/models/draft"},
        )
        self.assertEqual(plan.component_paths[ComponentKind.DRAFT_WEIGHTS], ("/models/draft",))

        wrong_backend_version, _ = resolver.resolve(
            descriptor(
                components=(
                    ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
                    ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
                ),
                profile=CapabilityProfile(tasks=frozenset({ModelTask.GENERATION})),
                model_type="qwen3_5",
                architecture="Qwen3_5ForConditionalGeneration",
                metadata={"hidden_size": 2560, "vocab_size": 248320},
            ),
            request,
            RuntimeEnvironment(
                backends=frozenset({"mlx-lm", "mlx-vlm"}),
                backend_versions={"mlx-vlm": "0.7.1"},
                accelerations=frozenset({Acceleration.MTP}),
                acceleration_components={Acceleration.MTP: "/models/draft"},
                acceleration_artifacts={Acceleration.MTP: mtp_draft_descriptor()},
            ),
        )
        self.assertEqual(wrong_backend_version.status, SupportStatus.DEPENDENCY_BLOCKED)
        self.assertEqual(
            wrong_backend_version.missing_dependencies,
            ("mlx-vlm==0.7.2 (found 0.7.1)",),
        )

    def test_qwen35_dflash_target_resolves_dflash_component_provider(self):
        model = descriptor(
            components=(
                ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
                ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
            ),
            profile=CapabilityProfile(tasks=frozenset({ModelTask.GENERATION})),
            model_type="qwen3_5",
            architecture="Qwen3_5ForConditionalGeneration",
            metadata={"hidden_size": 5120, "vocab_size": 248320},
        )
        request = InferenceRequest(
            ModelTask.GENERATION,
            text_request().messages,
            accelerations=frozenset({Acceleration.DFLASH}),
        )
        decision, plan = CapabilityResolver(builtin_provider_registry()).resolve(
            model,
            request,
            RuntimeEnvironment(
                backends=frozenset({"mlx-lm", "dflash"}),
                accelerations=frozenset({Acceleration.DFLASH}),
                acceleration_components={Acceleration.DFLASH: "/models/qwen-dflash"},
                acceleration_artifacts={Acceleration.DFLASH: dflash_draft_descriptor()},
            ),
        )
        self.assertEqual(decision.status, SupportStatus.SUPPORTED)
        self.assertEqual(plan.accelerations, {Acceleration.DFLASH})
        self.assertEqual(
            plan.acceleration_adapters,
            {Acceleration.DFLASH: "qwen35-dflash-draft-component"},
        )

    def test_qwen_base_route_requires_mlx_lm_and_mtp_adds_mlx_vlm_dependency(self):
        model = descriptor(
            components=(
                ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
                ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
            ),
            profile=CapabilityProfile(tasks=frozenset({ModelTask.GENERATION})),
            model_type="qwen3_5",
            architecture="Qwen3_5ForConditionalGeneration",
            metadata={"hidden_size": 2560, "vocab_size": 248320},
        )
        resolver = CapabilityResolver(builtin_provider_registry())

        base, base_plan = resolver.resolve(
            model, text_request(), RuntimeEnvironment(backends=frozenset({"mlx-lm"}))
        )
        self.assertEqual(base.status, SupportStatus.SUPPORTED)
        self.assertEqual(base_plan.accelerations, frozenset())

        mtp_request = InferenceRequest(
            ModelTask.GENERATION,
            text_request().messages,
            accelerations=frozenset({Acceleration.MTP}),
        )
        mtp_missing_vlm, _ = resolver.resolve(
            model,
            mtp_request,
            RuntimeEnvironment(
                backends=frozenset({"mlx-lm"}),
                accelerations=frozenset({Acceleration.MTP}),
                acceleration_components={Acceleration.MTP: "/models/draft"},
                acceleration_artifacts={Acceleration.MTP: mtp_draft_descriptor()},
            ),
        )
        self.assertEqual(mtp_missing_vlm.status, SupportStatus.DEPENDENCY_BLOCKED)
        self.assertEqual(mtp_missing_vlm.missing_dependencies, ("mlx-vlm",))

    def test_runtime_environment_only_advertises_mtp_for_complete_checkpoint(self):
        import os
        from unittest.mock import patch

        from minimlx_runtime.runtime import Runtime

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "config.json").write_text(json.dumps({"model_type": "qwen3_5_mtp"}))
            complete = root / "complete"
            complete.mkdir()
            (complete / "config.json").write_text(json.dumps({"model_type": "qwen3_5_mtp"}))
            (complete / "draft_model.safetensors").touch()

            with patch.dict(os.environ, {"MINIMLX_MTP_DRAFT_MODEL": str(incomplete)}):
                incomplete_environment = Runtime._environment()
            with patch.dict(os.environ, {"MINIMLX_MTP_DRAFT_MODEL": str(complete)}):
                complete_environment = Runtime._environment()

        self.assertNotIn(Acceleration.MTP, incomplete_environment.accelerations)
        self.assertEqual(
            complete_environment.acceleration_components[Acceleration.MTP],
            str(complete.resolve()),
        )
        self.assertEqual(
            complete_environment.acceleration_artifacts[Acceleration.MTP].model_type,
            "qwen3_5_mtp",
        )

    def test_acceleration_artifact_requirement_rejects_wrong_family_and_missing_role(self):
        requirement = AccelerationArtifactRequirement(
            model_types=frozenset({"qwen3_5_mtp"}),
            component_kinds=frozenset({ComponentKind.DRAFT_WEIGHTS, ComponentKind.WEIGHTS_SHARD}),
            matching_metadata_keys=("hidden_size", "vocab_size"),
        )
        self.assertTrue(
            CapabilityResolver._acceleration_artifact_matches(
                requirement, mtp_draft_descriptor(), mtp_draft_descriptor()
            )
        )
        self.assertFalse(
            CapabilityResolver._acceleration_artifact_matches(
                requirement,
                mtp_draft_descriptor(model_type="other_mtp"),
                mtp_draft_descriptor(),
            )
        )
        mismatched_size = descriptor(
            model_type="qwen3_5_mtp",
            architecture="Qwen3_5MTPForCausalLM",
            metadata={"hidden_size": 1024, "vocab_size": 248320},
            components=(
                ArtifactComponent("draft", ComponentKind.DRAFT_WEIGHTS, "draft.safetensors"),
                ArtifactComponent("draft", ComponentKind.WEIGHTS_SHARD, "draft.safetensors"),
            ),
        )
        self.assertFalse(
            CapabilityResolver._acceleration_artifact_matches(
                requirement, mismatched_size, mtp_draft_descriptor()
            )
        )
        self.assertFalse(
            CapabilityResolver._acceleration_artifact_matches(
                requirement,
                descriptor(
                    model_type="qwen3_5_mtp",
                    components=(
                        ArtifactComponent("weights", ComponentKind.WEIGHTS_SHARD, "weights.safetensors"),
                    ),
                ),
                mtp_draft_descriptor(),
            )
        )

    def test_mtp_route_blocks_wrong_drafter_when_requested_and_omits_it_for_auto(self):
        model = descriptor(
            components=(
                ArtifactComponent("tokenizer", ComponentKind.TOKENIZER, "tokenizer.json"),
                ArtifactComponent("weights", ComponentKind.TEXT_WEIGHTS, "model.safetensors"),
            ),
            profile=CapabilityProfile(tasks=frozenset({ModelTask.GENERATION})),
            model_type="qwen3_5",
            architecture="Qwen3_5ForConditionalGeneration",
            metadata={"hidden_size": 2560, "vocab_size": 248320},
        )
        resolver = CapabilityResolver(builtin_provider_registry())
        wrong_drafter_environment = RuntimeEnvironment(
            backends=frozenset({"mlx-lm", "mlx-vlm"}),
            backend_versions={"mlx-vlm": "0.7.2"},
            accelerations=frozenset({Acceleration.MTP}),
            acceleration_components={Acceleration.MTP: "/models/wrong-draft"},
            acceleration_artifacts={
                Acceleration.MTP: mtp_draft_descriptor(
                    model_type="qwen2_mtp", architecture="Qwen2MTPForCausalLM"
                )
            },
        )
        explicit_request = InferenceRequest(
            ModelTask.GENERATION,
            text_request().messages,
            options={"accelerations": ["mtp"]},
        )
        blocked, no_plan = resolver.resolve(model, explicit_request, wrong_drafter_environment)
        self.assertEqual(blocked.status, SupportStatus.DEPENDENCY_BLOCKED)
        self.assertIn("mtp artifact identity", blocked.missing_dependencies[0])
        self.assertIsNone(no_plan)

        normal, normal_plan = resolver.resolve(
            model, text_request(), wrong_drafter_environment
        )
        self.assertEqual(normal.status, SupportStatus.SUPPORTED)
        self.assertEqual(normal_plan.accelerations, frozenset())
        self.assertNotIn(ComponentKind.DRAFT_WEIGHTS, normal_plan.component_paths)

    def test_resolver_reports_missing_component_and_dependency(self):
        required = AdapterRegistration(
            "vision-provider",
            frozenset({AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}),
            tasks=frozenset({ModelTask.GENERATION}),
            families=("example",),
            required_components=frozenset({ComponentKind.VISION_ENCODER}),
            required_dependencies=frozenset({"vision-backend"}),
        )
        resolver = CapabilityResolver(ProviderRegistry([required]))
        missing_component, _ = resolver.resolve(
            descriptor(), text_request(), RuntimeEnvironment(backends=frozenset({"vision-backend"}))
        )
        self.assertEqual(missing_component.status, SupportStatus.MISSING_COMPONENT)
        self.assertEqual(missing_component.missing_components, ("vision_encoder",))

        complete = descriptor(
            components=(
                ArtifactComponent("vision", ComponentKind.VISION_ENCODER, "vision.safetensors"),
            )
        )
        resolver.providers.register(
            AdapterRegistration(
                "vision-loader",
                frozenset({AdapterRole.COMPONENT}),
                component_kind=ComponentKind.VISION_ENCODER,
            )
        )
        missing_dependency, _ = resolver.resolve(
            complete, text_request(), RuntimeEnvironment()
        )
        self.assertEqual(missing_dependency.status, SupportStatus.DEPENDENCY_BLOCKED)

    def test_known_task_with_unsupported_modality_is_rejected(self):
        registration = AdapterRegistration(
            "text-only",
            frozenset({AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}),
            tasks=frozenset({ModelTask.GENERATION}),
            input_modalities=frozenset({Modality.TEXT}),
            families=("example",),
        )
        request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (ContentPart(Modality.IMAGE, uri="file:///tmp/a.png"),)),),
        )
        known = descriptor(profile=CapabilityProfile(tasks=frozenset({ModelTask.GENERATION})))
        decision, plan = CapabilityResolver(ProviderRegistry([registration])).resolve(
            known, request, RuntimeEnvironment()
        )
        self.assertEqual(decision.status, SupportStatus.UNSUPPORTED)
        self.assertIsNone(plan)

    def test_task_output_modality_must_be_supported_by_provider_and_model(self):
        registration = AdapterRegistration(
            "audio-only-output",
            frozenset({AdapterRole.TASK, AdapterRole.FAMILY, AdapterRole.BACKEND}),
            tasks=frozenset({ModelTask.GENERATION}),
            input_modalities=frozenset({Modality.TEXT}),
            output_modalities=frozenset({Modality.AUDIO}),
            families=("example",),
        )
        model = descriptor(
            profile=CapabilityProfile(
                tasks=frozenset({ModelTask.GENERATION}),
                input_modalities=frozenset({Modality.TEXT}),
                output_modalities=frozenset({Modality.TEXT}),
            )
        )
        decision, plan = CapabilityResolver(ProviderRegistry([registration])).resolve(
            model, text_request(), RuntimeEnvironment()
        )

        self.assertEqual(decision.status, SupportStatus.UNSUPPORTED)
        self.assertIsNone(plan)

    def test_typed_request_and_tagged_result_round_trip(self):
        request = inference_request_from_dict(
            {
                "task": "generation",
                "messages": [
                    {"role": "user", "content": [{"modality": "text", "text": "hello"}]}
                ],
                "options": {"max_tokens": 8},
            }
        )
        self.assertEqual(request.messages[0].content[0].text, "hello")
        original = InferenceResult(
            ModelTask.EMBEDDING,
            EmbeddingPayload([[0.1, 0.2]]),
            usage={"prompt_tokens": 2},
        )
        decoded = inference_result_from_dict(inference_result_to_dict(original))
        self.assertEqual(decoded, original)

    def test_image_request_preserves_base64_and_content_order(self):
        from minimlx_runtime.runtime import Runtime

        request = inference_request_from_dict(
            {
                "task": "generation",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"modality": "text", "text": "before"},
                            {
                                "modality": "image",
                                "data_base64": "iVBORw0KGgo=",
                                "mime_type": "image/png",
                            },
                            {"modality": "text", "text": "after"},
                        ],
                    }
                ],
            }
        )
        parts = request.messages[0].content
        self.assertEqual(parts[1].data_base64, "iVBORw0KGgo=")
        self.assertEqual(
            Runtime._vlm_messages(request),
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image"},
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
        )

    def test_video_request_preserves_content_order_and_frame_timestamps(self):
        from minimlx_runtime.runtime import Runtime
        from minimlx_runtime.types import VideoFrame

        request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (
                ContentPart(Modality.TEXT, text="Summarize this clip: "),
                ContentPart(Modality.VIDEO, frames=(
                    VideoFrame(0.0, data_base64="aW1hZ2Uw", mime_type="image/png"),
                    VideoFrame(1.5, data_base64="aW1hZ2Ux", mime_type="image/png"),
                )),
                ContentPart(Modality.TEXT, text=" Focus on the change."),
            )),),
        )
        self.assertEqual(
            Runtime._vlm_messages(request),
            [{"role": "user", "content": [
                {"type": "text", "text": "Summarize this clip: "},
                {"type": "video"},
                {"type": "text", "text": " Focus on the change."},
            ]}],
        )
        self.assertEqual([frame.timestamp_seconds for frame in request.messages[0].content[1].frames], [0.0, 1.5])

    def test_shared_vlm_preparation_passes_video_metadata_and_template_kwargs(self):
        import base64
        from io import BytesIO
        from unittest.mock import patch
        from PIL import Image
        from minimlx_runtime.adapters.mlx_vlm import MLXVLMAdapter
        from minimlx_runtime.types import VideoFrame

        image = Image.new("RGB", (8, 8), "red")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        seen = {}

        def apply_template(_processor, _config, _messages, **kwargs):
            seen["template_kwargs"] = kwargs
            return "rendered prompt"

        def prepare_inputs(_processor, **kwargs):
            seen["prepared"] = kwargs
            return {"input_ids": object(), "pixel_values_videos": object(), "video_grid_thw": object()}

        prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
        prompt_utils.apply_chat_template = apply_template
        utils = types.ModuleType("mlx_vlm.utils")
        utils.prepare_inputs = prepare_inputs
        utils.should_add_special_tokens = lambda *_args: False
        mlx_vlm = types.ModuleType("mlx_vlm")
        mlx_vlm.__path__ = []
        adapter = MLXVLMAdapter()
        adapter.model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="qwen3_5"))
        adapter.processor = object()
        request_messages = [{"role": "user", "content": [{"type": "video"}]}]
        video_parts = [ContentPart(Modality.VIDEO, frames=(
            VideoFrame(0.0, data_base64=encoded, mime_type="image/png"),
            VideoFrame(0.5, data_base64=encoded, mime_type="image/png"),
        ))]
        with patch.dict(sys.modules, {
            "mlx_vlm": mlx_vlm,
            "mlx_vlm.prompt_utils": prompt_utils,
            "mlx_vlm.utils": utils,
        }):
            result = adapter._prepare(
                request_messages,
                [],
                video_parts,
                {"enable_thinking": False},
            )

        self.assertEqual(seen["template_kwargs"], {"num_images": 0, "enable_thinking": False})
        self.assertEqual(seen["prepared"]["prompts"], "rendered prompt")
        self.assertEqual(seen["prepared"]["video_metadata"][0]["frames_indices"], [0, 1])
        self.assertEqual(seen["prepared"]["videos"][0].shape, (2, 8, 8, 3))
        self.assertEqual(result[0], "rendered prompt")
        self.assertIn("pixel_values_videos", result[-1])

    def test_video_template_uses_native_processor_and_preserves_typed_content(self):
        from minimlx_runtime.adapters.mlx_vlm import _apply_vlm_chat_template

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "native rendered video prompt"

        processor = Processor()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "before"},
            {"type": "image"},
            {"type": "video"},
            {"type": "text", "text": "after"},
        ]}]
        prompt = _apply_vlm_chat_template(
            processor,
            object(),
            messages,
            1,
            [object()],
            {"enable_thinking": False},
        )
        self.assertEqual(prompt, "native rendered video prompt")
        self.assertEqual(processor.messages, messages)
        self.assertEqual(
            processor.kwargs,
            {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
        )

    def test_qwen35_mixed_image_video_is_rejected_with_actionable_reason(self):
        from minimlx_runtime.runtime import Runtime

        class AdapterStub:
            model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="qwen3_5"))

        request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (
                ContentPart(Modality.IMAGE, data_base64="aW1hZ2U="),
                ContentPart(Modality.VIDEO, data_base64="dmlkZW8=", mime_type="video/mp4"),
            )),),
        )
        with self.assertRaisesRegex(NotImplementedError, "mixing image and video"):
            Runtime._execute_generation(AdapterStub(), request)

    def test_vlm_executor_forwards_video_and_template_kwargs(self):
        from minimlx_runtime.runtime import Runtime
        from minimlx_runtime.types import VideoFrame

        class VlmAdapterStub:
            _vlm_enabled = True

            def generate_multimodal(self, messages, images, max_tokens, prefill_step_size, session_id, **kwargs):
                self.args = (messages, images, kwargs)
                return [1], [2], "answer", 3.0, 4.0

            def vision_feature_cache_metrics(self):
                return {}

        adapter = VlmAdapterStub()
        video = ContentPart(Modality.VIDEO, frames=(VideoFrame(0, data_base64="aW1hZ2U="),))
        request = InferenceRequest(
            ModelTask.GENERATION,
            (InputMessage("user", (ContentPart(Modality.TEXT, text="Describe"), video)),),
            options={"chat_template_kwargs": {"enable_thinking": False}},
        )
        Runtime._execute_generation(adapter, request)
        messages, images, kwargs = adapter.args
        self.assertEqual(messages[0]["content"][-1], {"type": "video"})
        self.assertEqual(images, [])
        self.assertEqual(kwargs["video_parts"], [video])
        self.assertEqual(kwargs["chat_template_kwargs"], {"enable_thinking": False})

    def test_remote_images_are_rejected_by_default(self):
        from minimlx_runtime.adapters.qwen35 import Qwen35Adapter

        with self.assertRaisesRegex(ValueError, "remote images are disabled"):
            Qwen35Adapter._image_input(
                ContentPart(Modality.IMAGE, uri="https://example.invalid/image.png")
            )

    def test_image_count_limit_is_enforced_before_backend_call(self):
        from minimlx_runtime.runtime import Runtime

        class FakeAdapter:
            def generate_multimodal(self, *args):
                raise AssertionError("backend must not receive an oversized request")

        request = InferenceRequest(
            ModelTask.GENERATION,
            (
                InputMessage(
                    "user",
                    tuple(
                        ContentPart(Modality.IMAGE, uri=f"file:///tmp/{index}.png")
                        for index in range(17)
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "maximum is 16"):
            Runtime._execute_generation(FakeAdapter(), request)

    def test_task_result_type_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires EmbeddingPayload"):
            InferenceResult(ModelTask.EMBEDDING, GenerationPayload("not an embedding"))

    def test_protocol_rejects_unknown_major_schema_version(self):
        with self.assertRaisesRegex(ValueError, "unsupported schema_version 2"):
            inference_request_from_dict(
                {
                    "schema_version": 2,
                    "task": "generation",
                    "messages": [
                        {"role": "user", "content": [{"modality": "text", "text": "hello"}]}
                    ],
                }
            )

    def test_every_task_result_has_a_tagged_round_trip_contract(self):
        from minimlx_runtime.types import (
            AudioPayload,
            RerankingPayload,
            SpeechToSpeechPayload,
            TranscriptionPayload,
        )

        results = (
            InferenceResult(ModelTask.GENERATION, GenerationPayload("text", [1], [2])),
            InferenceResult(ModelTask.EMBEDDING, EmbeddingPayload([[0.1, 0.2]])),
            InferenceResult(ModelTask.RERANKING, RerankingPayload([0.9], [0])),
            InferenceResult(ModelTask.SPEECH_TO_TEXT, TranscriptionPayload("words")),
            InferenceResult(
                ModelTask.TEXT_TO_SPEECH,
                AudioPayload("AQID", "audio/wav", sample_rate=24000),
            ),
            InferenceResult(
                ModelTask.SPEECH_TO_SPEECH,
                SpeechToSpeechPayload(
                    "words", AudioPayload("AQID", "audio/wav", sample_rate=24000)
                ),
            ),
        )

        for original in results:
            encoded = inference_result_to_dict(original)
            decoded = inference_result_from_dict(encoded)
            self.assertEqual(decoded, original, original.task.value)

    def test_result_protocol_rejects_malformed_payload_and_usage_shapes(self):
        base = {
            "schema_version": 1,
            "task": "generation",
            "payload_type": "GenerationPayload",
            "payload": {"text": "ok"},
        }
        for patch, message in (
            ({"payload": ["not", "an", "object"]}, "payload must be an object"),
            ({"usage": {"completion_tokens": 1.5}}, "usage must map"),
            ({"metrics": {"decode_ms": "fast"}}, "metrics must map"),
            ({"metadata": []}, "metadata must be an object"),
        ):
            with self.subTest(patch=patch), self.assertRaisesRegex(ValueError, message):
                inference_result_from_dict({**base, **patch})

    def test_protocol_rejects_wrong_stream_type_and_mismatched_result_tag(self):
        base = {
            "task": "generation",
            "messages": [
                {"role": "user", "content": [{"modality": "text", "text": "hi"}]}
            ],
        }
        with self.assertRaisesRegex(ValueError, "stream must be a boolean"):
            inference_request_from_dict({**base, "stream": "false"})
        with self.assertRaisesRegex(ValueError, "payload_type.*does not match"):
            inference_result_from_dict(
                {
                    "task": "generation",
                    "payload_type": "EmbeddingPayload",
                    "payload": {"text": "hi"},
                }
            )


if __name__ == "__main__":
    unittest.main()
