import json
import tempfile
import unittest
from pathlib import Path

from minimlx_runtime.registry import describe_model
from minimlx_runtime.types import (
    Acceleration,
    ArtifactComponent,
    CapabilityProfile,
    ComponentKind,
    DetectionEvidence,
    ExecutionPlan,
    GenerationResult,
    ModelDescriptor,
    ModelStructure,
    ModelTask,
    Modality,
)


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.model_dir = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _write_config(self, config):
        (self.model_dir / "config.json").write_text(json.dumps(config))

    def test_descriptor_reads_qwen35_config(self):
        self._write_config(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForCausalLM"],
                "hidden_size": 1024,
            }
        )
        descriptor = describe_model(self.model_dir)
        self.assertEqual(descriptor.model_type, "qwen3_5")
        self.assertIn("hybrid_attention", descriptor.capabilities)

    def test_descriptor_reads_nested_text_config_for_qwen35_multimodal(self):
        self._write_config(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "dtype": "bfloat16",
                    "hidden_size": 2560,
                    "num_hidden_layers": 32,
                },
            }
        )
        descriptor = describe_model(self.model_dir)
        self.assertEqual(descriptor.model_type, "qwen3_5")
        self.assertEqual(descriptor.metadata["dtype"], "bfloat16")
        self.assertEqual(descriptor.metadata["hidden_size"], 2560)
        self.assertEqual(descriptor.metadata["num_hidden_layers"], 32)

    def test_descriptor_exposes_qwen35_moe_capabilities(self):
        self._write_config(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_hidden_layers": 40,
                    "num_experts": 128,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": 512,
                },
            }
        )
        descriptor = describe_model(self.model_dir)
        self.assertIn("moe", descriptor.capabilities)
        self.assertIn("hybrid_attention", descriptor.capabilities)
        self.assertEqual(descriptor.metadata["num_experts"], 128)
        self.assertEqual(descriptor.metadata["num_experts_per_tok"], 8)
        self.assertEqual(descriptor.metadata["moe_intermediate_size"], 512)

    def test_execution_plan_defaults_to_single_request_decode(self):
        plan = ExecutionPlan(phase="decode")
        self.assertEqual(plan.batch_size, 1)
        self.assertFalse(plan.use_continuous_batching)

    def test_generation_result_calculates_throughput(self):
        result = GenerationResult([1], [2, 3], "ok", 1.0, 20.0)
        self.assertEqual(result.decode_tokens_per_second, 100.0)

    def test_capability_profile_keeps_dimensions_orthogonal(self):
        profile = CapabilityProfile(
            tasks=frozenset({ModelTask.GENERATION}),
            input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
            output_modalities=frozenset({Modality.TEXT}),
            structures=frozenset({ModelStructure.MOE, ModelStructure.HYBRID_ATTENTION}),
            accelerations=frozenset({Acceleration.MTP}),
        )
        self.assertEqual(profile.tasks, {ModelTask.GENERATION})
        self.assertIn(Modality.IMAGE, profile.input_modalities)
        self.assertIn(ModelStructure.MOE, profile.structures)
        self.assertIn(Acceleration.MTP, profile.accelerations)
        self.assertNotIn("mtp", {task.value for task in profile.tasks})

    def test_model_descriptor_records_artifacts_and_detection_provenance(self):
        descriptor = ModelDescriptor(
            model_dir="/models/example",
            model_type="example_vlm",
            architecture="ExampleForConditionalGeneration",
            tokenizer_family="example",
            components=(
                ArtifactComponent(
                    name="vision weights",
                    kind=ComponentKind.VISION_ENCODER,
                    path="vision.safetensors",
                ),
                ArtifactComponent(
                    name="mtp drafter",
                    kind=ComponentKind.DRAFT_WEIGHTS,
                    path="mtp.safetensors",
                    required=False,
                ),
            ),
            evidence=(
                DetectionEvidence(
                    field="architecture",
                    value="ExampleForConditionalGeneration",
                    source="config.json:architectures[0]",
                ),
            ),
        )
        self.assertIs(descriptor.components[0].kind, ComponentKind.VISION_ENCODER)
        self.assertFalse(descriptor.components[1].required)
        self.assertTrue(descriptor.evidence[0].source.startswith("config.json"))

    def test_detection_evidence_rejects_invalid_confidence(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            DetectionEvidence("task", "generation", "heuristic", confidence=1.5)
