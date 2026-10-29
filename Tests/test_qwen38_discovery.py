import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from minimlx_runtime.adapters.mlx_lm_standard import StandardTextAdapter
from minimlx_runtime.registry import CapabilityResolver, builtin_provider_registry, describe_model
from minimlx_runtime.types import ModelTask, Modality, ModelStructure
from minimlx_runtime.types import (
    ComponentKind,
    ContentPart,
    InferenceRequest,
    InputMessage,
    RuntimeEnvironment,
    SupportStatus,
)


class Qwen38DiscoveryTests(unittest.TestCase):
    def test_qwen38_flash_next_is_marked_as_offload_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_next",
                        "architectures": ["Qwen3NextForCausalLM"],
                        "num_experts": 64,
                        "num_experts_per_tok": 4,
                    }
                )
            )
            (model_dir / "model-00001.safetensors").write_bytes(b"weights")
            (model_dir / ".minimlx-repo-id").write_text(
                "mlx-community/Qwen3.8-Flash-Next-4bit"
            )

            descriptor = describe_model(model_dir)

            self.assertTrue(descriptor.metadata["is_qwen38_flash_next"])
            self.assertEqual(descriptor.metadata["total_weight_bytes"], 7)
            self.assertIn("ssd_offload_candidate", descriptor.capabilities)
            self.assertIn(ModelStructure.MOE, descriptor.support.structures)
            self.assertEqual(descriptor.support.tasks, {ModelTask.GENERATION})
            self.assertEqual(descriptor.support.input_modalities, {Modality.TEXT})

    def test_standard_adapter_uses_lazy_loader_and_offload_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_next",
                        "architectures": ["Qwen3NextForCausalLM"],
                    }
                )
            )
            header = json.dumps(
                {"layer.weight": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}},
                separators=(",", ":"),
            ).encode()
            (model_dir / "model-00001.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + b"ok"
            )
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.weight": "model-00001.safetensors"}})
            )
            descriptor = describe_model(model_dir)
            seen: dict = {}
            fake_module = types.SimpleNamespace(
                load=lambda path, **kwargs: (
                    seen.update(path=path, kwargs=kwargs) or (object(), object())
                )
            )

            with patch.dict(sys.modules, {"mlx_lm": fake_module}), patch.dict(
                "os.environ",
                {
                    "MINIMLX_QWEN38_OFFLOAD": "1",
                    "MINIMLX_QWEN38_CUSTOM_EXECUTOR": "0",
                },
                clear=False,
            ):
                adapter = StandardTextAdapter().load(descriptor)

            self.assertTrue(seen["kwargs"]["lazy"])
            self.assertIsNotNone(adapter.offload_session)

    def test_qwen4_exp_routes_to_vlm_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen4_exp",
                        "architectures": ["Qwen4ExpForConditionalGeneration"],
                        "text_config": {"num_experts": 512, "num_experts_per_tok": 10},
                        "vision_config": {"depth": 2},
                    }
                )
            )
            for filename in (
                "tokenizer.json",
                "processor_config.json",
                "model.safetensors",
            ):
                (model_dir / filename).write_bytes(b"fixture")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.visual.patch_embed.weight": "model.safetensors",
                            "model.language_model.layers.0.weight": "model.safetensors",
                        }
                    }
                )
            )
            descriptor = describe_model(model_dir)
            request = InferenceRequest(
                ModelTask.GENERATION,
                (InputMessage("user", (ContentPart(Modality.TEXT, text="hello"),)),),
            )
            decision, plan = CapabilityResolver(builtin_provider_registry()).resolve(
                descriptor,
                request,
                RuntimeEnvironment(backends=frozenset({"mlx-vlm"})),
            )

            self.assertTrue(descriptor.metadata["is_qwen38_flash_next"])
            self.assertIn(ComponentKind.VISION_ENCODER, {
                component.kind for component in descriptor.components
            })
            self.assertEqual(decision.status, SupportStatus.SUPPORTED)
            self.assertEqual(plan.family_adapter, "qwen38-vlm")

    def test_deepseek41_flash_is_detected_as_a_distinct_offload_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / ".minimlx-repo-id").write_text(
                "deepseek-ai/DeepSeek-V4.1-Flash-1017"
            )
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v41",
                        "architectures": ["DeepseekV41ForCausalLM"],
                        "text_config": {"num_experts": 384, "num_experts_per_tok": 6},
                        "vision_config": {"depth": 32},
                    }
                )
            )
            for filename in (
                "tokenizer.json",
                "processor_config.json",
                "model.safetensors",
                "model-engram-1-weight.safetensors",
                "model-sparse-indexer.safetensors",
                "model-dspark.safetensors",
            ):
                (model_dir / filename).write_bytes(b"fixture")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "model.safetensors",
                            "model.visual.patch_embed.weight": "model.safetensors",
                            "model.engram.table": "model-engram-1-weight.safetensors",
                            "model.attention.indexer": "model-sparse-indexer.safetensors",
                            "model.dspark.draft": "model-dspark.safetensors",
                        }
                    }
                )
            )

            descriptor = describe_model(model_dir)

            self.assertTrue(descriptor.metadata["is_deepseek41_flash"])
            self.assertIn("engram_memory", descriptor.capabilities)
            self.assertIn("dspark_mtp", descriptor.capabilities)
            self.assertEqual(descriptor.metadata["num_experts"], 384)
            self.assertTrue(descriptor.metadata["has_engram"])
            self.assertTrue(descriptor.metadata["has_sparse_indexer"])
            self.assertTrue(descriptor.metadata["has_dspark"])
            request = InferenceRequest(
                ModelTask.GENERATION,
                (InputMessage("user", (ContentPart(Modality.TEXT, text="hello"),)),),
            )
            decision, plan = CapabilityResolver(builtin_provider_registry()).resolve(
                descriptor,
                request,
                RuntimeEnvironment(backends=frozenset({"mlx-vlm"})),
            )
            self.assertEqual(decision.status, SupportStatus.SUPPORTED)
            self.assertEqual(plan.backend_adapter, "deepseek41-mlx-vlm-runtime")


if __name__ == "__main__":
    unittest.main()
