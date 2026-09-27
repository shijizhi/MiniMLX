import json

from minimlx_runtime.registry import describe_model
from minimlx_runtime.types import ExecutionPlan, GenerationResult


def test_descriptor_reads_qwen35_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 1024,
    }))
    descriptor = describe_model(tmp_path)
    assert descriptor.model_type == "qwen3_5"
    assert "hybrid_attention" in descriptor.capabilities


def test_descriptor_reads_nested_text_config_for_qwen35_multimodal(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "dtype": "bfloat16",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
        },
    }))

    descriptor = describe_model(tmp_path)

    assert descriptor.model_type == "qwen3_5"
    assert descriptor.metadata["dtype"] == "bfloat16"
    assert descriptor.metadata["hidden_size"] == 2560
    assert descriptor.metadata["num_hidden_layers"] == 32


def test_descriptor_exposes_qwen35_moe_capabilities(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
        },
    }))

    descriptor = describe_model(tmp_path)

    assert "moe" in descriptor.capabilities
    assert "hybrid_attention" in descriptor.capabilities
    assert descriptor.metadata["num_experts"] == 128
    assert descriptor.metadata["num_experts_per_tok"] == 8
    assert descriptor.metadata["moe_intermediate_size"] == 512


def test_execution_plan_defaults_to_single_request_decode():
    plan = ExecutionPlan(phase="decode")
    assert plan.batch_size == 1
    assert plan.use_continuous_batching is False


def test_generation_result_calculates_throughput():
    result = GenerationResult([1], [2, 3], "ok", 1.0, 20.0)
    assert result.decode_tokens_per_second == 100.0
