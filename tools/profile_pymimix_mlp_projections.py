"""Profile PyMiniMLX's L0 SwiGLU projections at Swift's boundaries."""

from pathlib import Path
import os
import sys
import time
from statistics import median

import mlx.core as mx


PYMINIMLX_ROOT = Path(os.environ.get("PYMINIMLX_ROOT", "../pyminimlx")).expanduser()
sys.path.insert(0, str(PYMINIMLX_ROOT / "src"))

from pyminimlx.models import Qwen35Model  # noqa: E402


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit(
            "usage: profile_pymimix_mlp_projections.py <model-directory> [layer]"
        )

    model = Qwen35Model.from_directory(Path(sys.argv[1]))
    index = int(sys.argv[2]) if len(sys.argv) == 3 else 0
    layer = model.layers[index]
    if layer.block_type != "linear_attention":
        raise SystemExit(f"layer {index} is {layer.block_type}, expected linear_attention")

    embedding_input = model.embedding[mx.array([9419], dtype=mx.int32)]
    mx.eval(embedding_input)
    from pyminimlx.runtime import ModelState

    state = ModelState(len(model.layers))
    normalized = layer.input_layernorm(embedding_input)
    layer_state = state.layers[index]
    mixer_state = {"recurrent": layer_state.recurrent, "conv": layer_state.conv}
    mixed, mixer_state = layer.mixer(normalized, mixer_state)
    layer_state.recurrent = mixer_state["recurrent"]
    layer_state.conv = mixer_state["conv"]
    input_value = layer.post_attention_layernorm(embedding_input + mixed)
    mx.eval(input_value)
    mlp = layer.mlp

    def describe(name, value):
        print(f"  L{index} MLP {name}: shape={value.shape} dtype={value.dtype}")

    describe("input", input_value)
    describe("gate_weight", mlp.gate_proj.quantized_weight)
    describe("gate_scales", mlp.gate_proj.quantized_scales)
    describe("gate_biases", mlp.gate_proj.quantized_biases)
    describe("up_weight", mlp.up_proj.quantized_weight)
    describe("up_scales", mlp.up_proj.quantized_scales)
    describe("up_biases", mlp.up_proj.quantized_biases)

    def run_once() -> None:
        value = mlp(input_value)
        mx.eval(value)

    for _ in range(5):
        run_once()

    samples = [[] for _ in range(8)]
    for _ in range(10):
        start = time.perf_counter()
        full = mlp(input_value)
        mx.eval(full)
        samples[0].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        gate = mlp.gate_proj(input_value)
        mx.eval(gate)
        samples[1].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        up = mlp.up_proj(input_value)
        mx.eval(up)
        samples[2].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        direct_gate = mx.quantized_matmul(
            input_value,
            mlp.gate_proj.quantized_weight,
            scales=mlp.gate_proj.quantized_scales,
            biases=mlp.gate_proj.quantized_biases,
            transpose=True,
            group_size=mlp.gate_proj.quantized_group_size,
            bits=mlp.gate_proj.quantized_bits,
            mode=mlp.gate_proj.quantized_mode,
        )
        mx.eval(direct_gate)
        samples[3].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        direct_up = mx.quantized_matmul(
            input_value,
            mlp.up_proj.quantized_weight,
            scales=mlp.up_proj.quantized_scales,
            biases=mlp.up_proj.quantized_biases,
            transpose=True,
            group_size=mlp.up_proj.quantized_group_size,
            bits=mlp.up_proj.quantized_bits,
            mode=mlp.up_proj.quantized_mode,
        )
        mx.eval(direct_up)
        samples[4].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        activated = gate * mx.sigmoid(gate) * up
        mx.eval(activated)
        samples[5].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        down = mlp.down_proj(activated)
        mx.eval(down)
        samples[6].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        direct_down = mx.quantized_matmul(
            activated,
            mlp.down_proj.quantized_weight,
            scales=mlp.down_proj.quantized_scales,
            biases=mlp.down_proj.quantized_biases,
            transpose=True,
            group_size=mlp.down_proj.quantized_group_size,
            bits=mlp.down_proj.quantized_bits,
            mode=mlp.down_proj.quantized_mode,
        )
        mx.eval(direct_down)
        samples[7].append((time.perf_counter() - start) * 1000.0)

    names = [
        "mlp_full", "mlp_gate", "mlp_up", "mlp_gate_direct_quantizedMM",
        "mlp_up_direct_quantizedMM", "mlp_activation",
        "mlp_down_wrapper", "mlp_down_direct_quantizedMM",
    ]
    print(
        f"PyMiniMLX L{index} MLP projection profile "
        "(postNormalized input, median=10)"
    )
    for name, values in zip(names, samples):
        print(f"  L{index} {name}={median(values):.3f} ms")

    # One fresh MLP graph, evaluated through cumulative checkpoints.  Use
    # adjacent differences between these values for segment attribution.
    gate = mlp.gate_proj(input_value)
    up = mlp.up_proj(input_value)
    activated = gate * mx.sigmoid(gate) * up
    down = mlp.down_proj(activated)
    cumulative = [
        ("gate_projection", gate),
        ("up_projection", up),
        ("mlp_activation", activated),
        ("mlp_down", down),
    ]
    start = time.perf_counter()
    print(f"PyMiniMLX L{index} MLP cumulative profile (one graph, sequential eval)")
    for name, value in cumulative:
        mx.eval(value)
        print(f"  L{index} cumulative_{name}={(time.perf_counter() - start) * 1000.0:.3f} ms")


if __name__ == "__main__":
    main()
