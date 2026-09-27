"""Profile one PyMiniMLX linear-attention decoder layer by Swift's boundaries."""

from pathlib import Path
import sys
import time
from statistics import median

import mlx.core as mx


PYMINIMLX_ROOT = Path("/Users/jizhishi/mycode/ai/pyminimlx")
sys.path.insert(0, str(PYMINIMLX_ROOT / "src"))

from pyminimlx.models import Qwen35Model  # noqa: E402


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit(
            "usage: profile_pymimix_linear_layer_parts.py <model-directory> [layer]"
        )

    model = Qwen35Model.from_directory(Path(sys.argv[1]))
    index = int(sys.argv[2]) if len(sys.argv) == 3 else 0
    layer = model.layers[index]
    if layer.block_type != "linear_attention":
        raise SystemExit(f"layer {index} is {layer.block_type}, expected linear_attention")

    input_value = model.embedding[mx.array([9419], dtype=mx.int32)]
    mx.eval(input_value)

    # Warm using the same public layer path with a real ModelState.
    from pyminimlx.runtime import ModelState

    def warm_once() -> None:
        state = ModelState(len(model.layers))
        value = model.layers[index](input_value, state)
        mx.eval(value)

    for _ in range(5):
        warm_once()

    samples = [[] for _ in range(5)]
    for _ in range(10):
        state = ModelState(len(model.layers))

        start = time.perf_counter()
        normalized = layer.input_layernorm(input_value)
        mx.eval(normalized)
        samples[0].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        layer_state = state.layers[index]
        mixer_state = {
            "recurrent": layer_state.recurrent,
            "conv": layer_state.conv,
        }
        mixed, mixer_state = layer.mixer(normalized, mixer_state)
        layer_state.recurrent = mixer_state["recurrent"]
        layer_state.conv = mixer_state["conv"]
        mx.eval(mixed)
        samples[1].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        residual = input_value + mixed
        post_normalized = layer.post_attention_layernorm(residual)
        mx.eval(post_normalized)
        samples[2].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        mlp_output = layer.mlp(post_normalized)
        mx.eval(mlp_output)
        samples[3].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        output = residual + mlp_output
        mx.eval(output)
        samples[4].append((time.perf_counter() - start) * 1000.0)

    names = ["input_norm", "mixer", "residual_post_norm", "mlp", "residual_add"]
    print(
        f"PyMiniMLX L{index} linear layer parts "
        "(empty state, single token, median=10)"
    )
    for name, values in zip(names, samples):
        print(f"  L{index} {name}={median(values):.3f} ms")


if __name__ == "__main__":
    main()
