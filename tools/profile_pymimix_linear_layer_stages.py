"""Exact production-path decomposition for one cached linear decoder layer."""

from pathlib import Path
import sys
import time

import mlx.core as mx

PYMINIMLX_ROOT = Path("/Users/jizhishi/mycode/ai/pyminimlx")
sys.path.insert(0, str(PYMINIMLX_ROOT / "src"))
from pyminimlx.models import Qwen35Model  # noqa: E402

NAMES = ("input_norm", "mixer", "residual_add", "post_attention_norm", "mlp", "residual_add_mlp")


def one_run(model, layer_index, steps):
    logits, state = model.prefill([9419])
    mx.eval(logits)
    next_token = int(mx.argmax(logits[-1]).item())
    totals = dict.fromkeys(NAMES, 0.0)
    for _ in range(steps):
        hidden = model.embedding[mx.array([next_token], dtype=mx.int32)]
        for index in range(layer_index):
            hidden = model.layers[index](hidden, state)
            mx.eval(hidden)
        layer = model.layers[layer_index]
        residual_input = hidden

        start = time.perf_counter()
        normalized = layer.input_layernorm(residual_input)
        mx.eval(normalized)
        totals["input_norm"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        layer_state = state.layers[layer_index]
        mixer_state = {"recurrent": layer_state.recurrent, "conv": layer_state.conv}
        mixed, mixer_state = layer.mixer(normalized, mixer_state)
        layer_state.recurrent = mixer_state["recurrent"]
        layer_state.conv = mixer_state["conv"]
        mx.eval(mixed)
        totals["mixer"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        residual = residual_input + mixed
        mx.eval(residual)
        totals["residual_add"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        post_normalized = layer.post_attention_layernorm(residual)
        mx.eval(post_normalized)
        totals["post_attention_norm"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        mlp_output = layer.mlp(post_normalized)
        mx.eval(mlp_output)
        totals["mlp"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        hidden = residual + mlp_output
        mx.eval(hidden)
        totals["residual_add_mlp"] += (time.perf_counter() - start) * 1000

        for index in range(layer_index + 1, len(model.layers)):
            hidden = model.layers[index](hidden, state)
            mx.eval(hidden)
        state.sequence_length += 1
        logits = model.lm_head(model.final_norm(hidden))
        mx.eval(logits)
        next_token = int(mx.argmax(logits[-1]).item())
    return totals


def main():
    if len(sys.argv) not in {2, 3, 4, 5, 6}:
        raise SystemExit("usage: profile_pymimix_linear_layer_stages.py <model> [layer] [steps] [warmup] [repeats]")
    model = Qwen35Model.from_directory(Path(sys.argv[1]))
    layer = int(sys.argv[2]) if len(sys.argv) >= 3 else 17
    steps = int(sys.argv[3]) if len(sys.argv) >= 4 else 16
    warmup = int(sys.argv[4]) if len(sys.argv) >= 5 else 3
    repeats = int(sys.argv[5]) if len(sys.argv) >= 6 else 5
    if model.layers[layer].block_type != "linear_attention":
        raise SystemExit(f"layer {layer} is not linear_attention")
    for _ in range(warmup):
        one_run(model, layer, steps)
    samples = {name: [] for name in NAMES}
    for _ in range(repeats):
        values = one_run(model, layer, steps)
        for name in NAMES:
            samples[name].append(values[name])
    print(f"PyMiniMLX exact linear-layer stage profile (L{layer}, steps={steps}, warmup={warmup}, repeats={repeats}, median)")
    for name in NAMES:
        ordered = sorted(samples[name])
        print(f"  L{layer} {name}={ordered[len(ordered) // 2]:.3f} ms")


if __name__ == "__main__":
    main()
