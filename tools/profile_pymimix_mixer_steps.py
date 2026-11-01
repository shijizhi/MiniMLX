"""Compare one cached decoder layer's mixer boundary with MiniMLX."""

from pathlib import Path
import os
import sys
import time

import mlx.core as mx


PYMINIMLX_ROOT = Path(os.environ.get("PYMINIMLX_ROOT", "../pyminimlx")).expanduser()
sys.path.insert(0, str(PYMINIMLX_ROOT / "src"))

from pyminimlx.models import Qwen35Model  # noqa: E402


def main() -> None:
    if len(sys.argv) not in {2, 3, 4}:
        raise SystemExit(
            "usage: profile_pymimix_mixer_steps.py <model-directory> [layer] [steps]"
        )

    model = Qwen35Model.from_directory(Path(sys.argv[1]))
    layer_index = int(sys.argv[2]) if len(sys.argv) > 2 else 17
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    logits, state = model.prefill([9419])
    mx.eval(logits)
    next_token = int(mx.argmax(logits[-1]).item())
    timings = []

    for step in range(steps):
        hidden = model.embedding[mx.array([next_token], dtype=mx.int32)]
        for index, layer in enumerate(model.layers):
            if index != layer_index:
                hidden = layer(hidden, state)
                mx.eval(hidden)
                continue

            residual = hidden
            normalized = layer.input_layernorm(hidden)
            mx.eval(normalized)
            start = time.perf_counter()
            if layer.block_type == "full_attention":
                layer_state = state.layers[index]
                if layer_state.kv is None:
                    from pyminimlx.cache import KVCache

                    layer_state.kv = KVCache(1)
                mixed = layer.mixer(
                    normalized, layer_state.kv, 0, state.sequence_length
                )
            else:
                layer_state = state.layers[index]
                mixer_state = {
                    "recurrent": layer_state.recurrent,
                    "conv": layer_state.conv,
                }
                mixed, mixer_state = layer.mixer(normalized, mixer_state)
                layer_state.recurrent = mixer_state["recurrent"]
                layer_state.conv = mixer_state["conv"]
            mx.eval(mixed)
            timings.append((step, (time.perf_counter() - start) * 1000.0))
            post_normalized = layer.post_attention_layernorm(residual + mixed)
            hidden = residual + mixed + layer.mlp(post_normalized)
            mx.eval(hidden)

        state.sequence_length += 1
        logits = model.lm_head(model.final_norm(hidden))
        mx.eval(logits)
        next_token = int(mx.argmax(logits[-1]).item())

    print(
        f"PyMiniMLX L{layer_index} mixer-only cached decode profile "
        f"(steps={steps}, sync after mixer)"
    )
    for step, milliseconds in timings:
        print(f"  step_{step}={milliseconds:.3f} ms")


if __name__ == "__main__":
    main()
