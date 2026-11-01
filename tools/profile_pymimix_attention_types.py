"""Compare Qwen3.5's two decoder-layer execution types with Swift MiniMLX.

This mirrors profileDecodeLayerTotals rather than grouping consecutive layers:
Qwen3.5 repeats 3 linear-attention layers followed by 1 full-attention layer.
Each layer is evaluated before moving to the next one, and totals are
aggregated only by execution type.
"""

from pathlib import Path
import os
import sys
import time

import mlx.core as mx


PYMINIMLX_ROOT = Path(os.environ.get("PYMINIMLX_ROOT", "../pyminimlx")).expanduser()
sys.path.insert(0, str(PYMINIMLX_ROOT / "src"))

from pyminimlx.models import Qwen35Model  # noqa: E402


def main() -> None:
    if len(sys.argv) not in {2, 3, 4, 5}:
        raise SystemExit(
            "usage: profile_pymimix_attention_types.py <model-directory> [steps] [warmup] [repeats]"
        )

    model = Qwen35Model.from_directory(Path(sys.argv[1]))
    steps = int(sys.argv[2]) if len(sys.argv) >= 3 else 16
    warmup = int(sys.argv[3]) if len(sys.argv) >= 4 else 3
    repeats = int(sys.argv[4]) if len(sys.argv) >= 5 else 5

    def one_run():
        logits, state = model.prefill([9419])
        mx.eval(logits)
        next_token = int(mx.argmax(logits[-1]).item())
        totals = {"linear_attention": 0.0, "full_attention": 0.0}
        for _ in range(steps):
            hidden = model.embedding[mx.array([next_token], dtype=mx.int32)]
            for layer in model.layers:
                start = time.perf_counter()
                hidden = layer(hidden, state)
                mx.eval(hidden)
                totals[layer.block_type] += (time.perf_counter() - start) * 1000.0
            state.sequence_length += 1
            logits = model.lm_head(model.final_norm(hidden))
            mx.eval(logits)
            next_token = int(mx.argmax(logits[-1]).item())
        return totals

    for _ in range(warmup):
        one_run()
    samples = {"linear_attention": [], "full_attention": []}
    for _ in range(repeats):
        totals = one_run()
        for layer_type in samples:
            samples[layer_type].append(totals[layer_type])

    print(
        "PyMiniMLX complete layer-type profile "
        f"(cached decode, steps={steps}, warmup={warmup}, repeats={repeats}, median)"
    )
    for layer_type in ("linear_attention", "full_attention"):
        ordered = sorted(samples[layer_type])
        print(f"  {layer_type}={ordered[len(ordered) // 2]:.3f} ms")


if __name__ == "__main__":
    main()
