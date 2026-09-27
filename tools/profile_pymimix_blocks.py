"""Profile PyMiniMLX with the same six-layer boundaries as Swift MiniMLX.

This intentionally mirrors MiniMLXQwen35Model.profileDecodeBlocks:
prefill once, decode 16 cached tokens, synchronize after layers 0..5,
6..11, 12..17, and 18..23, then synchronize the final norm + LM head.
"""

from pathlib import Path
import sys
import time

import mlx.core as mx


PYMINIMLX_ROOT = Path("/Users/jizhishi/mycode/ai/pyminimlx")
sys.path.insert(0, str(PYMINIMLX_ROOT / "src"))

from pyminimlx.models import Qwen35Model  # noqa: E402


def milliseconds(duration: float) -> float:
    return duration * 1000.0


def profile(model: Qwen35Model, token_id: int, steps: int) -> list[float]:
    token_ids = [token_id]
    logits, state = model.prefill(token_ids)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[-1]).item())

    boundaries = list(range(6, len(model.layers) + 1, 6))
    totals = [0.0 for _ in boundaries]
    final_total = 0.0

    for _ in range(steps):
        hidden = model.embedding[mx.array([next_token], dtype=mx.int32)]
        mx.eval(hidden)
        start = 0

        for block, end in enumerate(boundaries):
            timer = time.perf_counter()
            while start < end:
                hidden = model.layers[start](hidden, state)
                start += 1
            mx.eval(hidden)
            totals[block] += milliseconds(time.perf_counter() - timer)

        # Match Qwen35Model._run_layers: the layer state sees the current
        # sequence length, then the completed token advances it by one.
        state.sequence_length += 1

        timer = time.perf_counter()
        logits = model.lm_head(model.final_norm(hidden))
        mx.eval(logits)
        final_total += milliseconds(time.perf_counter() - timer)
        next_token = int(mx.argmax(logits[-1]).item())

    return totals + [final_total]


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit(
            "usage: profile_pymimix_blocks.py <model-directory> [steps]"
        )

    model_dir = Path(sys.argv[1])
    steps = int(sys.argv[2]) if len(sys.argv) == 3 else 16
    model = Qwen35Model.from_directory(model_dir)
    values = profile(model, token_id=9419, steps=steps)

    print(
        "PyMiniMLX decode block profile "
        f"(cached decode, {steps} tokens, sync per 6 layers)"
    )
    for index, value in enumerate(values[:-1]):
        start = index * 6
        print(f"  layers_{start}..{start + 5}={value:.3f} ms")
    print(f"  final_norm_lm_head={values[-1]:.3f} ms")


if __name__ == "__main__":
    main()
