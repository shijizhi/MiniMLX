"""Run a real Metal-backed VLM smoke/correctness check.

Example:
  PYTHONPATH=python ./.venv/bin/python tools/verify_vlm_support.py \
    /path/to/Qwen3.5-0.8B-MLX-4bit image.png \
    --prompt "Extract the text from this image." \
    --expect "Swift Transformers"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minimlx_runtime.runtime import Runtime
from minimlx_runtime.types import (
    ContentPart,
    InferenceRequest,
    InputMessage,
    ModelTask,
    Modality,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("image")
    parser.add_argument("--prompt", default="Extract the text from this image.")
    parser.add_argument("--expect", required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        parser.error(f"image does not exist: {image}")

    runtime = Runtime()
    runtime.load(args.model_dir)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    result = None
    for index in range(args.repeat):
        result = runtime.infer(
            InferenceRequest(
                task=ModelTask.GENERATION,
                messages=(
                    InputMessage(
                        "user",
                        (
                            ContentPart(Modality.IMAGE, uri=image.as_uri()),
                            ContentPart(Modality.TEXT, text=args.prompt),
                        ),
                    ),
                ),
                options={"max_tokens": args.max_tokens},
                session_id="vlm-verify" if args.repeat > 1 else None,
            )
        )
        text = result.payload.text
        print(f"[{index + 1}/{args.repeat}] {text}")
        print({"usage": result.usage, "metrics": result.metrics})
        if args.expect.casefold() not in text.casefold():
            print(f"expected substring not found: {args.expect!r}", file=sys.stderr)
            return 1
    print("vlm correctness: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
