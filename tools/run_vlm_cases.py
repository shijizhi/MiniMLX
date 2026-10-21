"""Run the checked-in deterministic VLM image regression cases."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Benchmarks" / "vlm_cases.json",
    )
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    cases = json.loads(args.cases.read_text())
    if not isinstance(cases, list):
        parser.error("case manifest must be an array")
    root = args.cases.parent
    runtime = Runtime()
    runtime.load(args.model_dir)
    failures = []
    for case in cases:
        images = [
            (root / image).resolve()
            for image in case.get("images", [])
        ]
        if not images or not all(image.is_file() for image in images):
            failures.append({"case": case.get("name"), "error": "missing image fixture"})
            continue
        parts = [ContentPart(Modality.IMAGE, uri=image.as_uri()) for image in images]
        parts.append(ContentPart(Modality.TEXT, text=case["prompt"]))
        for repeat in range(args.repeat):
            result = runtime.infer(
                InferenceRequest(
                    task=ModelTask.GENERATION,
                    messages=(InputMessage("user", tuple(parts)),),
                    options={"max_tokens": int(case.get("max_tokens", 16))},
                    session_id=f"vlm-case-{case['name']}" if args.repeat > 1 else None,
                )
            )
            text = result.payload.text
            missing = [
                expected
                for expected in case.get("expect", [])
                if expected.casefold() not in text.casefold()
            ]
            record = {
                "case": case["name"],
                "repeat": repeat + 1,
                "text": text,
                "metrics": result.metrics,
                "passed": not missing,
            }
            print(json.dumps(record, ensure_ascii=False))
            if missing:
                record["missing"] = missing
                failures.append(record)
    if failures:
        print(json.dumps({"failed": failures}, ensure_ascii=False), file=sys.stderr)
        return 1
    print("vlm cases: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
