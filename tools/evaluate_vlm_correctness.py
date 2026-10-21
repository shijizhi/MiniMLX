"""Deterministic correctness probes for MiniMLX vision-language checkpoints.

The image fixtures are generated in memory, so the gold answers are explicit
and the test does not depend on downloaded or user-provided images.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from minimlx_runtime.runtime import Runtime
from minimlx_runtime.types import (
    ContentPart,
    InferenceRequest,
    InputMessage,
    ModelTask,
    Modality,
)


FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _image_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _ocr_image() -> Image.Image:
    image = Image.new("RGB", (1400, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 95), "ORCHID 2741", fill="black", font=_font(150))
    draw.text((75, 285), "NORTH GATE", fill="#333333", font=_font(70))
    return image


def _count_image() -> Image.Image:
    image = Image.new("RGB", (900, 620), "white")
    draw = ImageDraw.Draw(image)
    for x, y in ((180, 170), (450, 170), (720, 170)):
        draw.ellipse((x - 65, y - 65, x + 65, y + 65), fill="#e53935")
    for x, y in ((300, 430), (600, 430)):
        draw.rectangle((x - 65, y - 65, x + 65, y + 65), fill="#1769d2")
    return image


def _position_image() -> Image.Image:
    image = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.polygon(((160, 310), (250, 100), (340, 310)), fill="#22a447")
    draw.ellipse((420, 115, 600, 295), fill="#e53935")
    draw.rectangle((660, 115, 840, 295), fill="#1769d2")
    return image


def _circles_image(count: int, color: str) -> Image.Image:
    image = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(image)
    if count == 1:
        positions = ((450, 210),)
    else:
        positions = tuple((100 + index * 160, 210) for index in range(count))
    for x, y in positions:
        draw.ellipse((x - 48, y - 48, x + 48, y + 48), fill=color)
    return image


def _data_image(image: Image.Image) -> ContentPart:
    return ContentPart(
        Modality.IMAGE,
        data_base64=_image_b64(image),
        mime_type="image/png",
    )


def _data_uri_image(image: Image.Image) -> ContentPart:
    return ContentPart(
        Modality.IMAGE,
        uri=f"data:image/png;base64,{_image_b64(image)}",
        mime_type="image/png",
    )


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _request(messages, *, max_tokens: int = 48, stream: bool = False):
    return InferenceRequest(
        task=ModelTask.GENERATION,
        messages=tuple(messages),
        options={"max_tokens": max_tokens, "temperature": 0.0},
        stream=stream,
    )


def _cases():
    ocr = _ocr_image()
    return (
        {
            "name": "text_only_control",
            "messages": [InputMessage("user", (ContentPart(Modality.TEXT, text="Reply with exactly CONTROL_OK."),))],
            "needles": ("control ok",),
        },
        {
            "name": "two_line_ocr",
            "messages": [InputMessage("user", (_data_image(ocr), ContentPart(Modality.TEXT, text="Transcribe every line of text in reading order. Do not add anything.")))],
            "needles": ("orchid", "2741", "north gate"),
        },
        {
            "name": "two_line_ocr_data_uri",
            "messages": [InputMessage("user", (_data_uri_image(ocr), ContentPart(Modality.TEXT, text="Transcribe every line of text in reading order. Do not add anything.")))],
            "needles": ("orchid", "2741", "north gate"),
        },
        {
            "name": "count_red_circles",
            "messages": [InputMessage("user", (_data_image(_count_image()), ContentPart(Modality.TEXT, text="How many red circles are shown? Answer with the number only.")))],
            "needles": ("3", "three"),
            "any_needles": True,
        },
        {
            "name": "negative_object_presence",
            "messages": [InputMessage("user", (_data_image(_count_image()), ContentPart(Modality.TEXT, text="Is there a yellow star anywhere in this image? Answer yes or no.")))],
            "exact_normalized": "no",
        },
        {
            "name": "center_object_and_color",
            "messages": [InputMessage("user", (_data_image(_position_image()), ContentPart(Modality.TEXT, text="What shape is in the center, and what color is it? Answer briefly.")))],
            "needles": ("circle", "red"),
        },
        {
            "name": "ordered_two_image_comparison",
            "messages": [InputMessage("user", (_data_image(_circles_image(2, "#e53935")), _data_image(_circles_image(5, "#1769d2")), ContentPart(Modality.TEXT, text="Which image contains more circles, the first or the second? Answer first or second.")))],
            "needles": ("second",),
        },
        {
            "name": "multi_turn_image_followup",
            "messages": [
                InputMessage("user", (ContentPart(Modality.TEXT, text="For the next image, identify the exact large word."),)),
                InputMessage("assistant", (ContentPart(Modality.TEXT, text="Understood."),)),
                InputMessage("user", (_data_image(ocr), ContentPart(Modality.TEXT, text="What is the large word?"))),
            ],
            "needles": ("orchid",),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    runtime = Runtime()
    runtime.load(args.model_dir)
    failures = []
    records = []
    for case in _cases():
        outputs = []
        for repeat in range(args.repeat):
            result = runtime.infer(_request(case["messages"]))
            text = result.payload.text
            normalized = _normalize(text)
            needles = case.get("needles", ())
            found = [needle for needle in needles if needle in normalized]
            if "exact_normalized" in case:
                passed = normalized == case["exact_normalized"]
            else:
                passed = bool(found) if case.get("any_needles") else len(found) == len(case["needles"])
            record = {
                "case": case["name"],
                "repeat": repeat + 1,
                "text": text,
                "passed": passed,
                "metrics": result.metrics,
            }
            print(json.dumps(record, ensure_ascii=False))
            outputs.append(normalized)
            if not passed:
                failures.append({**record, "missing": [item for item in needles if item not in normalized]})
        stable = len(set(outputs)) == 1
        records.append({"case": case["name"], "stable": stable})
        if not stable:
            failures.append({"case": case["name"], "error": "greedy output changed between repeats", "outputs": outputs})

    image_case = next(case for case in _cases() if case["name"] == "two_line_ocr")
    streamed = "".join(
        event.get("delta", "")
        for event in runtime.infer_stream(_request(image_case["messages"], stream=True))
        if event["event"] == "delta"
    )
    stream_normalized = _normalize(streamed)
    stream_passed = all(item in stream_normalized for item in image_case["needles"])
    print(json.dumps({"case": "two_line_ocr_stream", "text": streamed, "passed": stream_passed}, ensure_ascii=False))
    if not stream_passed:
        failures.append({"case": "two_line_ocr_stream", "text": streamed})

    runtime.unload()
    summary = {"cases": len(records), "repeats_per_case": args.repeat, "stream_passed": stream_passed,
               "stable_cases": sum(record["stable"] for record in records), "failures": len(failures)}
    print(json.dumps(summary, ensure_ascii=False))
    if failures:
        print(json.dumps({"failed": failures}, ensure_ascii=False), file=sys.stderr)
        return 1
    print("VLM correctness: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
