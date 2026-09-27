from __future__ import annotations

import json
import sys
from typing import Any

from ..runtime import Runtime


class Worker:
    """JSON-lines worker boundary; transport-agnostic for now.

    The same handler can later be attached to a Unix domain socket without
    changing model or scheduling code.
    """

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "load":
            self.runtime.load(request["model_dir"])
            return {"ok": True, "command": "load"}
        if command == "generate":
            result = self.runtime.generate(
                request["prompt"],
                max_tokens=int(request.get("max_tokens", 16)),
                context_tokens=(
                    int(request["context_tokens"])
                    if request.get("context_tokens") is not None
                    else None
                ),
                prefill_step_size=int(request.get("prefill_step_size", 2048)),
                session_id=request.get("session_id"),
            )
            return {
                "ok": True,
                "command": "generate",
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "text": result.text,
                "prefill_ms": result.prefill_ms,
                "decode_ms": result.decode_ms,
                "decode_tokens_per_second": result.decode_tokens_per_second,
            }
        if command == "clear_session":
            self.runtime.clear_session(str(request["session_id"]))
            return {"ok": True, "command": "clear_session"}
        if command == "profile_prefill":
            if self.runtime.adapter is None:
                raise RuntimeError("load() must be called before profile_prefill")
            return {
                "ok": True,
                "command": "profile_prefill",
                **self.runtime.adapter.profile_prefill(
                    request["prompt"],
                    context_tokens=int(request["context_tokens"]),
                    warmup=int(request.get("warmup", 1)),
                    prefill_step_size=int(request.get("prefill_step_size", 1024)),
                ),
            }
        if command == "profile_projection_fusion":
            if self.runtime.adapter is None:
                raise RuntimeError("load() must be called before profile_projection_fusion")
            return {
                "ok": True,
                "command": "profile_projection_fusion",
                **self.runtime.adapter.profile_projection_fusion(
                    request["prompt"],
                    context_tokens=int(request["context_tokens"]),
                    warmup=int(request.get("warmup", 1)),
                    repeats=int(request.get("repeats", 5)),
                ),
            }
        raise ValueError(f"unknown command: {command!r}")

    def run_stdio(self) -> None:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                response = self.handle(json.loads(line))
            except Exception as exc:  # protocol boundary must remain alive
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
