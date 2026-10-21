from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .api import Worker, serve_unix
from .runtime import Runtime
from .types import Acceleration, ContentPart, InferenceRequest, InputMessage, ModelTask, Modality


def _runtime(_args) -> Runtime:
    return Runtime()


def _inspection_request(args) -> InferenceRequest:
    task = ModelTask(args.task)
    modalities = tuple(Modality(value) for value in (args.input_modality or ["text"]))
    parts = []
    for modality in modalities:
        if modality is Modality.TEXT:
            parts.append(ContentPart(modality, text=""))
        elif modality is Modality.STRUCTURED:
            parts.append(ContentPart(modality, data={}))
        else:
            parts.append(ContentPart(modality, uri=f"file:///inspect.{modality.value}"))
    return InferenceRequest(
        task=task,
        messages=(InputMessage("user", tuple(parts)),),
        accelerations=frozenset(Acceleration(value) for value in (args.acceleration or ())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="minimlx-runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("model_dir")
    generate.add_argument("prompt")
    generate.add_argument("--max-tokens", type=int, default=16)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("model_dir")
    benchmark.add_argument("--prompt", default="Hello")
    benchmark.add_argument("--max-tokens", type=int, default=16)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--repeats", type=int, default=5)
    worker = subparsers.add_parser("worker")
    worker.add_argument("model_dir")
    worker.add_argument("--socket", help="Unix domain socket path")
    inspect = subparsers.add_parser(
        "inspect", help="report model support and execution plan without loading weights"
    )
    inspect.add_argument("model_dir")
    inspect.add_argument(
        "--task", choices=[task.value for task in ModelTask], default=ModelTask.GENERATION.value
    )
    inspect.add_argument(
        "--input-modality",
        choices=[modality.value for modality in Modality],
        action="append",
        help="input modality (repeat to describe a mixed request; defaults to text)",
    )
    inspect.add_argument(
        "--acceleration",
        choices=["mtp", "dflash", "speculative", "quantized_kv"],
        action="append",
        help="require an acceleration explicitly (repeatable)",
    )
    api = subparsers.add_parser("api", help="run an OpenAI-compatible local HTTP API")
    api.add_argument("model_dir")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--max-tokens", type=int, default=32768)
    api.add_argument("--temperature", type=float, default=1.0)
    api.add_argument("--top-p", type=float, default=0.95)
    api.add_argument("--top-k", type=int, default=0)
    api.add_argument("--min-p", type=float, default=0.0)
    api.add_argument("--repetition-penalty", type=float, default=1.0)
    api.add_argument("--presence-penalty", type=float, default=0.0)
    api.add_argument("--frequency-penalty", type=float, default=0.0)
    api.add_argument("--max-context-tokens", type=int, default=32768)
    api.add_argument("--model-alias")
    api.add_argument("--force-sampling", action="store_true")
    api.add_argument("--chat-template-kwargs", type=json.loads, default=None)
    api.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    api.add_argument("--record-history", action=argparse.BooleanOptionalAction, default=True)
    api.add_argument("--history-db", default="~/.minimlx/usage.sqlite3")
    args = parser.parse_args()
    if args.command == "api" and args.chat_template_kwargs is not None:
        if not isinstance(args.chat_template_kwargs, dict):
            parser.error("--chat-template-kwargs must decode to a JSON object")
    if args.command == "inspect":
        print(
            json.dumps(
                Runtime.inspect_model(args.model_dir, _inspection_request(args)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    runtime = _runtime(args)
    runtime.load(args.model_dir)
    if args.command == "api":
        from .api.http import serve_http

        serve_http(
            runtime,
            args.host,
            args.port,
            args.max_tokens,
            api_key=os.environ.get("MINIMLX_API_KEY"),
            model_alias=args.model_alias,
            force_sampling=args.force_sampling,
            default_options={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "repetition_penalty": args.repetition_penalty,
                "presence_penalty": args.presence_penalty,
                "frequency_penalty": args.frequency_penalty,
                "max_context_tokens": args.max_context_tokens,
                **({"chat_template_kwargs": args.chat_template_kwargs} if args.chat_template_kwargs else {}),
            },
            log_level=args.log_level,
            history_db_path=(str(Path(args.history_db).expanduser()) if args.record_history else None),
        )
    elif args.command == "generate":
        result = runtime.generate(args.prompt, max_tokens=args.max_tokens)
        print(json.dumps({
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "prefill_ms": result.prefill_ms,
            "decode_ms": result.decode_ms,
            "decode_tokens_per_second": result.decode_tokens_per_second,
        }, ensure_ascii=False, indent=2))
    elif args.command == "benchmark":
        for _ in range(args.warmup):
            runtime.generate(args.prompt, max_tokens=args.max_tokens)
        samples = []
        prefill_samples = []
        for _ in range(args.repeats):
            result = runtime.generate(args.prompt, max_tokens=args.max_tokens)
            samples.append(result.decode_ms)
            prefill_samples.append(result.prefill_ms)
        samples.sort()
        prefill_samples.sort()
        decode_ms = samples[len(samples) // 2]
        prefill_ms = prefill_samples[len(prefill_samples) // 2]
        print(
            "MiniMLX runtime benchmark "
            f"(warmup={args.warmup}, repeats={args.repeats}, statistic=median)"
        )
        print(
            f"prefill={prefill_ms:.1f} ms, "
            f"decode={decode_ms:.1f} ms / {args.max_tokens} tokens, "
            f"decode_tps={args.max_tokens * 1000.0 / decode_ms:.2f} tok/s"
        )
    else:
        instance = Worker(runtime)
        if args.socket:
            serve_unix(instance, args.socket)
        else:
            instance.run_stdio()


if __name__ == "__main__":
    main()
