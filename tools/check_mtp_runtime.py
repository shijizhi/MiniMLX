#!/usr/bin/env python3
"""Fast, non-loading probe for the optional Qwen3.5 MTP runtime stack."""

from __future__ import annotations

import inspect
import importlib.metadata
import json
import sys
from pathlib import Path


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    print("MiniMLX Qwen3.5 MTP runtime probe")
    for package in ("mlx", "mlx-lm", "mlx-vlm", "transformers"):
        print(f"package.{package}={package_version(package)}")

    try:
        import mlx.core as mx

        print(f"metal_available={mx.metal.is_available()}")
        from mlx_vlm.speculative.drafters import load_drafter
        from mlx_vlm.speculative.mtp import _buffer_mtp_target_cache, _mtp_rounds
        from mlx_vlm.models.qwen3_5.language import LanguageModel
        from mlx_vlm.utils import get_model_and_args, load_config
    except Exception as exc:
        print(f"probe=FAIL error={type(exc).__name__}: {exc}")
        return 2

    print("mtp_drafter_loader=available")
    print(f"load_drafter_signature={inspect.signature(load_drafter)}")
    print(f"mtp_rounds_signature={inspect.signature(_mtp_rounds)}")
    target_methods = (
        "speculative_verify_hidden",
        "speculative_verify_logits",
        "speculative_logits_from_hidden",
        "rollback_speculative_cache",
        "make_cache",
    )
    available_methods = [
        name for name in target_methods if callable(getattr(LanguageModel, name, None))
    ]
    print(f"target_methods={','.join(available_methods)}")
    print(f"target_speculation_cache={hasattr(LanguageModel, 'make_cache')}")

    if len(sys.argv) > 1:
        model_dir = Path(sys.argv[1]).expanduser()
        config = load_config(model_dir)
        model_class, model_type = get_model_and_args(
            config=config, model_path=model_dir
        )
        print(f"target_model_dir={model_dir}")
        print(f"target_model_type={model_type}")
        print(f"target_model_class={model_class.__name__}")
        print(f"target_architectures={json.dumps(config.get('architectures', []))}")

    print(f"buffer_helper={callable(_buffer_mtp_target_cache)}")
    print("probe=PASS (no weights loaded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
