"""Prepare a managed mlx-vlm expert store alongside its source checkpoint."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def prepare(source: Path) -> Path:
    source = source.resolve(strict=True)
    destination = source / ".minimlx-ssd"
    if destination.exists():
        raise FileExistsError(f"SSD copy already exists: {destination}")
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config") or config
    qwen35_moe = (config.get("model_type") == "qwen3_5_moe"
                  and text.get("num_experts") == 256
                  and text.get("hidden_size") == 2048
                  and text.get("num_hidden_layers") == 40)
    if config.get("model_type") != "qwen4_exp" and not qwen35_moe:
        raise ValueError("SSD preparation supports Qwen Flash Next and Qwen3.5/3.6 35B-A3B MLX checkpoints")
    from mlx_vlm import moe_offload

    # The upstream classifier groups experts by layer number alone. The MTP
    # head also has a layer 0, which must not overwrite the target's layer 0.
    # Keep the head intact in the resident checkpoint until it has its own
    # paging implementation.
    upstream_plan = moe_offload.plan

    def target_only_plan(names):
        names = list(names)
        mtp_prefixes = ("mtp.", "language_model.mtp.", "model.mtp.")
        mtp = [name for name in names if name.startswith(mtp_prefixes)]
        result = upstream_plan([name for name in names if not name.startswith(mtp_prefixes)])
        result["resident"] = sorted(result["resident"] + mtp)
        return result

    staging = Path(tempfile.mkdtemp(prefix=".minimlx-ssd-preparing-", dir=source))
    print(f"Preparing SSD copy in {staging}", flush=True)
    try:
        moe_offload.plan = target_only_plan
        try:
            moe_offload.repack(str(source), str(staging))
        finally:
            moe_offload.plan = upstream_plan
        index = json.loads((staging / "offload_index.json").read_text())
        if not index.get("layers") or not index.get("num_experts"):
            raise ValueError("Incomplete SSD expert index")
        for layer in index["layers"]:
            if not (staging / "experts" / f"layer_{layer:04d}.safetensors").is_file():
                raise ValueError(f"Missing SSD expert layer {layer}")
        resident = sorted(staging.glob("resident-*.safetensors"))
        if not resident:
            raise ValueError("Missing resident weights")
        (staging / ".minimlx-ssd-manifest.json").write_text(json.dumps({
            "version": 1,
            "resident": [p.name for p in resident],
            "source_index_bytes": (source / "model.safetensors.index.json").stat().st_size,
        }))
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"SSD copy ready: {destination}", flush=True)
    return destination


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
