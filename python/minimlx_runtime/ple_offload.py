"""Create a managed, hard-linked view with Qwen4 PLE rows read from SSD."""
from __future__ import annotations

import json
import math
import mmap
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PLE_MARKER = ".ple.ple_embedding.ngram_embedding.shards."
_DTYPES = {"U32": np.dtype("<u4"), "BF16": np.dtype("<u2"), "U8": np.dtype("u1")}
_PLE_LOAD_LOCK = threading.RLock()


def supports_paged_ple(paged: str | Path) -> bool:
    """Match the PLE storage formats supported by the installed mlx-vlm."""
    try:
        paged = Path(paged)
        source = paged.parent
        config = json.loads((source / "config.json").read_text())
        index = json.loads((source / "model.safetensors.index.json").read_text())
        if config.get("model_type") != "qwen4_exp":
            return False
        prefixes = {name.rsplit(".", 1)[0] for name in index["weight_map"] if PLE_MARKER in name}
        if not prefixes:
            return False
        quant = config.get("quantization_config") or config.get("quantization", {})
        layouts = {
            (params.get("bits"), params.get("group_size"), params.get("mode"))
            for prefix in prefixes
            for params in [quant.get(prefix, quant)]
        }
        return layouts in ({(4, 32, "affine")}, {(4, 16, "nvfp4")})
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def init_shared_ple_ranges(store, manifest, source_root):
    """Map each source shard once, instead of opening every tensor separately."""
    maps = {}
    shards = []
    expected_start = 0
    source_root = Path(source_root)
    for entry in manifest["shards"]:
        start, count = int(entry["row_start"]), int(entry["row_count"])
        if start != expected_start or count <= 0:
            raise ValueError("PLE shards must be positive, contiguous, and ordered")
        arrays = {}
        for name, expected_dtype in store._tensor_layout:
            tensor = entry[name]
            if tensor["dtype"] != expected_dtype:
                raise ValueError(f"unexpected {name} dtype: {tensor['dtype']}")
            filename = Path(tensor["file"])
            if filename.is_absolute() or filename.name != str(filename):
                raise ValueError("tensor files must be filenames under source_root")
            path = source_root / filename
            offset = int(tensor["offset"])
            shape = tuple(int(dim) for dim in tensor["shape"])
            if not shape or shape[0] != count or any(dim < 0 for dim in shape):
                raise ValueError(f"invalid {name} tensor shape")
            dtype = _DTYPES[expected_dtype]
            byte_count = math.prod(shape) * dtype.itemsize
            if offset < 0 or path.stat().st_size < offset + byte_count:
                raise ValueError(f"{name} byte range exceeds {path}")
            if path not in maps:
                with path.open("rb") as stream:
                    maps[path] = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            arrays[name] = np.ndarray(shape, dtype=dtype, buffer=maps[path], offset=offset)
        if arrays["weight"].shape[1] * 8 != store.row_width:
            raise ValueError("packed weight width does not match row_width")
        groups = store.row_width // store._group_size
        if any(arrays[name].shape[1:] != (groups,) for name in store._tensor_names if name != "weight"):
            raise ValueError("auxiliary tensor shape does not match Q4 group layout")
        shards.append((start, start + count, arrays))
        expected_start += count
    store._shards = shards
    store._minimlx_shared_mmaps = maps
    store.row_count = expected_start


@contextmanager
def shared_ple_mappings():
    """Scope the loader adaptation to model construction."""
    from mlx_vlm.models.qwen4_exp.ple_storage import QuantizedMMapNGramEmbedding
    with _PLE_LOAD_LOCK:
        original = QuantizedMMapNGramEmbedding._init_safetensors_ranges
        QuantizedMMapNGramEmbedding._init_safetensors_ranges = init_shared_ple_ranges
        try:
            yield
        finally:
            QuantizedMMapNGramEmbedding._init_safetensors_ranges = original


def prepare_paged_ple_view(paged: str | Path) -> Path:
    """Build metadata and hard links only; never rewrite checkpoint payloads."""
    paged = Path(paged).resolve(strict=True)
    source = paged.parent
    view = paged / ".minimlx-ple-view"
    if view.is_symlink():
        raise RuntimeError("PLE view must not be a symbolic link")
    if (view / ".minimlx-ple-ready").is_file():
        return view
    if view.exists():
        raise RuntimeError(f"Incomplete PLE view: {view}")

    config = json.loads((paged / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    if config.get("model_type") != "qwen4_exp" or not any(
        PLE_MARKER in name for name in index.get("weight_map", {})
    ):
        raise ValueError("Checkpoint does not contain supported Qwen4 PLE weights")
    from mlx_vlm.models.qwen4_exp.ple_storage import build_quantized_ple_manifest

    staging = Path(tempfile.mkdtemp(prefix=".minimlx-ple-preparing-", dir=paged))
    try:
        build_quantized_ple_manifest(source, staging / "ple-store.json", cache_rows=8192)
        for item in paged.iterdir():
            if item.is_symlink():
                raise ValueError(f"Unexpected symlink in paged checkpoint: {item}")
            if item.name == "experts":
                (staging / "experts").mkdir()
                for expert in item.iterdir():
                    if expert.is_symlink() or not expert.is_file():
                        raise ValueError(f"Invalid expert file: {expert}")
                    os.link(expert, staging / "experts" / expert.name)
            elif item.is_file() and item.name.startswith("resident-") and item.suffix == ".safetensors":
                os.link(item, staging / item.name)
            elif item.is_file() and item.name not in {"config.json", "model.safetensors.index.json"} and item.suffix != ".safetensors":
                shutil.copy2(item, staging / item.name)

        if not list(staging.glob("resident-*.safetensors")) or not (staging / "offload_index.json").is_file():
            raise ValueError("Incomplete paged model: missing resident weights or expert index")
        config.setdefault("text_config", {})["ple_storage"] = {
            "manifest": "ple-store.json", "cache_rows": 8192,
        }
        for field in ("quantization", "quantization_config"):
            if isinstance(config.get(field), dict):
                config[field] = {key: value for key, value in config[field].items() if PLE_MARKER not in key}
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / ".minimlx-ple-ready").write_text("1\n")
        try:
            os.rename(staging, view)
        except OSError:
            if view.is_symlink() or not (view / ".minimlx-ple-ready").is_file():
                raise
        return view
    finally:
        if staging.exists():
            shutil.rmtree(staging)
