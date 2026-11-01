"""Opt-in experiment: parallel pread of selected mmap-backed expert tensors."""
import argparse
import concurrent.futures
import json
import os
import struct
import time

import numpy as np
import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.models.switch_layers import OffloadedSwitchGLU
from mlx_vlm.prompt_utils import apply_chat_template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    model, processor = load(args.model, lazy=True, expert_cache_gb=12)
    store = model.moe_offload_store
    original = OffloadedSwitchGLU.__call__
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    headers = {}
    stats = {"bytes": 0, "requests": 0, "seconds": 0.0}
    fds = {}

    def read_expert(fd, base, entries):
        payload = []
        for name, metadata in entries:
            start, end = metadata["data_offsets"]
            data = bytearray()
            while len(data) < end - start:
                chunk = os.pread(fd, end - start - len(data), base + start + len(data))
                if not chunk:
                    raise OSError(f"short read for {name}")
                data.extend(chunk)
            payload.append((name, metadata, data))
        return payload

    def read_range(path, offset, length):
        fd = os.open(path, os.O_RDONLY)
        try:
            data = os.pread(fd, length, offset)
            if len(data) != length:
                raise OSError("short expert read")
        finally:
            os.close(fd)

    def invoke(layer, x, indices):
        ids = np.unique(np.asarray(indices))
        # Read only uncached experts. All MLX access and cache mutations stay
        # on the inference thread; workers perform plain POSIX reads only.
        missing = [int(i) for i in ids if (layer.layer_id, int(i)) not in store._lru]
        if args.direct and missing and len(ids) * 2 <= store.num_experts:
            path = store._paths[layer.layer_id]
            if path not in headers:
                with open(path, "rb") as f:
                    size = struct.unpack("<Q", f.read(8))[0]
                    headers[path] = (8 + size, json.loads(f.read(size)))
                fds[path] = os.open(path, os.O_RDONLY)
            base, header = headers[path]
            entries = {j: [(n, m) for n, m in header.items() if n.startswith(f"e{j}.")] for j in missing}
            start = time.monotonic()
            values = list(executor.map(lambda j: read_expert(fds[path], base, entries[j]), missing))
            stats["seconds"] += time.monotonic() - start
            stats["bytes"] += sum(len(data) for value in values for _, _, data in value)
            stats["requests"] += len(missing)
            pending = dict(zip(missing, values))
            previous = store._ensure_loaded
            def install(lid, expert, mapping):
                if lid != layer.layer_id or expert not in pending:
                    return previous(lid, expert, mapping)
                for name, meta, data in pending.pop(expert):
                    dtype = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32, "I32": np.int32, "U8": np.uint8}[meta["dtype"]]
                    array = mx.array(np.frombuffer(data, dtype=dtype).reshape(meta["shape"]))
                    mapping[name] = array.view(mx.bfloat16) if meta["dtype"] == "BF16" else array
            store._ensure_loaded = install
            try:
                return original(layer, x, indices)
            finally:
                store._ensure_loaded = previous
        if args.parallel and missing and len(ids) * 2 <= store.num_experts:
            path = store._paths[layer.layer_id]
            if path not in headers:
                with open(path, "rb") as f:
                    size = struct.unpack("<Q", f.read(8))[0]
                    headers[path] = (8 + size, json.loads(f.read(size)))
            base, header = headers[path]
            regions = []
            for expert in missing:
                for name, metadata in header.items():
                    if name.startswith(f"e{expert}."):
                        start, end = metadata["data_offsets"]
                        regions.append((path, base + start, end - start))
            started = time.monotonic()
            list(executor.map(lambda r: read_range(*r), regions))
            stats["seconds"] += time.monotonic() - started
            stats["bytes"] += sum(r[2] for r in regions)
            stats["requests"] += len(regions)
        return original(layer, x, indices)

    if args.parallel or args.direct:
        OffloadedSwitchGLU.__call__ = invoke
    try:
        prompt = apply_chat_template(processor, model.config, [{"role": "user", "content": "Explain in English, in three short sentences, why the sky looks blue."}], num_images=0, enable_thinking=False)
        result = generate(model, processor, prompt=prompt, max_tokens=64, verbose=False, prefill_step_size=128, temperature=0)
        print(json.dumps({"text": result.text, "tokens": result.token_ids, "prefill_tps": result.prompt_tps, "decode_tps": result.generation_tps, "peak_memory_gb": result.peak_memory, "cache": store.stats(), "pread": stats}), flush=True)
    finally:
        OffloadedSwitchGLU.__call__ = original
        executor.shutdown(wait=True)
        for fd in fds.values():
            os.close(fd)


if __name__ == "__main__":
    main()
