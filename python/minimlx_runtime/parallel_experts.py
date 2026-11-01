"""Bounded parallel expert reads for mlx-vlm's paged Qwen4 model."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import struct
import threading

_exact_expert_rows = ContextVar("minimlx_exact_expert_rows", default=False)


@contextmanager
def exact_expert_rows():
    """Keep each routed projection equivalent to one-token decode."""
    token = _exact_expert_rows.set(True)
    try:
        yield
    finally:
        _exact_expert_rows.reset(token)


class ParallelExpertReads:
    def __init__(self, store, workers=12):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="minimlx-expert")
        self.lock = threading.RLock()
        self.headers = {}
        self.fds = {}

    def close(self):
        with self.lock:
            self.pool.shutdown(wait=True)
            for fd in self.fds.values():
                os.close(fd)
            self.fds.clear()

    def _index(self, layer):
        if layer not in self.headers:
            path = self.store._paths[layer]
            with open(path, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                if not 0 < size <= 100 * 1024 * 1024:
                    raise ValueError("invalid expert safetensors header")
                header = json.loads(f.read(size))
            grouped = {}
            for name, meta in header.items():
                if not name.startswith("e") or "." not in name:
                    continue
                prefix = name.split(".", 1)[0][1:]
                if prefix.isdigit():
                    grouped.setdefault(int(prefix), []).append((name, meta))
            self.headers[layer] = (8 + size, grouped)
            self.fds[layer] = os.open(path, os.O_RDONLY)
        return self.headers[layer]

    @staticmethod
    def _read(fd, base, entries):
        result = []
        for name, meta in entries:
            start, end = meta["data_offsets"]
            if not 0 <= start <= end:
                raise ValueError(f"invalid offsets for {name}")
            data = bytearray(end - start)
            offset = 0
            while offset < len(data):
                if hasattr(os, "preadv"):
                    count = os.preadv(fd, [memoryview(data)[offset:]], base + start + offset)
                else:
                    chunk = os.pread(fd, len(data) - offset, base + start + offset)
                    count = len(chunk)
                    data[offset:offset + count] = chunk
                if not count:
                    raise OSError(f"short expert read: {name}")
                offset += count
            result.append((name, meta, data))
        return result

    def run(self, layer, x, indices, original):
        import numpy as np
        import mlx.core as mx

        with self.lock:
            ids = np.unique(np.asarray(indices))
            missing = [int(i) for i in ids if (layer.layer_id, int(i)) not in self.store._lru]
            if not missing or len(ids) * 2 > self.store.num_experts:
                return original(layer, x, indices)
            base, grouped = self._index(layer.layer_id)
            if any(j not in grouped for j in missing):
                raise ValueError("missing expert in SSD weight index")
            # Bound temporary host payloads. A medium prefill can touch many
            # experts without triggering the upstream bulk path; falling back
            # to mx.load here reopens an entire 4608-tensor shard for every
            # cache miss and can exhaust the process's file descriptors.
            size = sum(m["data_offsets"][1] - m["data_offsets"][0] for j in missing for _, m in grouped[j])
            fd = self.fds[layer.layer_id]
            pending = None
            if size <= 128 * 1024 * 1024:
                futures = [self.pool.submit(self._read, fd, base, grouped[j]) for j in missing]
                try:
                    pending = dict(zip(missing, [future.result() for future in futures]))
                except BaseException:
                    for future in futures:
                        future.cancel()
                    for future in futures:
                        if not future.cancelled():
                            try:
                                future.result()
                            except BaseException:
                                pass
                    raise
            previous = self.store._ensure_loaded

            def install(lid, expert, mapping):
                if lid != layer.layer_id:
                    return previous(lid, expert, mapping)
                if f"e{expert}.gate_proj.weight" in mapping:
                    return
                if pending is None:
                    if expert not in grouped:
                        raise ValueError("missing expert in SSD weight index")
                    entries = self._read(fd, base, grouped[expert])
                elif expert in pending:
                    entries = pending.pop(expert)
                else:
                    # An expert that was resident at the beginning of this
                    # call can be evicted while processing earlier experts.
                    # Re-read it directly rather than reopening the shard.
                    if expert not in grouped:
                        raise ValueError("missing expert in SSD weight index")
                    entries = self._read(fd, base, grouped[expert])
                dtypes = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32, "I32": np.int32, "U8": np.uint8}
                for name, meta, data in entries:
                    array = mx.array(np.frombuffer(data, dtype=dtypes[meta["dtype"]]).reshape(meta["shape"]))
                    mapping[name] = array.view(mx.bfloat16) if meta["dtype"] == "BF16" else array

            self.store._ensure_loaded = install
            try:
                return original(layer, x, indices)
            finally:
                self.store._ensure_loaded = previous


def install_parallel_expert_reads(model, *, exact_rows=False):
    from mlx_vlm.models.switch_layers import OffloadedSwitchGLU

    controller = ParallelExpertReads(model.moe_offload_store)

    class ParallelSwitchGLU(OffloadedSwitchGLU):
        def __call__(self, x, indices):
            return controller.run(self, x, indices, OffloadedSwitchGLU.__call__)

        def _proj(self, x, weight, scales, biases, quant):
            # The offload implementation groups all rows routed to one expert.
            # M>1 quantized_matmul uses different reductions from the M=1
            # decode kernel, so keep the expert fetch shared but replay each
            # projection with the single-row arithmetic during verification.
            if (exact_rows or _exact_expert_rows.get()) and x.ndim == 2 and x.shape[0] > 1:
                import mlx.core as mx
                parent = super()._proj
                return mx.concatenate([
                    parent(x[i:i + 1], weight, scales, biases, quant)
                    for i in range(x.shape[0])
                ], axis=0)
            return super()._proj(x, weight, scales, biases, quant)

    changed = []
    try:
        for _, module in model.named_modules():
            if type(module) is OffloadedSwitchGLU:
                object.__setattr__(module, "__class__", ParallelSwitchGLU)
                changed.append(module)
        if not changed:
            raise RuntimeError("No compatible paged expert layers found")
    except BaseException:
        for module in changed:
            object.__setattr__(module, "__class__", OffloadedSwitchGLU)
        controller.close()
        raise
    return controller
