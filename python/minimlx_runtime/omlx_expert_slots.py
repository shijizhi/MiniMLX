"""Adapt MiniMLX expert shards to oMLX's slot cache."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


def _use_legacy_prefill(shape):
    return len(shape) >= 3 and shape[-2] > 8


def install_omlx_expert_slots(model, controller):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_vlm.models.switch_layers import OffloadedSwitchGLU
    from .omlx_kernels.moe_expert_offload import OffloadSwitchGLU, _ReadPlan

    store = controller.store
    layers = [(path, layer) for path, layer in model.named_modules() if isinstance(layer, OffloadedSwitchGLU)]
    if not layers:
        raise RuntimeError("No offloaded expert layers found")
    original_budget = store._budget
    budget = max(0, original_budget - 512_000_000) // len(layers)
    config = json.loads((Path(next(iter(store._paths.values()))).parent.parent / "config.json").read_text())
    text_config = config.get("text_config") or config
    minimum = max(8, int(text_config.get("num_experts_per_tok") or 0))
    dtypes = {"BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, mx.float16),
              "F32": (np.float32, mx.float32), "U32": (np.uint32, mx.uint32),
              "I32": (np.int32, mx.int32), "U8": (np.uint8, mx.uint8)}

    class Projection(dict):
        pass

    class HybridSwitch(nn.Module):
        def __init__(self, legacy, slots):
            super().__init__()
            self.legacy = legacy
            self.slots = slots

        def __call__(self, x, indices):
            # Preserve the existing prefill's numerical behavior; slot
            # eviction and sorted gather batching can vary with residency.
            if _use_legacy_prefill(x.shape) and os.environ.get("MINIMLX_OMLX_STABLE_PREFILL", "1") != "1":
                return self.legacy(x, indices)
            return self.slots(x, indices)

    class Disk:
        def __init__(self, layer):
            self.layer = layer
            self._store = SimpleNamespace(_staging=None)
            self.base, self.grouped = controller._index(layer)
            self.fd = controller.fds[layer]
            self.file_size = os.fstat(self.fd).st_size

        def entry(self, projection, field, expert):
            name = f"e{expert}.{projection}.{field}"
            return next((meta for key, meta in self.grouped[expert] if key == name), None)

        def has(self, projection, field):
            return self.entry(projection, field, 0) is not None

        def plan(self, projection, field, expert):
            meta = self.entry(projection, field, expert)
            if meta is None:
                raise ValueError(f"Missing tensor {self.layer}:{expert}:{projection}.{field}")
            start, end = meta["data_offsets"]
            transport, view = dtypes[meta["dtype"]]
            if (not 0 <= start <= end or self.base + end > self.file_size
                    or math.prod(meta["shape"]) * np.dtype(transport).itemsize != end - start):
                raise ValueError(f"Invalid expert tensor offsets or shape: {self.layer}:{expert}:{projection}.{field}")
            return _ReadPlan(self.fd, self.base + start, end - start, transport,
                             view if meta["dtype"] == "BF16" else None, tuple(meta["shape"]))

    replacements = []
    for path, layer in layers:
        if layer.activation is None or any(getattr(layer, name, None) is not None for name in
                                           ("gate_bias", "up_bias", "down_bias", "gate_scale", "out_scale")):
            raise RuntimeError(f"Unsupported expert layer {path}")
        disk = Disk(layer.layer_id)
        glu = SimpleNamespace(activation=layer.activation)
        expert_bytes = 0
        for name in ("gate_proj", "up_proj", "down_proj"):
            group, bits, mode = getattr(layer, name.removesuffix("_proj") + "_quant")
            if mode != "affine":
                raise RuntimeError(f"Unsupported quantization in {path}")
            projection = Projection()
            projection.group_size, projection.bits, projection.mode = group, bits, mode
            for field in ("weight", "scales", "biases"):
                meta = disk.entry(name, field, 0)
                if meta is None:
                    if field != "biases":
                        raise RuntimeError(f"Missing {name}.{field} in {path}")
                    continue
                projection[field] = SimpleNamespace(shape=(store.num_experts, *meta["shape"]), dtype=dtypes[meta["dtype"]][1])
                expert_bytes += meta["data_offsets"][1] - meta["data_offsets"][0]
            setattr(glu, name, projection)
        capacity = min(store.num_experts, budget // expert_bytes)
        if capacity < minimum:
            return 0
        replacements.append((path, HybridSwitch(layer, OffloadSwitchGLU(glu, capacity, disk, floor=minimum))))
    for path, replacement in replacements:
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        if isinstance(parent, list):
            parent[int(parts[-1])] = replacement
        else:
            parent[parts[-1]] = replacement
    store._budget = min(original_budget, 512_000_000)
    return len(replacements)
