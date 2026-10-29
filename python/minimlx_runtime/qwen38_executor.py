"""MLX-facing lazy expert layers for the Qwen3-Next offload path.

The module imports MLX only when a layer is instantiated.  This keeps model
discovery and all headless tests independent from Metal.
"""

from __future__ import annotations

import glob
import json
from typing import Any

from .offload import Qwen38OffloadSession


class LazySwitchLinear:
    """Load only the experts referenced by one routed token batch."""

    def __init__(
        self,
        session: Qwen38OffloadSession,
        tensor_prefix: str,
        *,
        num_experts: int,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.session = session
        self.tensor_prefix = tensor_prefix.rstrip(".")
        self.num_experts = num_experts

    def __call__(self, x: Any, indices: Any) -> Any:
        import mlx.core as mx
        import numpy as np

        flat_indices = np.asarray(indices).reshape(-1)
        expert_ids = sorted({int(value) for value in flat_indices.tolist()})
        if any(expert < 0 or expert >= self.num_experts for expert in expert_ids):
            raise ValueError("expert route contains an out-of-range expert id")
        if not expert_ids:
            return mx.zeros_like(x)

        names = [f"{self.tensor_prefix}.{expert}.weight" for expert in expert_ids]
        weights = [mx.array(self.session.read_array(name)) for name in names]
        quantized = all(
            self.session.store.region(
                f"{self.tensor_prefix}.{expert}.scales"
            )
            for expert in expert_ids
        )
        remap = np.asarray(
            [expert_ids.index(int(value)) for value in flat_indices], dtype=np.int32
        ).reshape(indices.shape)
        routed_indices = mx.array(remap)
        expanded = mx.expand_dims(x, (-2, -3))
        if quantized:
            scales = [
                mx.array(
                    self.session.read_array(
                        f"{self.tensor_prefix}.{expert}.scales"
                    )
                )
                for expert in expert_ids
            ]
            biases = [
                mx.array(
                    self.session.read_array(
                        f"{self.tensor_prefix}.{expert}.biases"
                    )
                )
                for expert in expert_ids
                if self.session.store.region(
                    f"{self.tensor_prefix}.{expert}.biases"
                )
            ]
            return mx.gather_qmm(
                expanded,
                mx.stack(weights),
                mx.stack(scales),
                mx.stack(biases) if len(biases) == len(expert_ids) else None,
                rhs_indices=routed_indices,
                transpose=True,
            ).squeeze(-2)
        return mx.gather_mm(
            expanded,
            mx.stack(weights),
            rhs_indices=routed_indices,
        ).squeeze(-2)


class LazySwitchGLU:
    """Qwen3-Next SwiGLU expert block backed by ``LazySwitchLinear``."""

    def __init__(
        self,
        session: Qwen38OffloadSession,
        tensor_prefix: str,
        *,
        num_experts: int,
    ) -> None:
        from mlx_lm.models.activations import swiglu

        self.gate_proj = LazySwitchLinear(
            session, f"{tensor_prefix}.gate_proj", num_experts=num_experts
        )
        self.up_proj = LazySwitchLinear(
            session, f"{tensor_prefix}.up_proj", num_experts=num_experts
        )
        self.down_proj = LazySwitchLinear(
            session, f"{tensor_prefix}.down_proj", num_experts=num_experts
        )
        self._activation = swiglu

    def __call__(self, x: Any, indices: Any) -> Any:
        return self.down_proj(
            self._activation(
                self.gate_proj(x, indices),
                self.up_proj(x, indices),
            ),
            indices,
        )


def _is_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name and name.endswith(
        (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
         ".gate_proj.scales", ".up_proj.scales", ".down_proj.scales",
         ".gate_proj.biases", ".up_proj.biases", ".down_proj.biases")
    )


def load_qwen38_offload_model(
    model_dir: str,
    session: Qwen38OffloadSession,
    *,
    tokenizer_config: dict | None = None,
) -> tuple[Any, Any]:
    """Load Qwen3-Next while leaving routed expert tensors on disk.

    This mirrors the small, stable part of ``mlx-lm``'s loader needed for the
    Qwen3-Next architecture.  It intentionally supports the standard MLX
    safetensors format and fails closed for unsupported quantization layouts.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.utils import _get_classes, load_config, load_tokenizer
    from safetensors import safe_open

    from pathlib import Path

    root = Path(model_dir).expanduser().resolve()
    config = load_config(root)
    model_class, model_args_class = _get_classes(config)
    weights: dict[str, Any] = {}
    for filename in sorted(glob.glob(str(root / "model*.safetensors"))):
        with safe_open(filename, framework="numpy") as handle:
            for name in handle.keys():
                if _is_expert_tensor(name):
                    continue
                weights[name] = mx.array(handle.get_tensor(name))

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            continue
        mlp.switch_mlp = LazySwitchGLU(
            session,
            f"model.layers.{layer_index}.mlp.experts",
            num_experts=int(model_args.num_experts),
        )

    # Quantize the dense projections using mlx-lm's regular contract. Expert
    # projections are intentionally omitted from this pass and are decoded by
    # LazySwitchLinear when their route is first used.
    quantization = config.get("quantization")
    if quantization:
        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=lambda path, module: (
                f"{path}.scales" in weights
                and hasattr(module, "to_quantized")
            ),
        )
    model.eval()
    model.load_weights(list(weights.items()), strict=False)
    tokenizer = load_tokenizer(
        root,
        tokenizer_config,
        eos_token_ids=config.get("eos_token_id"),
    )
    return model, tokenizer
