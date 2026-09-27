"""Build a fixed-shape Core ML linear probe for the ANE experiment.

This deliberately creates a standalone probe model. It does not modify the
MiniMLX runtime or convert production model weights yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=2048)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--layer", type=int, default=17)
    args = parser.parse_args()

    if args.model_dir is not None:
        import mlx.core as mx
        from mlx_lm import load

        model, _ = load(str(args.model_dir))
        layer = model.language_model.model.layers[args.layer]
        projection = layer.linear_attn.in_proj_z
        weights = np.array(
            mx.dequantize(
                projection.weight,
                projection.scales,
                projection.biases,
                projection.group_size,
                projection.bits,
                projection.mode,
                dtype=mx.float16,
            )
        )
        args.output_dim, args.input_dim = weights.shape
        print(f"source=Qwen3.5 L{args.layer} in_proj_z")
    else:
        rng = np.random.default_rng(9419)
        weights = rng.standard_normal(
            (args.output_dim, args.input_dim), dtype=np.float32
        ).astype(np.float16)
    bias = np.zeros(args.output_dim, dtype=np.float16)

    input_features = [("x", datatypes.Array(args.tokens, args.input_dim))]
    output_features = [("y", datatypes.Array(args.tokens, args.output_dim))]
    builder = NeuralNetworkBuilder(
        input_features=input_features,
        output_features=output_features,
        disable_rank5_shape_mapping=True,
    )
    builder.add_inner_product(
        name="linear",
        W=weights,
        b=bias,
        input_channels=args.input_dim,
        output_channels=args.output_dim,
        has_bias=True,
        input_name="x",
        output_name="y",
    )
    spec = builder.spec
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ct.utils.save_spec(spec, str(args.output))
    np.save(str(args.output) + ".weights.npy", weights)
    print(f"saved={args.output}")
    print(f"shape=[{args.tokens}, {args.input_dim}] -> [{args.tokens}, {args.output_dim}]")


if __name__ == "__main__":
    main()
