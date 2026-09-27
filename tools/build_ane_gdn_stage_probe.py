"""Build one Core ML linear probe for the four GDN input projections."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import mlx.core as mx
import numpy as np
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from mlx_lm import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--layer", type=int, default=17)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    mixer = model.language_model.model.layers[args.layer].linear_attn
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    projections = [getattr(mixer, name) for name in names]
    input_dim = projections[0].weight.shape[1] * 32 // projections[0].bits
    weights = []
    outputs = []
    for name, projection in zip(names, projections):
        if projection.weight.shape[1] * 32 // projection.bits != input_dim:
            raise ValueError(f"{name} has a different input width")
        value = mx.dequantize(
            projection.weight,
            projection.scales,
            projection.biases,
            projection.group_size,
            projection.bits,
            projection.mode,
            dtype=mx.float16,
        )
        weights.append(np.asarray(value))
        outputs.append(value.shape[0])

    joined = np.concatenate(weights, axis=0).astype(np.float16, copy=False)
    output_dim = joined.shape[0]
    bias = np.zeros(output_dim, dtype=np.float16)
    builder = NeuralNetworkBuilder(
        input_features=[("x", datatypes.Array(args.tokens, input_dim))],
        output_features=[("y", datatypes.Array(args.tokens, output_dim))],
        disable_rank5_shape_mapping=True,
    )
    builder.add_inner_product(
        name="gdn_stage_projection",
        W=joined,
        b=bias,
        input_channels=input_dim,
        output_channels=output_dim,
        has_bias=True,
        input_name="x",
        output_name="y",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ct.utils.save_spec(builder.spec, str(args.output))
    np.save(str(args.output) + ".weights.npy", joined)
    print(f"saved={args.output}")
    print(f"layer={args.layer}, tokens={args.tokens}, input_dim={input_dim}")
    print(f"projections={dict(zip(names, outputs))}, output_dim={output_dim}")


if __name__ == "__main__":
    main()
