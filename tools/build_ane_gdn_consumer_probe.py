"""Build a fixed-shape Core ML GDN recurrence consumer.

This is a shape/control-flow prototype: it consumes q/k/v/g/beta/z and an
output matrix, keeps the recurrence inside Core ML, then performs gated RMS
normalization and output projection before returning the sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil import Builder as mb, types


def build(tokens: int, heads: int, key_dim: int, value_dim: int, hidden: int):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(tokens, heads, key_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(tokens, heads, key_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(tokens, heads, value_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(tokens, heads), dtype=types.fp32),
            mb.TensorSpec(shape=(tokens, heads), dtype=types.fp32),
            mb.TensorSpec(shape=(tokens, heads, value_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, heads, value_dim, key_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(heads * value_dim, hidden), dtype=types.fp32),
        ]
    )
    def program(q, k, v, g, beta, z, state, output_weight):
        limit = mb.const(val=np.array([tokens], dtype=np.int32))
        zero = mb.const(val=np.array([0], dtype=np.int32))
        one = mb.const(val=np.array([1], dtype=np.int32))
        empty = mb.const(val=np.zeros((tokens, heads, value_dim), dtype=np.float32))

        def cond(i, _state, _ys):
            return mb.less(x=i, y=limit)

        def body(i, state, ys):
            qi = mb.squeeze(x=mb.gather(x=q, indices=i, axis=0), axes=[0])
            ki = mb.squeeze(x=mb.gather(x=k, indices=i, axis=0), axes=[0])
            vi = mb.squeeze(x=mb.gather(x=v, indices=i, axis=0), axes=[0])
            gi = mb.squeeze(x=mb.gather(x=g, indices=i, axis=0), axes=[0])
            betai = mb.squeeze(x=mb.gather(x=beta, indices=i, axis=0), axes=[0])
            decay = mb.reshape(x=gi, shape=[1, heads, 1, 1])
            state_decay = mb.mul(x=state, y=decay)
            key_for_state = mb.reshape(x=ki, shape=[1, heads, 1, key_dim])
            kv = mb.reduce_sum(x=mb.mul(x=state_decay, y=key_for_state), axes=[-1])
            delta = mb.mul(x=mb.sub(x=vi, y=mb.squeeze(x=kv, axes=[0])), y=mb.reshape(x=betai, shape=[1, heads, 1]))
            new_state = mb.add(
                x=state_decay,
                y=mb.mul(x=mb.reshape(x=ki, shape=[1, heads, 1, key_dim]), y=mb.reshape(x=delta, shape=[1, heads, value_dim, 1])),
            )
            y = mb.squeeze(x=mb.reduce_sum(x=mb.mul(x=new_state, y=mb.reshape(x=qi, shape=[1, heads, 1, key_dim])), axes=[-1]), axes=[0])
            update = mb.reshape(x=y, shape=[1, heads, value_dim])
            index = mb.reshape(x=i, shape=[1, 1])
            next_ys = mb.scatter_nd(data=ys, indices=index, updates=update, mode="update")
            return mb.add(x=i, y=one), new_state, next_ys

        _, final_state, recurrent = mb.while_loop(_cond=cond, _body=body, loop_vars=(zero, state, empty))
        rms = mb.reduce_mean(x=mb.mul(x=recurrent, y=recurrent), axes=[-1], keep_dims=True)
        normalized = mb.mul(x=recurrent, y=mb.rsqrt(x=mb.add(x=rms, y=mb.const(val=np.float32(1e-6)))))
        gated = mb.mul(x=normalized, y=mb.silu(x=z))
        flat = mb.reshape(x=gated, shape=[tokens, heads * value_dim])
        projected = mb.matmul(x=flat, y=output_weight)
        return projected, final_state

    return program


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--key-dim", type=int, default=128)
    p.add_argument("--value-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=1024)
    args = p.parse_args()
    model = ct.convert(build(args.tokens, args.heads, args.key_dim, args.value_dim, args.hidden), source="milinternal", convert_to="neuralnetwork")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"saved={args.output} tokens={args.tokens} heads={args.heads} key_dim={args.key_dim} value_dim={args.value_dim} hidden={args.hidden}")


if __name__ == "__main__":
    main()
