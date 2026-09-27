"""Build a small Core ML MIL GDN recurrence probe.

This deliberately starts with the stateful recurrence only.  It returns the
last output and final state, so it can validate Core ML control flow before we
add convolution, normalization, and projection consumers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil import Builder as mb, types


def build(tokens: int, heads: int, key_dim: int, value_dim: int):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(tokens, heads, key_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(tokens, heads, key_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(tokens, heads, value_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(tokens, heads), dtype=types.fp32),
            mb.TensorSpec(shape=(tokens, heads), dtype=types.fp32),
            mb.TensorSpec(shape=(1, heads, value_dim, key_dim), dtype=types.fp32),
        ]
    )
    def program(q, k, v, g, beta, state):
        limit = mb.const(val=np.array([tokens], dtype=np.int32))
        zero = mb.const(val=np.array([0], dtype=np.int32))
        one = mb.const(val=np.array([1], dtype=np.int32))

        def cond(i, _state, _y):
            return mb.less(x=i, y=limit)

        def body(i, state, _y):
            qi = mb.squeeze(x=mb.gather(x=q, indices=i, axis=0), axes=[0])
            ki = mb.squeeze(x=mb.gather(x=k, indices=i, axis=0), axes=[0])
            vi = mb.squeeze(x=mb.gather(x=v, indices=i, axis=0), axes=[0])
            gi = mb.squeeze(x=mb.gather(x=g, indices=i, axis=0), axes=[0])
            betai = mb.squeeze(x=mb.gather(x=beta, indices=i, axis=0), axes=[0])

            decay = mb.reshape(x=gi, shape=[1, heads, 1, 1])
            state_decay = mb.mul(x=state, y=decay)
            key_for_state = mb.reshape(x=ki, shape=[1, heads, 1, key_dim])
            kv = mb.reduce_sum(
                x=mb.mul(x=state_decay, y=key_for_state), axes=[-1]
            )
            value_error = mb.sub(x=vi, y=mb.squeeze(x=kv, axes=[0]))
            beta_scale = mb.reshape(x=betai, shape=[1, heads, 1])
            delta = mb.mul(x=value_error, y=beta_scale)
            key_for_update = mb.reshape(x=ki, shape=[1, heads, 1, key_dim])
            delta_for_update = mb.reshape(x=delta, shape=[1, heads, value_dim, 1])
            new_state = mb.add(
                x=state_decay,
                y=mb.mul(x=key_for_update, y=delta_for_update),
            )
            query_for_output = mb.reshape(x=qi, shape=[1, heads, 1, key_dim])
            y = mb.squeeze(
                x=mb.reduce_sum(
                    x=mb.mul(x=new_state, y=query_for_output), axes=[-1]
                ),
                axes=[0],
            )
            return mb.add(x=i, y=one), new_state, y

        _i, final_state, final_y = mb.while_loop(
            _cond=cond,
            _body=body,
            loop_vars=(
                zero,
                state,
                mb.const(val=np.zeros((heads, value_dim), dtype=np.float32)),
            ),
        )
        return final_y, final_state

    return program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--key-dim", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=4)
    args = parser.parse_args()

    prog = build(args.tokens, args.heads, args.key_dim, args.value_dim)
    model = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(
        f"saved={args.output} tokens={args.tokens} heads={args.heads} "
        f"key_dim={args.key_dim} value_dim={args.value_dim}"
    )


if __name__ == "__main__":
    main()
