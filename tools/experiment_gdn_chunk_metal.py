#!/usr/bin/env python3
"""Experimental fused Metal implementation of one scalar-gated GDN chunk.

This is a research kernel only.  It intentionally supports the dense target
shape used by the reference experiment and is not installed into the model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, "python"))
from experiment_gdn_chunk_reference import normalize_last, sequential
from minimlx_runtime.kernels.gdn import gated_delta_prefill


_kernels = {}


def kernel(tokens, heads, key_dim, value_dim):
    key = (tokens, heads, key_dim, value_dim)
    if key in _kernels:
        return _kernels[key]
    source = f"""
        constexpr int NT = {tokens};
        constexpr int NH = {heads};
        constexpr int DK = {key_dim};
        constexpr int DV = {value_dim};

        threadgroup float coeff[NT * DV];

        uint tid = thread_position_in_threadgroup.x;
        uint h = threadgroup_position_in_grid.x;
        uint b = threadgroup_position_in_grid.y;

        auto q_base = q + ((size_t)b * NT * NH + h) * DK;
        auto k_base = k + ((size_t)b * NT * NH + h) * DK;
        auto v_base = v + ((size_t)b * NT * NH + h) * DV;
        auto y_base = y + ((size_t)b * NT * NH + h) * DV;
        auto decay_base = decay + ((size_t)b * NT * NH + h);
        auto beta_base = beta + ((size_t)b * NT * NH + h);
        auto state_base = state_in + ((size_t)b * NH + h) * DV * DK;
        auto state_out_base = state_out + ((size_t)b * NH + h) * DV * DK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0; t < NT; ++t) {{
            if (tid < DV) {{
                uint dv = tid;
                float cumulative_t = 1.0f;
                for (uint p = 0; p <= t; ++p) {{
                    cumulative_t *= float(decay_base[p * NH]);
                }}
                float state_k = 0.0f;
                for (uint d = 0; d < DK; ++d) {{
                    state_k += float(state_base[dv * DK + d]) *
                               float(k_base[t * NH * DK + d]);
                }}
                float rhs = float(beta_base[t * NH]) *
                    (float(v_base[t * NH * DV + dv]) / cumulative_t - state_k);
                for (uint j = 0; j < t; ++j) {{
                    float key_dot = 0.0f;
                    for (uint d = 0; d < DK; ++d) {{
                        key_dot += float(k_base[t * NH * DK + d]) *
                                   float(k_base[j * NH * DK + d]);
                    }}
                    rhs -= float(beta_base[t * NH]) * key_dot *
                           coeff[j * DV + dv];
                }}
                coeff[t * DV + dv] = rhs;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint dv = tid; dv < DV; dv += 256) {{
            float cumulative_t = 1.0f;
            for (uint t = 0; t < NT; ++t) {{
                cumulative_t *= float(decay_base[t * NH]);
                float output = 0.0f;
                for (uint d = 0; d < DK; ++d) {{
                    float normalized = float(state_base[dv * DK + d]);
                    for (uint j = 0; j <= t; ++j) {{
                        normalized += coeff[j * DV + dv] *
                            float(k_base[j * NH * DK + d]);
                    }}
                    float current = cumulative_t * normalized;
                    output += current * float(q_base[t * NH * DK + d]);
                    if (t == NT - 1) {{
                        state_out_base[dv * DK + d] = StT(current);
                    }}
                }}
                y_base[t * NH * DV + dv] = InT(output);
            }}
        }}
    """
    compiled = mx.fast.metal_kernel(
        name=f"minimlx_gdn_chunk_wy_{tokens}_{heads}_{key_dim}_{value_dim}",
        input_names=["q", "k", "v", "decay", "beta", "state_in"],
        output_names=["y", "state_out"],
        source=source,
    )
    _kernels[key] = compiled
    return compiled


def fused(q, k, v, decay, beta, state):
    b, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    return kernel(tokens, heads, key_dim, value_dim)(
        inputs=[q, k, v, decay, beta, state],
        template=[("InT", q.dtype), ("StT", state.dtype)],
        # ``grid`` is the total thread grid, not the number of threadgroups.
        # One 256-thread group owns one head.
        grid=(256 * heads, b, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, tokens, heads, value_dim), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def fused_chunks(q, k, v, decay, beta, state, chunk_size):
    outputs = []
    for start in range(0, q.shape[1], chunk_size):
        end = min(start + chunk_size, q.shape[1])
        chunk_y, state = fused(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            decay[:, start:end],
            beta[:, start:end],
            state,
        )
        outputs.append(chunk_y)
    return mx.concatenate(outputs, axis=1), state


def timed(fn, warmup, repeats):
    for _ in range(warmup):
        mx.eval(*fn())
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        mx.eval(*out)
        values.append((time.perf_counter() - start) * 1000.0)
    values.sort()
    return values[len(values) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    shape = (1, args.tokens, args.heads, args.key_dim)
    q = normalize_last(mx.random.normal(shape, dtype=mx.float32))
    k = normalize_last(mx.random.normal(shape, dtype=mx.float32))
    v = mx.random.normal(
        (1, args.tokens, args.heads, args.value_dim), dtype=mx.float32)
    decay = mx.random.uniform(
        0.85, 0.99, (1, args.tokens, args.heads), dtype=mx.float32)
    beta = mx.random.uniform(
        0.05, 0.95, (1, args.tokens, args.heads), dtype=mx.float32)
    state = 0.01 * mx.random.normal(
        (1, args.heads, args.value_dim, args.key_dim), dtype=mx.float32)
    mx.eval(q, k, v, decay, beta, state)

    reference = lambda: sequential(q, k, v, decay, beta, state)
    candidate = lambda: fused_chunks(q, k, v, decay, beta, state, args.chunk)
    current_custom = lambda: gated_delta_prefill(q, k, v, decay, beta, state)
    ref = reference()
    got = candidate()
    current = current_custom()
    mx.eval(*ref, *got, *current)
    y_error = mx.max(mx.abs(ref[0] - got[0]))
    state_error = mx.max(mx.abs(ref[1] - got[1]))
    current_y_error = mx.max(mx.abs(ref[0] - current[0]))
    current_state_error = mx.max(mx.abs(ref[1] - current[1]))
    mx.eval(y_error, state_error, current_y_error, current_state_error)

    print("GDN fused Metal chunk experiment")
    print(f"tokens={args.tokens}, chunk={args.chunk}, H={args.heads}, "
          f"Dk={args.key_dim}, Dv={args.value_dim}")
    print(f"max_abs_error_y={float(y_error):.8e}")
    print(f"max_abs_error_state={float(state_error):.8e}")
    print(f"current_custom_error_y={float(current_y_error):.8e}")
    print(f"current_custom_error_state={float(current_state_error):.8e}")
    print(f"sequential={timed(reference, args.warmup, args.repeats):.3f} ms")
    print(f"current_custom={timed(current_custom, args.warmup, args.repeats):.3f} ms")
    print(f"fused_metal={timed(candidate, args.warmup, args.repeats):.3f} ms")
    if float(y_error) > 1e-4 or float(state_error) > 1e-4:
        raise SystemExit("fused Metal kernel does not match sequential recurrence")


if __name__ == "__main__":
    main()
