#!/usr/bin/env python3
"""Correctness-first reference for chunking the GDN recurrence.

This intentionally uses explicit Dk x Dk transition matrices.  It is not a
production kernel.  Its job is to make the chunk boundary algebra executable
and compare it with the sequential recurrence before attempting a WY/Metal
implementation.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx


def sequential(q, k, v, decay, beta, state):
    ys = []
    for t in range(q.shape[1]):
        state = state * decay[:, t, :, None, None]
        kv = (state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv) * beta[:, t, :, None]
        state = state + delta[..., None] * k[:, t, :, None, :]
        ys.append((state * q[:, t, :, None, :]).sum(axis=-1))
    return mx.stack(ys, axis=1), state


def solve_lower_gpu(interaction, rhs):
    """Solve (I + interaction) @ x = rhs with GPU MLX primitives.

    ``interaction`` is strictly lower triangular.  The Python loop only
    builds the fixed-size graph; every numerical operation remains an MLX
    operation on the current device.  This is deliberately a reference for a
    later fused Metal triangular solve, not a claim of optimal scheduling.
    """
    count = rhs.shape[-2]
    solved = []
    for t in range(count):
        current = rhs[..., t, :]
        if t:
            previous = mx.concatenate(solved, axis=-2)
            current = current - (interaction[..., t, :t, None] * previous).sum(axis=-2)
        solved.append(current[..., None, :])
    return mx.concatenate(solved, axis=-2)


def explicit_chunked(q, k, v, decay, beta, state, chunk_size):
    """Same recurrence, but merged chunk-by-chunk through explicit transforms.

    For each token, the state transition is

      S' = S @ [d * (I - beta * k k^T)] + beta * v k^T.

    A chunk is represented by prefix products of those transforms and prefix
    additive terms.  This is algebraically equivalent to the token loop, but
    makes the future WY/block implementation's boundary contract explicit.
    """
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    ys = []
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        count = end - start
        local_q = q[:, start:end]
        local_k = k[:, start:end]
        local_v = v[:, start:end]
        local_d = decay[:, start:end]
        local_b = beta[:, start:end]

        # One independent chunk for every batch/head.  The dimensions are
        # deliberately small in this experiment; correctness is the point.
        chunk_y = []
        for t in range(count):
            d = local_d[:, t, :, None, None]
            krow = local_k[:, t, :, None, :]
            kcol = local_k[:, t, :, :, None]
            vt = local_v[:, t]
            bt = local_b[:, t, :, None]
            eye = mx.broadcast_to(mx.eye(key_dim, dtype=state.dtype),
                                  (batch, heads, key_dim, key_dim))
            transform = d * (eye - bt[..., None] * (kcol * krow))
            additive = bt[..., None] * (vt[..., :, None] * krow)

            # Apply this token's affine map to the current state.  Keeping
            # this inside the chunk preserves the exact state orientation and
            # is a useful oracle for a future batched prefix implementation.
            state = mx.matmul(state, transform) + additive
            chunk_y.append((state * local_q[:, t, :, None, :]).sum(axis=-1))
        ys.extend(chunk_y)
    return mx.stack(ys, axis=1), state


def wy_chunked_reference(q, k, v, decay, beta, state, chunk_size):
    """Chunkwise solve of the scalar-gated GDN recurrence.

    Normalize the state by the cumulative decay inside a chunk:

      R_t = S_t / prod_{r<=t} decay_r

    If ``c_t`` is the rank-one update coefficient, then the recurrence gives

      c_t + beta_t * sum_{j<t} <k_t,k_j> c_j
        = beta_t * (v_t / D_t - H k_t),

    where ``H`` is the incoming state and ``D_t`` is cumulative decay.  This
    is a lower-triangular system over the time dimension.  Solving it in one
    MLX operation is the reference for a future fused WY/Metal kernel.
    """
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    ys = []
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        count = end - start

        # Move heads before time so each head is an independent batched solve.
        qq = q[:, start:end].transpose(0, 2, 1, 3)
        kk = k[:, start:end].transpose(0, 2, 1, 3)
        vv = v[:, start:end].transpose(0, 2, 1, 3)
        dd = decay[:, start:end].transpose(0, 2, 1)
        bb = beta[:, start:end].transpose(0, 2, 1)

        cumulative = mx.cumprod(dd, axis=-1)
        incoming = state
        incoming_k = mx.matmul(incoming, kk.swapaxes(-1, -2))
        rhs = bb[..., None] * (vv / cumulative[..., None] -
                               incoming_k.swapaxes(-1, -2))

        gram = mx.matmul(kk, kk.swapaxes(-1, -2))
        row = mx.arange(count)[:, None]
        col = mx.arange(count)[None, :]
        lower = row > col
        lower = lower.reshape((1, 1, count, count))
        interaction = mx.where(lower, bb[..., None] * gram, 0.0)
        coeff = solve_lower_gpu(interaction, rhs)
        rank_one = coeff[..., :, :, None] * kk[..., :, None, :]
        normalized = incoming[..., None, :, :] + mx.cumsum(rank_one, axis=2)
        current = normalized * cumulative[..., None, None]
        chunk_y = (current * qq[..., None, :]).sum(axis=-1)
        ys.append(chunk_y.transpose(0, 2, 1, 3))
        state = current[:, :, -1]

    return mx.concatenate(ys, axis=1), state


def timed(fn, warmup, repeats):
    for _ in range(warmup):
        out = fn()
        mx.eval(*out)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        mx.eval(*out)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def normalize_last(x):
    norm = mx.sqrt((x * x).sum(axis=-1, keepdims=True) + 1e-6)
    return x / norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--key-dim", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    b, h = 1, args.heads
    # GDN's q/k are normalized before the recurrence.  Keeping that invariant
    # here is important: unnormalized Dk=128 random keys make the triangular
    # system badly conditioned and exaggerate harmless fp32 ordering errors.
    q = normalize_last(mx.random.normal(
        (b, args.tokens, h, args.key_dim), dtype=mx.float32))
    k = normalize_last(mx.random.normal(
        (b, args.tokens, h, args.key_dim), dtype=mx.float32))
    v = mx.random.normal((b, args.tokens, h, args.value_dim), dtype=mx.float32)
    # Keep decay in a stable range; this is the same multiplicative quantity
    # consumed by mlx-lm's recurrence, not the pre-softplus gate parameter.
    decay = mx.random.uniform(0.85, 0.99, (b, args.tokens, h), dtype=mx.float32)
    beta = mx.random.uniform(0.05, 0.95, (b, args.tokens, h), dtype=mx.float32)
    state = 0.01 * mx.random.normal(
        (b, h, args.value_dim, args.key_dim), dtype=mx.float32)
    mx.eval(q, k, v, decay, beta, state)

    seq = lambda: sequential(q, k, v, decay, beta, state)
    chunk = lambda: explicit_chunked(q, k, v, decay, beta, state, args.chunk)
    wy = lambda: wy_chunked_reference(q, k, v, decay, beta, state, args.chunk)
    seq_out = seq()
    chunk_out = chunk()
    wy_out = wy()
    mx.eval(*seq_out, *chunk_out, *wy_out)
    y_err = mx.max(mx.abs(seq_out[0] - chunk_out[0]))
    state_err = mx.max(mx.abs(seq_out[1] - chunk_out[1]))
    wy_y_err = mx.max(mx.abs(seq_out[0] - wy_out[0]))
    wy_state_err = mx.max(mx.abs(seq_out[1] - wy_out[1]))
    mx.eval(y_err, state_err, wy_y_err, wy_state_err)

    print("GDN explicit chunk reference")
    print(f"tokens={args.tokens}, chunk={args.chunk}, H={h}, "
          f"Dk={args.key_dim}, Dv={args.value_dim}")
    print(f"max_abs_error_y={float(y_err):.8e}")
    print(f"max_abs_error_state={float(state_err):.8e}")
    print(f"wy_max_abs_error_y={float(wy_y_err):.8e}")
    print(f"wy_max_abs_error_state={float(wy_state_err):.8e}")
    print(f"sequential={timed(seq, args.warmup, args.repeats):.3f} ms")
    print(f"explicit_chunked={timed(chunk, args.warmup, args.repeats):.3f} ms")
    print(f"wy_chunked_reference={timed(wy, args.warmup, args.repeats):.3f} ms")
    if (float(y_err) > 1e-4 or float(state_err) > 1e-4 or
            float(wy_y_err) > 1e-4 or float(wy_state_err) > 1e-4):
        raise SystemExit("chunk reference does not match sequential recurrence")


if __name__ == "__main__":
    main()
