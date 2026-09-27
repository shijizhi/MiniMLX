#!/usr/bin/env python3
"""Create deterministic inputs and a NumPy reference for the Core ML GDN probe."""

import argparse
from pathlib import Path

import numpy as np


def reference(q, k, v, g, beta, state):
    state = state.copy()
    ys = []
    for t in range(q.shape[0]):
        state *= g[t, :, None, None]
        kv = np.einsum("hd,hvd->hv", k[t], state)
        delta = (v[t] - kv) * beta[t, :, None]
        state += np.einsum("hd,hv->hvd", k[t], delta)
        ys.append(np.einsum("hd,hvd->hv", q[t], state))
    return np.stack(ys), state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/tmp/minimlx-gdn-recurrence-8-input"))
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--key-dim", type=int, default=8)
    p.add_argument("--value-dim", type=int, default=4)
    args = p.parse_args()

    rng = np.random.default_rng(20260919)
    shape = (args.tokens, args.heads)
    q = rng.normal(0, 0.2, (*shape, args.key_dim)).astype(np.float32)
    k = rng.normal(0, 0.2, (*shape, args.key_dim)).astype(np.float32)
    v = rng.normal(0, 0.2, (*shape, args.value_dim)).astype(np.float32)
    # Keep the recurrence numerically tame while retaining nontrivial dynamics.
    g = rng.uniform(0.90, 0.999, shape).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, shape).astype(np.float32)
    state = rng.normal(0, 0.05, (args.heads, args.value_dim, args.key_dim)).astype(np.float32)
    y, final_state = reference(q, k, v, g, beta, state)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "reference.npz", q=q, k=k, v=v, g=g, beta=beta, state=state, y=y, final_state=final_state)
    for name, value in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("state", state)):
        value.tofile(args.output / f"{name}.bin")
    (args.output / "meta.txt").write_text(
        f"tokens={args.tokens}\nheads={args.heads}\nkey_dim={args.key_dim}\nvalue_dim={args.value_dim}\n",
        encoding="utf-8",
    )
    print(f"saved={args.output} tokens={args.tokens} heads={args.heads} key_dim={args.key_dim} value_dim={args.value_dim}")


if __name__ == "__main__":
    main()
