"""Measure MiniMLX's real cached-decode Q/K/V preparation boundary."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from minimlx_runtime.adapters.qwen35 import Qwen35Adapter
from minimlx_runtime.registry import describe_model


def median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def build_qkv_input(adapter: Qwen35Adapter, layer_index: int):
    model = adapter.model
    assert model is not None
    core = model.language_model.model
    token = mx.array([[9419]], dtype=mx.uint32)
    cache = model.make_cache()

    # Establish the cached-decode state with the same token used by the
    # existing PyMiniMLX diagnostic, then select the next token.
    logits = model(token, cache=cache)
    mx.eval(logits, [c.state for c in cache])
    next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)

    hidden = core.embed_tokens(next_token[:, None])
    for index in range(layer_index):
        layer = core.layers[index]
        hidden = layer(hidden, cache=cache[index])
        mx.eval(hidden)

    layer = core.layers[layer_index]
    if not layer.is_linear:
        raise RuntimeError(f"layer {layer_index} is not linear attention")
    mixer = layer.linear_attn
    normalized = layer.input_layernorm(hidden)
    mx.eval(normalized)

    qkv = mixer.in_proj_qkv(normalized)
    mx.eval(qkv)
    conv_state = cache[layer_index][0]
    if conv_state is None:
        conv_state = mx.zeros(
            (1, mixer.conv_kernel_size - 1, mixer.conv_dim),
            dtype=normalized.dtype,
        )
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = nn.silu(mixer.conv1d(conv_input))
    mx.eval(conv_out)
    return mixer, conv_out


def timed_qkv_preparation(mixer, activated):
    q_end = mixer.key_dim
    k_end = 2 * q_end
    repeats = mixer.num_v_heads // mixer.num_k_heads

    q = activated[:, :, :q_end].reshape(
        -1, mixer.num_k_heads, mixer.head_k_dim
    )
    k = activated[:, :, q_end:k_end].reshape(
        -1, mixer.num_k_heads, mixer.head_k_dim
    )
    v = activated[:, :, k_end:].reshape(
        -1, mixer.num_v_heads, mixer.head_v_dim
    )
    if repeats != 1:
        q = mx.repeat(q, repeats, axis=1)
        k = mx.repeat(k, repeats, axis=1)
    q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k * mx.rsqrt(mx.mean(k * k, axis=-1, keepdims=True) + 1e-6)
    q = q / mixer.head_k_dim
    k = k / (mixer.head_k_dim**0.5)
    v = v.astype(mx.float32)
    return q, k, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    adapter = Qwen35Adapter().load(describe_model(args.model_dir))
    mixer, conv_out = build_qkv_input(adapter, args.layer)
    activated = conv_out

    for _ in range(args.warmup):
        result = timed_qkv_preparation(mixer, activated)
        mx.eval(*result)

    samples: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = timed_qkv_preparation(mixer, activated)
        mx.eval(*result)
        samples.append((time.perf_counter() - start) * 1000.0)

    print(
        "MiniMLX cached QKV preparation profile "
        f"(decode, L{args.layer}, warmup={args.warmup}, repeats={args.repeats})"
    )
    print(f"  input_shape={list(activated.shape)} dtype={activated.dtype}")
    print(f"  qkv_preparation_full={median(samples):.3f} ms")


if __name__ == "__main__":
    main()
