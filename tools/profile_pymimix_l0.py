"""Profile the PyMiniMLX L0 mixer with the same boundary as Swift MiniMLX.

Run with the PyMiniMLX virtual environment so MLX uses the local Metal device.
"""

from pathlib import Path
import os
from statistics import median
import sys
import time

import mlx.core as mx

PYMINIMLX_ROOT = Path(os.environ.get("PYMINIMLX_ROOT", "../pyminimlx")).expanduser()
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "path/to/model")).expanduser()

sys.path.insert(0, str(PYMINIMLX_ROOT / "src"))
from pyminimlx.models import Qwen35Model  # noqa: E402
from pyminimlx.nn.gated import _run_gated_delta_metal, _softplus  # noqa: E402


def main() -> None:
    model = Qwen35Model.from_directory(MODEL_DIR)
    layer = model.layers[0]
    token = mx.array([9419], dtype=mx.int32)
    input_value = model.embedding[token]
    mx.eval(input_value)

    def run_once():
        normalized = layer.input_layernorm(input_value)
        mixer_state = {"recurrent": None, "conv": None}
        mixed, _ = layer.mixer(normalized, mixer_state)
        mx.eval(mixed)

    for _ in range(5):
        run_once()

    samples = []
    for _ in range(10):
        start = time.perf_counter()
        run_once()
        samples.append((time.perf_counter() - start) * 1000)

    print("PyMiniMLX L0 mixer profile (empty state, single token, median=10)")
    print(f"  L0 mixer={median(samples):.3f} ms")

    def timed(name, value):
        start = time.perf_counter()
        mx.eval(value)
        return name, (time.perf_counter() - start) * 1000

    def profile_stages():
        mixer = layer.mixer
        projected = mixer.in_proj_qkv(normalized)
        z = mixer.in_proj_z(normalized)
        beta_projection = mixer.in_proj_b(normalized)
        a_projection = mixer.in_proj_a(normalized)
        result = [timed("qkv_projection", projected)]
        result.append(timed("z_projection", z))
        result.append(timed("beta_projection", mx.sigmoid(beta_projection)))
        result.append(timed("a_projection", a_projection + mixer.dt_bias))

        convolved, _ = mixer._conv(projected, None)
        result.append(timed("depthwise_conv", convolved))
        activated = convolved * mx.sigmoid(convolved)
        result.append(timed("silu", activated))

        q_end = mixer.key_size
        k_end = q_end * 2
        q = activated[:, :q_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
        k = activated[:, q_end:k_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
        v = activated[:, k_end:].reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
        repeats = mixer.num_value_heads // mixer.num_key_heads
        if repeats != 1:
            q = mx.repeat(q, repeats, axis=1)
            k = mx.repeat(k, repeats, axis=1)
        q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + 1e-6)
        q = q / mixer.key_head_dim
        result.append(timed("q_normalization_and_scale", q))
        k = k * mx.rsqrt(mx.mean(k * k, axis=-1, keepdims=True) + 1e-6)
        k = k / (mixer.key_head_dim ** 0.5)
        result.append(timed("k_normalization_and_scale", k))
        result.append(timed("v_reshape", v))

        beta = mx.sigmoid(beta_projection)
        decay = -mx.exp(mixer.a_log) * _softplus(a_projection + mixer.dt_bias)
        recurrent = mx.zeros(
            (mixer.num_value_heads, mixer.key_head_dim, mixer.value_head_dim),
            dtype=normalized.dtype,
        )
        recurrent_output, _ = _run_gated_delta_metal(
            q,
            k,
            v,
            beta,
            decay,
            recurrent,
        )
        result.append(timed("state_input", recurrent))
        result.append(timed("decay", decay))
        result.append(timed("delta_recurrence", recurrent_output))
        # Match MiniMLX: evaluate all kernel inputs first, then time only the
        # recurrent kernel and its two outputs.
        mx.eval(q, k, v, beta, decay, recurrent)
        kernel_start = time.perf_counter()
        kernel_output, kernel_state = _run_gated_delta_metal(
            q, k, v, beta, decay, recurrent
        )
        mx.eval(kernel_output, kernel_state)
        result.append((
            "delta_recurrence_kernel_only",
            (time.perf_counter() - kernel_start) * 1000,
        ))
        z_gate = z.reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
        normalized_output = mixer.norm(recurrent_output, z_gate).reshape(1, -1)
        result.append(timed("gated_rms_norm", normalized_output))
        result.append(timed("output_projection", mixer.o_proj(normalized_output)))
        return result

    normalized = layer.input_layernorm(input_value)
    mx.eval(normalized)
    for _ in range(5):
        profile_stages()
    print("PyMiniMLX L0 mixer stage profile (empty state, single token)")
    for name, milliseconds in profile_stages():
        print(f"  {name}={milliseconds:.3f} ms")

    def cumulative_profile():
        # Build one graph and advance it through checkpoints.  Rebuilding a
        # graph for every checkpoint is not cumulative under lazy evaluation.
        mixer = layer.mixer
        projected = mixer.in_proj_qkv(normalized)
        z = mixer.in_proj_z(normalized)
        beta_projection = mixer.in_proj_b(normalized)
        a_projection = mixer.in_proj_a(normalized)
        # _conv stacks the single-token result back to [1, channels].
        # Taking one raw window here would produce [channels] and break the
        # subsequent [sequence, channels] indexing.
        convolved, _ = mixer._conv(projected, None)
        activated = convolved * mx.sigmoid(convolved)
        q_end = mixer.key_size
        k_end = q_end * 2
        q = activated[:, :q_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
        k = activated[:, q_end:k_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
        v = activated[:, k_end:].reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
        q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + 1e-6) / mixer.key_head_dim
        k = k * mx.rsqrt(mx.mean(k * k, axis=-1, keepdims=True) + 1e-6) / (mixer.key_head_dim ** 0.5)
        beta = mx.sigmoid(beta_projection)
        decay = -mx.exp(mixer.a_log) * _softplus(a_projection + mixer.dt_bias)
        recurrent = mx.zeros((mixer.num_value_heads, mixer.key_head_dim, mixer.value_head_dim), dtype=normalized.dtype)
        recurrent_output, _ = _run_gated_delta_metal(q, k, v, beta, decay, recurrent)
        z_gate = z.reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
        normalized_output = mixer.norm(recurrent_output, z_gate).reshape(1, -1)
        selected = {
            "projection": mx.concatenate([projected, z, beta, a_projection], axis=1),
            "convolution": convolved,
            "activation": activated,
            "qkv_preparation": mx.concatenate([q, k, v], axis=-1),
            "recurrence": recurrent_output,
            "gated_rms_norm": normalized_output,
            "output_projection": mixer.o_proj(normalized_output),
        }
        start = time.perf_counter()
        result = []
        for name in ("projection", "convolution", "activation", "qkv_preparation",
                     "recurrence", "gated_rms_norm", "output_projection"):
            mx.eval(selected[name])
            result.append((f"cumulative_{name}", (time.perf_counter() - start) * 1000))
        return result

    print("PyMiniMLX L0 mixer cumulative profile (one graph, sequential eval)")
    for label, milliseconds in cumulative_profile():
        print(f"  {label}={milliseconds:.3f} ms")

    output_input = mixer_output = None
    mixer = layer.mixer
    projected = mixer.in_proj_qkv(normalized)
    convolved, _ = mixer._conv(projected, None)
    activated = convolved * mx.sigmoid(convolved)
    q_end = mixer.key_size
    k_end = q_end * 2
    value = activated[:, k_end:].reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
    z = mixer.in_proj_z(normalized).reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
    recurrent = mx.zeros((mixer.num_value_heads, mixer.key_head_dim, mixer.value_head_dim), dtype=normalized.dtype)
    q = activated[:, :q_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
    k = activated[:, q_end:k_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
    repeats = mixer.num_value_heads // mixer.num_key_heads
    if repeats != 1:
        q = mx.repeat(q, repeats, axis=1)
        k = mx.repeat(k, repeats, axis=1)
    q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + 1e-6) / mixer.key_head_dim
    k = k * mx.rsqrt(mx.mean(k * k, axis=-1, keepdims=True) + 1e-6) / (mixer.key_head_dim ** 0.5)
    beta = mx.sigmoid(mixer.in_proj_b(normalized))
    decay = -mx.exp(mixer.a_log) * _softplus(mixer.in_proj_a(normalized) + mixer.dt_bias)
    recurrent_output, _ = _run_gated_delta_metal(q, k, value, beta, decay, recurrent)
    output_input = mixer.norm(recurrent_output, z).reshape(1, -1)
    mx.eval(output_input)

    def describe(name, value):
        print(f"  {name}: shape={value.shape} dtype={value.dtype}")

    describe("mixer output input", output_input)
    describe("mixer output weight", mixer.o_proj.quantized_weight)
    describe("mixer output scales", mixer.o_proj.quantized_scales)
    describe("mixer output biases", mixer.o_proj.quantized_biases)

    def wrapped_output():
        return mixer.o_proj(output_input)

    def direct_output():
        return mx.quantized_matmul(
            output_input,
            mixer.o_proj.quantized_weight,
            scales=mixer.o_proj.quantized_scales,
            biases=mixer.o_proj.quantized_biases,
            transpose=True,
            group_size=mixer.o_proj.quantized_group_size,
            bits=mixer.o_proj.quantized_bits,
            mode=mixer.o_proj.quantized_mode,
        )

    for _ in range(5):
        mx.eval(wrapped_output())
        mx.eval(direct_output())

    build_start = time.perf_counter()
    def timed_build_eval(builder):
        build_start = time.perf_counter()
        value = builder()
        build_end = time.perf_counter()
        mx.eval(value)
        end = time.perf_counter()
        return (build_end - build_start) * 1000, (end - build_start) * 1000

    for _ in range(5):
        mx.eval(wrapped_output())
        mx.eval(direct_output())

    wrapped_build_samples = []
    wrapped_total_samples = []
    direct_build_samples = []
    direct_total_samples = []

    def record(builder, build_samples, total_samples):
        build, total = timed_build_eval(builder)
        build_samples.append(build)
        total_samples.append(total)

    for index in range(10):
        if index % 2 == 0:
            record(wrapped_output, wrapped_build_samples, wrapped_total_samples)
            record(direct_output, direct_build_samples, direct_total_samples)
        else:
            record(direct_output, direct_build_samples, direct_total_samples)
            record(wrapped_output, wrapped_build_samples, wrapped_total_samples)

    print(f"  mixer output wrapper_total={median(wrapped_total_samples):.3f} ms")
    print(f"  mixer output direct_total={median(direct_total_samples):.3f} ms")
    print(f"  mixer output wrapper_build={median(wrapped_build_samples):.3f} ms")
    print(f"  mixer output direct_build={median(direct_build_samples):.3f} ms")
    print(f"  mixer output wrapper_eval_est={median(wrapped_total_samples) - median(wrapped_build_samples):.3f} ms")
    print(f"  mixer output direct_eval_est={median(direct_total_samples) - median(direct_build_samples):.3f} ms")


if __name__ == "__main__":
    main()
