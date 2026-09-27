"""Profile cached GatedDeltaNet stages with the same boundaries as Swift."""

from pathlib import Path
import sys
import time

import mlx.core as mx

PYMINIMLX_ROOT = Path("/Users/jizhishi/mycode/ai/pyminimlx")
sys.path.insert(0, str(PYMINIMLX_ROOT / "src"))

from pyminimlx.models import Qwen35Model  # noqa: E402
from pyminimlx.nn.gated import _run_gated_delta_metal, _softplus  # noqa: E402


ENDPOINT_NAMES = (
    "qkv_projection", "z_projection", "beta_projection", "a_projection",
    "convolution", "activation", "qkv_preparation",
    "recurrence", "gated_rms_norm", "output_projection",
)
NAMES = (*ENDPOINT_NAMES, "gated_rms_norm_full")


def profile_once(model, layer_index: int):
    logits, state = model.prefill([9419])
    mx.eval(logits)
    next_token = int(mx.argmax(logits[-1]).item())
    hidden = model.embedding[mx.array([next_token], dtype=mx.int32)]
    mx.eval(hidden)
    for index in range(layer_index):
        hidden = model.layers[index](hidden, state)
        mx.eval(hidden)

    layer = model.layers[layer_index]
    normalized = layer.input_layernorm(hidden)
    mx.eval(normalized)
    mixer = layer.mixer
    projected = mixer.in_proj_qkv(normalized)
    z = mixer.in_proj_z(normalized)
    beta_projection = mixer.in_proj_b(normalized)
    a_projection = mixer.in_proj_a(normalized)
    convolved, _ = mixer._conv(
        projected, state.layers[layer_index].conv)
    activated = convolved * mx.sigmoid(convolved)
    q_end = mixer.key_size
    k_end = q_end * 2
    q = activated[:, :q_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
    k = activated[:, q_end:k_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
    v = activated[:, k_end:].reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
    repeats = mixer.num_value_heads // mixer.num_key_heads
    if repeats != 1:
        q = mx.repeat(q, repeats, axis=1)
        k = mx.repeat(k, repeats, axis=1)
    q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + 1e-6) / mixer.key_head_dim
    k = k * mx.rsqrt(mx.mean(k * k, axis=-1, keepdims=True) + 1e-6) / (mixer.key_head_dim ** 0.5)
    beta = mx.sigmoid(beta_projection)
    a = a_projection + mixer.dt_bias
    decay = -mx.exp(mixer.a_log) * _softplus(a)
    recurrent = state.layers[layer_index].recurrent
    if recurrent is None:
        recurrent = mx.zeros(
            (mixer.num_value_heads, mixer.key_head_dim, mixer.value_head_dim),
            dtype=normalized.dtype,
        )
    recurrent_output, _ = _run_gated_delta_metal(
        q, k, v, beta, decay, recurrent)
    z_gate = z.reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
    normalized_output = mixer.norm(recurrent_output, z_gate).reshape(1, -1)
    output = mixer.o_proj(normalized_output)
    endpoints = (
        projected,
        z,
        beta,
        a,
        convolved,
        activated,
        mx.concatenate([q, k, v], axis=-1),
        recurrent_output,
        normalized_output,
        output,
    )
    values = {}
    previous = time.perf_counter()
    for name, value in zip(ENDPOINT_NAMES, endpoints):
        mx.eval(value)
        now = time.perf_counter()
        values[name] = (now - previous) * 1000.0
        previous = now
    # Keep a focused probe for the current optimization target.  The normal
    # stage value above is sigmoid(in_proj_b(x)); these two timings separate
    # the quantized projection from the scalar sigmoid boundary.
    beta_raw_probe = mixer.in_proj_b(normalized)
    start = time.perf_counter()
    mx.eval(beta_raw_probe)
    values["beta_projection_eval"] = (time.perf_counter() - start) * 1000.0
    beta_sigmoid_probe = mx.sigmoid(beta_raw_probe)
    start = time.perf_counter()
    mx.eval(beta_sigmoid_probe)
    values["beta_sigmoid_eval"] = (time.perf_counter() - start) * 1000.0

    # Function-level probe: measure the production norm call as one graph
    # boundary.  The manually split probes below are useful for diagnosis,
    # but their sum includes multiple dispatch/synchronization boundaries.
    norm_probe = mixer.norm(recurrent_output, z_gate)
    start = time.perf_counter()
    mx.eval(norm_probe)
    values["gated_rms_norm_full"] = (time.perf_counter() - start) * 1000.0
    return values


def profile_real_once(model, layer_index: int):
    """Measure the production dependency chain, syncing at phase boundaries."""
    logits, state = model.prefill([9419])
    mx.eval(logits)
    next_token = int(mx.argmax(logits[-1]).item())
    hidden = model.embedding[mx.array([next_token], dtype=mx.int32)]
    mx.eval(hidden)
    for index in range(layer_index):
        hidden = model.layers[index](hidden, state)
        mx.eval(hidden)

    layer = model.layers[layer_index]
    normalized = layer.input_layernorm(hidden)
    mx.eval(normalized)
    mixer = layer.mixer
    previous = time.perf_counter()
    projected = mixer.in_proj_qkv(normalized)
    mx.eval(projected)
    values = {"real_qkv_projection": (time.perf_counter() - previous) * 1000.0}
    previous = time.perf_counter()
    z = mixer.in_proj_z(normalized)
    mx.eval(z)
    values["real_z_projection"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    beta = mx.sigmoid(mixer.in_proj_b(normalized))
    mx.eval(beta)
    values["real_beta_projection"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    a = mixer.in_proj_a(normalized) + mixer.dt_bias
    mx.eval(a)
    values["real_a_projection"] = (time.perf_counter() - previous) * 1000.0

    previous = time.perf_counter()
    convolved, _ = mixer._conv(projected, state.layers[layer_index].conv)
    mx.eval(convolved)
    values["real_convolution"] = (time.perf_counter() - previous) * 1000.0

    previous = time.perf_counter()
    activated = convolved * mx.sigmoid(convolved)
    mx.eval(activated)
    values["real_activation"] = (time.perf_counter() - previous) * 1000.0

    previous = time.perf_counter()
    q_end = mixer.key_size
    k_end = q_end * 2
    q = activated[:, :q_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
    k = activated[:, q_end:k_end].reshape(-1, mixer.num_key_heads, mixer.key_head_dim)
    v = activated[:, k_end:].reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
    mx.eval(q, k, v)
    values["real_qkv_reshape"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    repeats = mixer.num_value_heads // mixer.num_key_heads
    if repeats != 1:
        q = mx.repeat(q, repeats, axis=1)
        k = mx.repeat(k, repeats, axis=1)
    mx.eval(q, k)
    values["real_qkv_repeat"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + 1e-6)
    mx.eval(q)
    values["real_q_normalization"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    k = k * mx.rsqrt(mx.mean(k * k, axis=-1, keepdims=True) + 1e-6)
    mx.eval(k)
    values["real_k_normalization"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    q = q / mixer.key_head_dim
    mx.eval(q)
    values["real_q_scale"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    k = k / (mixer.key_head_dim ** 0.5)
    v = v.astype(mx.float32)
    mx.eval(k, v)
    values["real_k_scale"] = (time.perf_counter() - previous) * 1000.0

    # Function-level Q/K/V preparation probe.  Rebuild the complete
    # preparation graph from the convolution output and synchronize only at
    # its output, so the individual probes above are not summed as a cost.
    previous = time.perf_counter()
    q_probe = activated[:, :q_end].reshape(
        -1, mixer.num_key_heads, mixer.key_head_dim
    )
    k_probe = activated[:, q_end:k_end].reshape(
        -1, mixer.num_key_heads, mixer.key_head_dim
    )
    v_probe = activated[:, k_end:].reshape(
        -1, mixer.num_value_heads, mixer.value_head_dim
    )
    if repeats != 1:
        q_probe = mx.repeat(q_probe, repeats, axis=1)
        k_probe = mx.repeat(k_probe, repeats, axis=1)
    q_probe = q_probe * mx.rsqrt(
        mx.mean(q_probe * q_probe, axis=-1, keepdims=True) + 1e-6
    ) / mixer.key_head_dim
    k_probe = k_probe * mx.rsqrt(
        mx.mean(k_probe * k_probe, axis=-1, keepdims=True) + 1e-6
    ) / (mixer.key_head_dim ** 0.5)
    v_probe = v_probe.astype(mx.float32)
    mx.eval(q_probe, k_probe, v_probe)
    values["real_qkv_preparation_full"] = (time.perf_counter() - previous) * 1000.0

    previous = time.perf_counter()
    decay = -mx.exp(mixer.a_log) * _softplus(a)
    recurrent = state.layers[layer_index].recurrent
    if recurrent is None:
        recurrent = mx.zeros((mixer.num_value_heads, mixer.key_head_dim, mixer.value_head_dim), dtype=normalized.dtype)
    recurrent_output, recurrent = _run_gated_delta_metal(q, k, v, beta, decay, recurrent)
    mx.eval(recurrent_output, recurrent)
    values["real_recurrence"] = (time.perf_counter() - previous) * 1000.0

    previous = time.perf_counter()
    value = recurrent_output.astype(mx.float32)
    mx.eval(value)
    values["real_gated_norm_cast"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    rms = mx.mean(value * value, axis=-1, keepdims=True)
    mx.eval(rms)
    values["real_gated_norm_mean"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    normalized_value = (value * mx.rsqrt(rms + mixer.norm.eps)).astype(recurrent_output.dtype)
    mx.eval(normalized_value)
    values["real_gated_norm_rsqrt"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    gate_value = z.reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
    gate_value = gate_value * mx.sigmoid(gate_value)
    mx.eval(gate_value)
    values["real_gated_norm_gate_silu"] = (time.perf_counter() - previous) * 1000.0
    previous = time.perf_counter()
    normalized_output = (normalized_value * mixer.norm.weight * gate_value).reshape(1, -1)
    mx.eval(normalized_output)
    values["real_gated_norm_multiply"] = (time.perf_counter() - previous) * 1000.0

    # Measure the actual production function with a single synchronization.
    # Do not infer this value by summing the split probes above: each split
    # forces a separate graph boundary and can overstate the real cost.
    gate_probe = z.reshape(-1, mixer.num_value_heads, mixer.value_head_dim)
    norm_probe = mixer.norm(recurrent_output, gate_probe)
    previous = time.perf_counter()
    mx.eval(norm_probe)
    values["real_gated_norm_full"] = (time.perf_counter() - previous) * 1000.0

    previous = time.perf_counter()
    output = mixer.o_proj(normalized_output)
    mx.eval(output)
    values["real_output_projection"] = (time.perf_counter() - previous) * 1000.0
    return values


def main() -> None:
    real = len(sys.argv) > 1 and sys.argv[1] == "--real"
    args = sys.argv[2:] if real else sys.argv[1:]
    if len(args) not in {1, 2}:
        raise SystemExit(
            "usage: profile_pymimix_cached_linear_stages.py [--real] <model-directory> [layer]"
        )
    model = Qwen35Model.from_directory(Path(args[0]))
    layer_index = int(args[1]) if len(args) == 2 else 0
    if model.layers[layer_index].block_type != "linear_attention":
        raise SystemExit(f"layer {layer_index} is not linear_attention")
    profile = profile_real_once if real else profile_once
    for _ in range(3):
        profile(model, layer_index)
    if real:
        names = (
            "real_qkv_projection", "real_z_projection", "real_beta_projection",
            "real_a_projection", "real_convolution", "real_activation",
            "real_qkv_reshape", "real_qkv_repeat", "real_q_normalization",
            "real_k_normalization", "real_q_scale", "real_k_scale",
            "real_qkv_preparation_full",
            "real_recurrence",
            "real_gated_norm_cast", "real_gated_norm_mean",
            "real_gated_norm_rsqrt", "real_gated_norm_gate_silu",
            "real_gated_norm_multiply", "real_gated_norm_full",
            "real_output_projection",
        )
        samples = {name: [] for name in names}
        for _ in range(5):
            values = profile_real_once(model, layer_index)
            for name in names:
                samples[name].append(values[name])
        print(f"PyMiniMLX real linear-attention phase profile (cached decode, L{layer_index})")
        for name in names:
            ordered = sorted(samples[name])
            print(f"  L{layer_index} {name}={ordered[len(ordered) // 2]:.3f} ms")
        return
    samples = {name: [] for name in (*NAMES, "beta_projection_eval", "beta_sigmoid_eval")}
    for _ in range(5):
        values = profile_once(model, layer_index)
        for name in samples:
            samples[name].append(values[name])
    print(
        "PyMiniMLX cached linear-attention stage profile "
        f"(prefill + decode, L{layer_index}, warmup=3, repeats=5)"
    )
    for name in NAMES:
        ordered = sorted(samples[name])
        print(f"  L{layer_index} {name}={ordered[len(ordered) // 2]:.3f} ms")
    for name in ("beta_projection_eval", "beta_sigmoid_eval"):
        ordered = sorted(samples[name])
        print(f"  L{layer_index} {name}={ordered[len(ordered) // 2]:.3f} ms")


if __name__ == "__main__":
    main()
