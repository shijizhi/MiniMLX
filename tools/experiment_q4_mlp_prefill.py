#!/usr/bin/env python3
"""Isolated Q4 MLP prefill experiment.

This deliberately does not patch mlx-lm.  It tests a small, inference-only
affine-Q4 matmul kernel against MLX's quantized_matmul on the real Qwen3.5
MLP shapes.  The kernel is correctness-first and is not yet production code.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def q4_affine_kernel():
    source = r"""
        constexpr uint SIMD = 32;
        constexpr uint VALUES_PER_THREAD = 16;
        constexpr uint PACKS_PER_THREAD = 2;
        constexpr uint BLOCK_K = SIMD * VALUES_PER_THREAD;
        uint row = threadgroup_position_in_grid.x;
        uint out_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint out_base = out_tile * 8 + simd_gid * 4;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float x_local[VALUES_PER_THREAD];

        for (uint k0 = 0; k0 < K; k0 += BLOCK_K) {
            uint input_base = row * K + k0 + lane * VALUES_PER_THREAD;
            uint word_base = k0 / 8 + lane * PACKS_PER_THREAD;
            for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
                x_local[i] = float(input[input_base + i]);
            }
            for (uint p = 0; p < PACKS_PER_THREAD; ++p) {
                uint packed_word = word_base + p;
                uint packed_offset = p * 8;
                for (uint o = 0; o < 4; ++o) {
                    uint out = out_base + o;
                    if (out < O) {
                        uint w = weight[out * PACKED_K + packed_word];
                        uint group = (k0 + lane * VALUES_PER_THREAD + packed_offset) / GROUP_SIZE;
                        float scale = float(scales[out * GROUPS + group]);
                        float bias = float(biases[out * GROUPS + group]);
                        for (uint qlane = 0; qlane < 8; ++qlane) {
                            uint q = (w >> (qlane * 4)) & 0xFu;
                            acc[o] += x_local[packed_offset + qlane] *
                                (scale * float(q) + bias);
                        }
                    }
                }
            }
        }
        for (uint o = 0; o < 4; ++o) {
            uint out = out_base + o;
            if (out < O) {
                float total = simd_sum(acc[o]);
                if (lane == 0) output[row * O + out] = OutT(total);
            }
        }
    """
    return mx.fast.metal_kernel(
        name="minimlx_q4_affine_prefill_reference",
        input_names=["input", "weight", "scales", "biases"],
        output_names=["output"],
        source=source,
    )


def q4_affine_simd_matrix_kernel():
    """A deliberately small 8x8 MMA probe.

    This is not intended to replace MLX's qmm.  It tests whether a custom
    Metal kernel can reach the native path's regime by using the hardware
    simdgroup matrix multiply while decoding the affine Q4 weights on load.
    """
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint qid = lane / 4;
        uint sm = (qid & 4) + (lane / 2) % 4;
        uint sn = (qid & 2) * 2 + (lane % 2) * 2;
        uint m0 = threadgroup_position_in_grid.x * 8;
        uint n0 = threadgroup_position_in_grid.y * 8;

        simdgroup_matrix<float, 8, 8> c;
        c.thread_elements()[0] = 0.0f;
        c.thread_elements()[1] = 0.0f;

        for (uint k0 = 0; k0 < K; k0 += 8) {
            simdgroup_matrix<float, 8, 8> a;
            simdgroup_matrix<float, 8, 8> b;

            uint ar0 = m0 + sm;
            uint ak0 = k0 + sn;
            a.thread_elements()[0] = ar0 < M && ak0 < K
                ? float(input[ar0 * K + ak0]) : 0.0f;
            a.thread_elements()[1] = ar0 < M && ak0 + 1 < K
                ? float(input[ar0 * K + ak0 + 1]) : 0.0f;

            uint bk0 = k0 + sm;
            uint bc0 = n0 + sn;
            for (uint j = 0; j < 2; ++j) {
                uint out = bc0 + j;
                uint kk = bk0;
                uint word = weight[out * PACKED_K + kk / 8];
                uint q = (word >> ((kk % 8) * 4)) & 0xfu;
                uint group = kk / GROUP_SIZE;
                float w0 = float(scales[out * GROUPS + group]) * float(q) +
                    float(biases[out * GROUPS + group]);
                if (j == 0) b.thread_elements()[0] = out < O && kk < K ? w0 : 0.0f;
                else b.thread_elements()[1] = out < O && kk < K ? w0 : 0.0f;
            }

            simdgroup_multiply_accumulate(c, a, b, c);
        }

        uint out_r = m0 + sm;
        uint out_c = n0 + sn;
        if (out_r < M && out_c < O) output[out_r * O + out_c] = OutT(c.thread_elements()[0]);
        if (out_r < M && out_c + 1 < O) output[out_r * O + out_c + 1] = OutT(c.thread_elements()[1]);
    """
    return mx.fast.metal_kernel(
        name="minimlx_q4_affine_simd_matrix_probe",
        input_names=["input", "weight", "scales", "biases"],
        output_names=["output"],
        source=source,
    )


def q4_affine_simd_tiled_kernel():
    """8x8 simdgroup tile probe with explicit shared tiles.

    Unlike the first matrix probe, this version uses the documented 8x8
    fragment lane mapping and stages both A and dequantized B tiles in
    threadgroup memory before the MMA.  It remains an experiment only.
    """
    source = r"""
        constexpr uint SIMD = 32;
        uint lane = thread_index_in_simdgroup;
        uint qid = lane / 4;
        uint fm = (qid & 4) + ((lane / 2) % 4);
        uint fn = (qid & 2) * 2 + (lane % 2) * 2;
        uint m0 = threadgroup_position_in_grid.x * 8;
        uint n0 = threadgroup_position_in_grid.y * 8;

        threadgroup float a_tile[64];
        threadgroup float b_tile[64];
        simdgroup_matrix<float, 8, 8> a;
        simdgroup_matrix<float, 8, 8> b;
        simdgroup_matrix<float, 8, 8> c;
        c.thread_elements()[0] = 0.0f;
        c.thread_elements()[1] = 0.0f;

        for (uint k0 = 0; k0 < K; k0 += 8) {
            uint ar = m0 + fm;
            uint ac = k0 + fn;
            a_tile[fm * 8 + fn] = ar < M && ac < K
                ? float(input[ar * K + ac]) : 0.0f;

            uint br = k0 + fm;
            uint bc = n0 + fn;
            float w = 0.0f;
            if (br < K && bc < O) {
                uint word = weight[bc * PACKED_K + br / 8];
                uint q = (word >> ((br % 8) * 4)) & 0xfu;
                uint group = br / GROUP_SIZE;
                w = float(scales[bc * GROUPS + group]) * float(q) +
                    float(biases[bc * GROUPS + group]);
            }
            b_tile[fm * 8 + fn] = w;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(a, a_tile, 8);
            simdgroup_load(b, b_tile, 8);
            simdgroup_multiply_accumulate(c, a, b, c);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint out_r = m0 + fm;
        uint out_c = n0 + fn;
        if (out_r < M && out_c < O) {
            output[out_r * O + out_c] = OutT(c.thread_elements()[0]);
        }
        if (out_r < M && out_c + 1 < O) {
            output[out_r * O + out_c + 1] = OutT(c.thread_elements()[1]);
        }
    """
    return mx.fast.metal_kernel(
        name="minimlx_q4_affine_simd_tiled_probe",
        input_names=["input", "weight", "scales", "biases"],
        output_names=["output"],
        source=source,
    )


def custom_qmm(kernel, x, layer):
    m = x.shape[0] * x.shape[1]
    k = x.shape[-1]
    out = layer.weight.shape[0]
    packed_k = layer.weight.shape[1]
    groups = layer.scales.shape[1]
    output = kernel(
        inputs=[x.reshape(m, k), layer.weight, layer.scales, layer.biases],
        template=[
            ("InT", x.dtype),
            ("OutT", x.dtype),
            ("K", k),
            ("O", out),
            ("M", m),
            ("PACKED_K", packed_k),
            ("GROUP_SIZE", layer.group_size),
            ("GROUPS", groups),
        ],
        grid=(64 * m, ((out + 7) // 8), 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(m, out)],
        output_dtypes=[x.dtype],
    )[0]
    return output.reshape(*x.shape[:-1], out)


def custom_simd_matrix_qmm(kernel, x, layer):
    m = x.shape[0] * x.shape[1]
    k = x.shape[-1]
    out = layer.weight.shape[0]
    packed_k = layer.weight.shape[1]
    groups = layer.scales.shape[1]
    output = kernel(
        inputs=[x.reshape(m, k), layer.weight, layer.scales, layer.biases],
        template=[
            ("InT", x.dtype),
            ("OutT", x.dtype),
            ("K", k),
            ("O", out),
            ("M", m),
            ("PACKED_K", packed_k),
            ("GROUP_SIZE", layer.group_size),
            ("GROUPS", groups),
        ],
        grid=(((m + 7) // 8) * 32, ((out + 7) // 8) * 32, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, out)],
        output_dtypes=[x.dtype],
    )[0]
    return output.reshape(*x.shape[:-1], out)


def custom_simd_tiled_qmm(kernel, x, layer):
    m = x.shape[0] * x.shape[1]
    k = x.shape[-1]
    out = layer.weight.shape[0]
    packed_k = layer.weight.shape[1]
    groups = layer.scales.shape[1]
    output = kernel(
        inputs=[x.reshape(m, k), layer.weight, layer.scales, layer.biases],
        template=[
            ("InT", x.dtype),
            ("OutT", x.dtype),
            ("K", k),
            ("O", out),
            ("M", m),
            ("PACKED_K", packed_k),
            ("GROUP_SIZE", layer.group_size),
            ("GROUPS", groups),
        ],
        grid=(((m + 7) // 8) * 32, ((out + 7) // 8) * 32, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, out)],
        output_dtypes=[x.dtype],
    )[0]
    return output.reshape(*x.shape[:-1], out)


def timed(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    model, _ = load(args.model_dir)
    layer = model.language_model.model.layers[0]
    mlp = layer.mlp
    hidden = mlp.gate_proj.weight.shape[1] * 32 // mlp.gate_proj.bits
    x = mx.random.normal((1, args.tokens, hidden), dtype=mx.bfloat16)
    mx.eval(x)
    kernel = q4_affine_kernel()
    simd_matrix_kernel = q4_affine_simd_matrix_kernel()

    def stock_gate():
        return mlp.gate_proj(x)

    def custom_gate():
        return custom_qmm(kernel, x, mlp.gate_proj)

    def stock_up():
        return mlp.up_proj(x)

    def custom_up():
        return custom_qmm(kernel, x, mlp.up_proj)

    def stock_down(value):
        return mlp.down_proj(value)

    def custom_down(value):
        return custom_qmm(kernel, value, mlp.down_proj)

    def custom_simd_matrix_gate():
        return custom_simd_matrix_qmm(simd_matrix_kernel, x, mlp.gate_proj)

    gate = stock_gate()
    up = stock_up()
    mx.eval(gate, up)
    down_input = nn.silu(gate) * up
    mx.eval(down_input)

    fused_weight = mx.concatenate([mlp.gate_proj.weight, mlp.up_proj.weight], axis=0)
    fused_scales = mx.concatenate([mlp.gate_proj.scales, mlp.up_proj.scales], axis=0)
    fused_biases = mx.concatenate([mlp.gate_proj.biases, mlp.up_proj.biases], axis=0)
    mx.eval(fused_weight, fused_scales, fused_biases)

    def stock_gate_up_fused():
        output = mx.quantized_matmul(
            x,
            fused_weight,
            scales=fused_scales,
            biases=fused_biases,
            transpose=True,
            group_size=mlp.gate_proj.group_size,
            bits=mlp.gate_proj.bits,
            mode=mlp.gate_proj.mode,
        )
        return output[..., : gate.shape[-1]], output[..., gate.shape[-1] :]

    def stock_mlp():
        return mlp.down_proj(nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x))

    def fused_mlp():
        fused_gate, fused_up = stock_gate_up_fused()
        return mlp.down_proj(nn.silu(fused_gate) * fused_up)

    fused_gate, fused_up = stock_gate_up_fused()
    mx.eval(fused_gate, fused_up)
    fused_gate_error = float(mx.max(mx.abs(gate.astype(mx.float32) - fused_gate.astype(mx.float32))))
    fused_up_error = float(mx.max(mx.abs(up.astype(mx.float32) - fused_up.astype(mx.float32))))
    stock_mlp_out = stock_mlp()
    fused_mlp_out = fused_mlp()
    mx.eval(stock_mlp_out, fused_mlp_out)
    fused_mlp_error = float(
        mx.max(mx.abs(stock_mlp_out.astype(mx.float32) - fused_mlp_out.astype(mx.float32)))
    )

    custom_gate_out = custom_gate()
    custom_up_out = custom_up()
    mx.eval(custom_gate_out, custom_up_out)
    gate_error = float(mx.max(mx.abs(gate.astype(mx.float32) - custom_gate_out.astype(mx.float32))))
    up_error = float(mx.max(mx.abs(up.astype(mx.float32) - custom_up_out.astype(mx.float32))))
    simd_matrix_gate_out = custom_simd_matrix_gate()
    mx.eval(simd_matrix_gate_out)
    simd_matrix_gate_error = float(
        mx.max(mx.abs(gate.astype(mx.float32) - simd_matrix_gate_out.astype(mx.float32)))
    )

    print(
        "MiniMLX isolated Q4 MLP prefill A/B "
        f"(L0, tokens={args.tokens}, warmup={args.warmup}, repeats={args.repeats})"
    )
    print(f"shape: input={x.shape}, gate={gate.shape}, down_input={down_input.shape}")
    print(f"gate_max_abs_error={gate_error:.8e}")
    print(f"up_max_abs_error={up_error:.8e}")
    print(f"simd_matrix_gate_max_abs_error={simd_matrix_gate_error:.8e}")
    print(f"fused_gate_max_abs_error={fused_gate_error:.8e}")
    print(f"fused_up_max_abs_error={fused_up_error:.8e}")
    print(f"fused_mlp_max_abs_error={fused_mlp_error:.8e}")
    print(f"stock_gate={timed(stock_gate, args.warmup, args.repeats):.3f} ms")
    print(f"custom_gate={timed(custom_gate, args.warmup, args.repeats):.3f} ms")
    print(
        f"custom_simd_matrix_gate={timed(custom_simd_matrix_gate, args.warmup, args.repeats):.3f} ms"
    )
    print(f"stock_up={timed(stock_up, args.warmup, args.repeats):.3f} ms")
    print(f"custom_up={timed(custom_up, args.warmup, args.repeats):.3f} ms")
    print(f"stock_gate_up_fused={timed(stock_gate_up_fused, args.warmup, args.repeats):.3f} ms")
    print(f"stock_mlp={timed(stock_mlp, args.warmup, args.repeats):.3f} ms")
    print(f"fused_mlp={timed(fused_mlp, args.warmup, args.repeats):.3f} ms")
    print(f"stock_down={timed(lambda: stock_down(down_input), args.warmup, args.repeats):.3f} ms")
    print(f"custom_down={timed(lambda: custom_down(down_input), args.warmup, args.repeats):.3f} ms")


if __name__ == "__main__":
    main()
