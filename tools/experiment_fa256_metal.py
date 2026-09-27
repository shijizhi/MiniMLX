#!/usr/bin/env python3
"""Independent, deliberately simple FA-256 Metal kernel experiment.

This is a correctness-first kernel, not yet the production implementation.
Each threadgroup owns one (batch, query-head, query-token) and computes the
online-softmax recurrence over causal keys. It establishes the Metal API and
numerical contract before introducing tiled simdgroup MMA.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx


_KERNEL = None
_TILED_KERNEL = None
_QUERY_TILED_KERNEL = None
_MMA_KERNEL = None


def kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    source = r"""
        threadgroup float dot_partials[256];
        threadgroup float dot_value;

        uint tid = thread_position_in_threadgroup.x;
        uint t = threadgroup_position_in_grid.x;
        uint qh = threadgroup_position_in_grid.y;
        uint b = threadgroup_position_in_grid.z;
        uint d = tid;
        uint kvh = qh / GQA;

        size_t q_head_base = ((size_t)b * HQ + qh) * T * D;
        size_t k_head_base = ((size_t)b * HKV + kvh) * T * D;
        size_t v_head_base = k_head_base;
        size_t y_head_base = q_head_base;

        float q_value = float(q[q_head_base + (size_t)t * D + d]);
        float max_score = -INFINITY;
        float denom = 0.0f;
        float accumulator = 0.0f;

        for (uint j = 0; j <= t; ++j) {
            dot_partials[tid] = q_value * float(k[k_head_base + (size_t)j * D + d]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float dot = 0.0f;
                for (uint i = 0; i < D; ++i) dot += dot_partials[i];
                dot_value = dot * 0.0625f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float score = dot_value;
            float new_max = max(max_score, score);
            float old_scale = isfinite(max_score) ? exp(max_score - new_max) : 0.0f;
            float weight = exp(score - new_max);
            denom = denom * old_scale + weight;
            accumulator = accumulator * old_scale + weight * float(v[v_head_base + (size_t)j * D + d]);
            max_score = new_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        y[y_head_base + (size_t)t * D + d] = OutT(accumulator / denom);
    """
    _KERNEL = mx.fast.metal_kernel(
        name="minimlx_fa256_online_softmax_reference",
        input_names=["q", "k", "v"],
        output_names=["y"],
        source=source,
    )
    return _KERNEL


def custom(q, k, v):
    batch, q_heads, tokens, dim = q.shape
    kv_heads = k.shape[1]
    return kernel()(
        inputs=[q, k, v],
        template=[
            ("InT", q.dtype),
            ("OutT", q.dtype),
            ("T", tokens),
            ("HQ", q_heads),
            ("HKV", kv_heads),
            ("D", dim),
            ("GQA", q_heads // kv_heads),
        ],
        # mlx.fast.metal_kernel's grid is in global threads, not threadgroups.
        grid=(tokens * 256, q_heads, batch),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def tiled_kernel():
    global _TILED_KERNEL
    if _TILED_KERNEL is not None:
        return _TILED_KERNEL
    source = r"""
        constexpr uint KB = 8;
        threadgroup float dot_partials[KB * 8];
        threadgroup float dot_values[KB];

        uint tid = thread_position_in_threadgroup.x;
        uint t = threadgroup_position_in_grid.x;
        uint qh = threadgroup_position_in_grid.y;
        uint b = threadgroup_position_in_grid.z;
        uint d = tid;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint kvh = qh / GQA;
        size_t q_head_base = ((size_t)b * HQ + qh) * T * D;
        size_t k_head_base = ((size_t)b * HKV + kvh) * T * D;
        size_t v_head_base = k_head_base;
        size_t y_head_base = q_head_base;
        float q_value = float(q[q_head_base + (size_t)t * D + d]);

        float max_score = -INFINITY;
        float denom = 0.0f;
        float accumulator = 0.0f;
        uint limit = t + 1;
        for (uint j0 = 0; j0 < limit; j0 += KB) {
            uint count = min(KB, limit - j0);
            for (uint r = 0; r < KB; ++r) {
                float partial = 0.0f;
                if (r < count) {
                    partial = q_value * float(k[k_head_base + (size_t)(j0 + r) * D + d]);
                }
                float group_sum = simd_sum(partial);
                if (lane == 0) dot_partials[r * 8 + sg] = group_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                for (uint r = 0; r < KB; ++r) {
                    float dot = 0.0f;
                    for (uint i = 0; i < 8; ++i) dot += dot_partials[r * 8 + i];
                    dot_values[r] = dot * 0.0625f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < count; ++r) {
                float score = dot_values[r];
                float new_max = max(max_score, score);
                float old_scale = isfinite(max_score) ? exp(max_score - new_max) : 0.0f;
                float weight = exp(score - new_max);
                denom = denom * old_scale + weight;
                accumulator = accumulator * old_scale + weight *
                    float(v[v_head_base + (size_t)(j0 + r) * D + d]);
                max_score = new_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        y[y_head_base + (size_t)t * D + d] = OutT(accumulator / denom);
    """
    _TILED_KERNEL = mx.fast.metal_kernel(
        name="minimlx_fa256_online_softmax_tiled_reference",
        input_names=["q", "k", "v"],
        output_names=["y"],
        source=source,
    )
    return _TILED_KERNEL


def custom_tiled(q, k, v):
    batch, q_heads, tokens, dim = q.shape
    kv_heads = k.shape[1]
    return tiled_kernel()(
        inputs=[q, k, v],
        template=[
            ("InT", q.dtype),
            ("OutT", q.dtype),
            ("T", tokens),
            ("HQ", q_heads),
            ("HKV", kv_heads),
            ("D", dim),
            ("GQA", q_heads // kv_heads),
        ],
        grid=(tokens * 256, q_heads, batch),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def query_tiled_kernel():
    global _QUERY_TILED_KERNEL
    if _QUERY_TILED_KERNEL is not None:
        return _QUERY_TILED_KERNEL
    source = r"""
        constexpr uint QB = 8;
        constexpr uint KB = 8;
        constexpr uint SG = 8;
        threadgroup float dot_partials[QB * KB * SG];
        threadgroup float dot_values[QB * KB];

        uint tid = thread_position_in_threadgroup.x;
        uint q_block = threadgroup_position_in_grid.x;
        uint qh = threadgroup_position_in_grid.y;
        uint b = threadgroup_position_in_grid.z;
        uint lane = tid % 32;
        uint sg = tid / 32;
        uint d = tid;
        uint kvh = qh / GQA;
        uint q0 = q_block * QB;
        size_t q_head_base = ((size_t)b * HQ + qh) * T * D;
        size_t k_head_base = ((size_t)b * HKV + kvh) * T * D;
        size_t v_head_base = k_head_base;

        float max_score[QB];
        float denom[QB];
        float accumulator[QB];
        for (uint qi = 0; qi < QB; ++qi) {
            max_score[qi] = -INFINITY;
            denom[qi] = 0.0f;
            accumulator[qi] = 0.0f;
        }

        for (uint j0 = 0; j0 < T; j0 += KB) {
            uint key_count = min(KB, T - j0);
            for (uint qi = 0; qi < QB; ++qi) {
                uint qt = q0 + qi;
                bool valid_q = qt < T;
                for (uint r = 0; r < KB; ++r) {
                    uint jt = j0 + r;
                    float partial = 0.0f;
                    if (valid_q && r < key_count && jt <= qt) {
                        partial = float(q[q_head_base + (size_t)qt * D + d]) *
                            float(k[k_head_base + (size_t)jt * D + d]);
                    }
                    float group_sum = simd_sum(partial);
                    if (lane == 0) {
                        dot_partials[(qi * KB + r) * SG + sg] = group_sum;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                for (uint qi = 0; qi < QB; ++qi) {
                    uint qt = q0 + qi;
                    for (uint r = 0; r < KB; ++r) {
                        uint jt = j0 + r;
                        float dot = 0.0f;
                        for (uint s = 0; s < SG; ++s) {
                            dot += dot_partials[(qi * KB + r) * SG + s];
                        }
                        dot_values[qi * KB + r] =
                            (qt < T && r < key_count && jt <= qt) ? dot * 0.0625f : -INFINITY;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint qi = 0; qi < QB; ++qi) {
                uint qt = q0 + qi;
                if (qt >= T) continue;
                for (uint r = 0; r < KB; ++r) {
                    uint jt = j0 + r;
                    if (r >= key_count || jt > qt) continue;
                    float score = dot_values[qi * KB + r];
                    float new_max = max(max_score[qi], score);
                    float old_scale = isfinite(max_score[qi]) ?
                        exp(max_score[qi] - new_max) : 0.0f;
                    float weight = exp(score - new_max);
                    denom[qi] = denom[qi] * old_scale + weight;
                    accumulator[qi] = accumulator[qi] * old_scale + weight *
                        float(v[v_head_base + (size_t)jt * D + d]);
                    max_score[qi] = new_max;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint qi = 0; qi < QB; ++qi) {
            uint qt = q0 + qi;
            if (qt < T) {
                y[q_head_base + (size_t)qt * D + d] = OutT(accumulator[qi] / denom[qi]);
            }
        }
    """
    _QUERY_TILED_KERNEL = mx.fast.metal_kernel(
        name="minimlx_fa256_query_tiled_reference",
        input_names=["q", "k", "v"],
        output_names=["y"],
        source=source,
    )
    return _QUERY_TILED_KERNEL


def custom_query_tiled(q, k, v):
    batch, q_heads, tokens, dim = q.shape
    kv_heads = k.shape[1]
    return query_tiled_kernel()(
        inputs=[q, k, v],
        template=[
            ("InT", q.dtype),
            ("OutT", q.dtype),
            ("T", tokens),
            ("HQ", q_heads),
            ("HKV", kv_heads),
            ("D", dim),
            ("GQA", q_heads // kv_heads),
        ],
        grid=(((tokens + 7) // 8) * 256, q_heads, batch),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def mma_kernel():
    global _MMA_KERNEL
    if _MMA_KERNEL is not None:
        return _MMA_KERNEL
    source = r"""
        constexpr uint QB = 8;
        constexpr uint KB = 8;
        constexpr uint SG = 8;
        threadgroup float score_tiles[SG * QB * KB];

        uint tid = thread_position_in_threadgroup.x;
        uint lane = tid % 32;
        uint sg = tid / 32;
        uint q_block = threadgroup_position_in_grid.x;
        uint b = threadgroup_position_in_grid.y;
        uint q0 = q_block * QB;
        uint qh = sg;
        uint kvh = qh / GQA;
        // Metal's 8x8 matrix elements are distributed like the Q4
        // simdgroup-matrix probe: each lane owns one row and two adjacent
        // columns.  The previous qid-derived mapping did not match the
        // matrix layout, so the MMA result was numerically invalid despite
        // looking fast.
        uint qid = lane / 4;
        uint sm = (qid & 4) + (lane / 2) % 4;
        uint sn = (qid & 2) * 2 + (lane % 2) * 2;
        uint d = sg * 32 + lane;

        size_t q_head_base = ((size_t)b * HQ + qh) * T * D;
        size_t k_head_base = ((size_t)b * HKV + kvh) * T * D;
        size_t v_head_base = k_head_base;
        float max_score[QB];
        float denom[QB];
        float accumulator[QB];
        for (uint qi = 0; qi < QB; ++qi) {
            max_score[qi] = -INFINITY;
            denom[qi] = 0.0f;
            accumulator[qi] = 0.0f;
        }

        for (uint j0 = 0; j0 < T; j0 += KB) {
            simdgroup_matrix<float, 8, 8> c;
            c.thread_elements()[0] = 0.0f;
            c.thread_elements()[1] = 0.0f;

            for (uint k0 = 0; k0 < D; k0 += 8) {
                simdgroup_matrix<float, 8, 8> a;
                simdgroup_matrix<float, 8, 8> bb;
                uint ar = q0 + sm;
                uint ac = k0 + sn;
                a.thread_elements()[0] = ar < T && ac < D
                    ? float(q[q_head_base + (size_t)ar * D + ac]) : 0.0f;
                a.thread_elements()[1] = ar < T && ac + 1 < D
                    ? float(q[q_head_base + (size_t)ar * D + ac + 1]) : 0.0f;

                uint br = k0 + sm;
                uint bc = j0 + sn;
                bb.thread_elements()[0] = br < D && bc < T
                    ? float(k[k_head_base + (size_t)bc * D + br]) : 0.0f;
                bb.thread_elements()[1] = br < D && bc + 1 < T
                    ? float(k[k_head_base + (size_t)(bc + 1) * D + br]) : 0.0f;
                simdgroup_multiply_accumulate(c, a, bb, c);
            }

            uint score_base = sg * QB * KB + sm * KB + sn;
            uint qt0 = q0 + sm;
            uint kt0 = j0 + sn;
            score_tiles[score_base] =
                (qt0 < T && kt0 < T && kt0 <= qt0) ? c.thread_elements()[0] * 0.0625f : -INFINITY;
            score_tiles[score_base + 1] =
                (qt0 < T && kt0 + 1 < T && kt0 + 1 <= qt0) ? c.thread_elements()[1] * 0.0625f : -INFINITY;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint qi = 0; qi < QB; ++qi) {
                uint qt = q0 + qi;
                if (qt >= T) continue;
                for (uint r = 0; r < KB; ++r) {
                    uint jt = j0 + r;
                    if (jt >= T || jt > qt) continue;
                    float score = score_tiles[sg * QB * KB + qi * KB + r];
                    float new_max = max(max_score[qi], score);
                    float old_scale = isfinite(max_score[qi]) ?
                        exp(max_score[qi] - new_max) : 0.0f;
                    float weight = exp(score - new_max);
                    denom[qi] = denom[qi] * old_scale + weight;
                    accumulator[qi] = accumulator[qi] * old_scale + weight *
                        float(v[v_head_base + (size_t)jt * D + d]);
                    max_score[qi] = new_max;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint qi = 0; qi < QB; ++qi) {
            uint qt = q0 + qi;
            if (qt < T) {
                y[q_head_base + (size_t)qt * D + d] = OutT(accumulator[qi] / denom[qi]);
            }
        }
    """
    _MMA_KERNEL = mx.fast.metal_kernel(
        name="minimlx_fa256_simdgroup_mma_gridfix_reference",
        input_names=["q", "k", "v"],
        output_names=["y"],
        source=source,
    )
    return _MMA_KERNEL


def custom_mma(q, k, v):
    batch, q_heads, tokens, dim = q.shape
    kv_heads = k.shape[1]
    if q_heads != 8 or dim != 256:
        raise ValueError("MMA prototype currently requires 8 query heads and D=256")
    return mma_kernel()(
        inputs=[q, k, v],
        template=[
            ("InT", q.dtype),
            ("OutT", q.dtype),
            ("T", tokens),
            ("HQ", q_heads),
            ("HKV", kv_heads),
            ("D", dim),
            ("GQA", q_heads // kv_heads),
        ],
        # MLX's custom-kernel grid is expressed in threads, not threadgroups.
        # Match the working tiled kernels: one 256-thread group per query block.
        grid=(((tokens + 7) // 8) * 256, 1, batch),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def timed(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        values.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    q = mx.random.normal((1, 8, args.tokens, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, args.tokens, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, args.tokens, 256)).astype(mx.bfloat16)
    mx.eval(q, k, v)

    def reference():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=256**-0.5, mask="causal")

    def candidate():
        return custom(q, k, v)

    expected = reference()
    actual = candidate()
    tiled = custom_tiled(q, k, v)
    query_tiled = custom_query_tiled(q, k, v)
    mma = custom_mma(q, k, v)
    mx.eval(expected, actual, tiled, query_tiled, mma)
    error = float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))))
    tiled_error = float(mx.max(mx.abs(expected.astype(mx.float32) - tiled.astype(mx.float32))))
    query_tiled_error = float(mx.max(mx.abs(expected.astype(mx.float32) - query_tiled.astype(mx.float32))))
    mma_error = float(mx.max(mx.abs(expected.astype(mx.float32) - mma.astype(mx.float32))))
    print("MiniMLX independent FA-256 Metal reference experiment")
    print(f"tokens={args.tokens}, q_heads=8, kv_heads=2, head_dim=256")
    print(f"max_abs_error={error:.8e}")
    print(f"tiled_max_abs_error={tiled_error:.8e}")
    print(f"query_tiled_max_abs_error={query_tiled_error:.8e}")
    print(f"mma_max_abs_error={mma_error:.8e}")
    if args.tokens <= 128:
        print(f"reference_prefix={nprefix(expected)}")
        print(f"custom_prefix={nprefix(actual)}")
        print(f"mma_prefix={nprefix(mma)}")
        mma_diff = mx.abs(expected.astype(mx.float32) - mma.astype(mx.float32))
        mma_flat_index = int(mx.argmax(mma_diff.reshape(-1)))
        print(
            f"mma_worst_index={mma_flat_index} "
            f"reference={float(expected.reshape(-1)[mma_flat_index])} "
            f"mma={float(mma.reshape(-1)[mma_flat_index])}"
        )
    print(f"stock_sdpa={timed(reference, args.warmup, args.repeats):.3f} ms")
    print(f"custom_online_softmax={timed(candidate, args.warmup, args.repeats):.3f} ms")
    print(f"custom_tiled_online_softmax={timed(lambda: custom_tiled(q, k, v), args.warmup, args.repeats):.3f} ms")
    print(f"custom_query_tiled_online_softmax={timed(lambda: custom_query_tiled(q, k, v), args.warmup, args.repeats):.3f} ms")
    print(f"custom_simdgroup_mma={timed(lambda: custom_mma(q, k, v), args.warmup, args.repeats):.3f} ms")


def nprefix(value):
    return [float(x) for x in value.reshape(-1)[:8]]


if __name__ == "__main__":
    main()
