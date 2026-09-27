#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;

constant constexpr short kSimd = 32;
constant constexpr short kBM = 64;
constant constexpr short kBK = 32;
constant constexpr short kBN = 64;
constant constexpr short kBKPadded = 40;

template <typename T>
struct MiniMLXAdaptedQ4BlockLoader {
  threadgroup T* dst;
  const device uint32_t* src;
  const device T* scales;
  const device T* biases;
  int src_ld;
  short group_step;

  MiniMLXAdaptedQ4BlockLoader(
      const device uint32_t* src_,
      const device T* scales_,
      const device T* biases_,
      int src_ld_,
      threadgroup T* dst_,
      ushort simd_gid,
      ushort simd_lid)
      : dst(dst_),
        src(src_),
        scales(scales_),
        biases(biases_),
        src_ld(src_ld_),
        group_step(0) {
    const short tid = simd_gid * kSimd + simd_lid;
    const short row = tid / 2;
    const short packed_col = (tid % 2) * 2;
    dst += row * kBKPadded + packed_col * 8;
    src += row * (src_ld / 8) + packed_col;
    scales += row * (src_ld / 64);
    biases += row * (src_ld / 64);
  }

  void load() {
    // Match oMLX's QuantizedBlockLoader dequantization: do scale*q+bias in
    // FP32, then round once when storing the dequantized tile to T. Doing
    // the multiply and add directly in BF16 introduces an extra rounding
    // step for every Q4 weight and does not match MLX's affine QMM path.
    const float scale = float(scales[0]);
    const float bias = float(biases[0]);
    for (short p = 0; p < 2; ++p) {
      const uint32_t word = src[p];
      for (short q = 0; q < 8; ++q) {
        dst[p * 8 + q] = T(scale * float((word >> (4 * q)) & 0xfu) + bias);
      }
    }
  }

  void next() {
    src += kBK / 8;
    group_step++;
    if (group_step == 2) {
      group_step = 0;
      scales++;
      biases++;
    }
  }
};

template <typename T>
[[kernel]] void minimlx_q4_affine_qmm_t(
    const device uint32_t* weight [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* input [[buffer(3)]],
    device T* output [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup T x_tile[kBM * kBKPadded];
  threadgroup T w_tile[kBN * kBKPadded];

  using Mma = mlx::steel::BlockMMA<
      T, T, kBM, kBN, kBK, 2, 2, false, true, kBKPadded, kBKPadded>;
  using XLoader = mlx::steel::BlockLoader<
      T, kBM, kBK, kBKPadded, 1, 128>;

  const int row = int(tg.y) * kBM;
  const int num_els = min(int(kBM), M - row);
  const int col = int(tg.x) * kBN;
  const device T* x = input + int64_t(row) * K;
  const device uint32_t* w = weight + int64_t(col) * (K / 8);
  const device T* s = scales + int64_t(col) * (K / 64);
  const device T* b = biases + int64_t(col) * (K / 64);

  XLoader x_loader(x, K, x_tile, simd_gid, simd_lid);
  MiniMLXAdaptedQ4BlockLoader<T> w_loader(
      w, s, b, K, w_tile, simd_gid, simd_lid);
  Mma mma(simd_gid, simd_lid);

  for (int k = 0; k < K; k += kBK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (num_els < kBM) {
      x_loader.load_safe(short2(kBK, num_els));
    } else {
      x_loader.load_unsafe();
    }
    w_loader.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma.mma(x_tile, w_tile);
    x_loader.next();
    w_loader.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < kBM) {
    mma.store_result_safe(
        output + int64_t(row) * N + col, N, short2(kBN, num_els));
  } else {
    mma.store_result(output + int64_t(row) * N + col, N);
  }
}

template [[host_name("minimlx_q4_affine_qmm_t_bfloat16_t_bm_64_bk_32_bn_64")]]
[[kernel]] void minimlx_q4_affine_qmm_t<bfloat16_t>(
    const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*,
    const device bfloat16_t*, device bfloat16_t*, const constant int&,
    const constant int&, const constant int&, uint3, uint, uint);
