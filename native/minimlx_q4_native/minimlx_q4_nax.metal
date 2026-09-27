// MiniMLX's narrow adaptation of the oMLX Q4 NAX path.
// Only BF16, affine Q4, group_size=64, and oMLX's NAX variant 4 are built.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

using namespace metal;

template <
    typename T,
    const int BM,
    const int BK,
    const int BN,
    const int WM,
    const int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
minimlx_q4_affine_qmm_t_nax(
    const device uint32_t* weight [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* input [[buffer(3)]],
    device T* output [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(T);
  threadgroup T w_tile[BN * BK_padded];

  qmm_t_nax_tgp_impl<T, 64, 4, true, BM, BK, BN, WM, WN>(
      weight,
      scales,
      biases,
      input,
      output,
      w_tile,
      K,
      N,
      M,
      tg,
      lid,
      simd_gid,
      simd_lid);
}

template [[host_name("minimlx_q4_affine_qmm_t_nax_bfloat16_t_bm_64_bk_32_bn_64_wm_2_wn_2")]]
[[kernel]] void minimlx_q4_affine_qmm_t_nax<bfloat16_t, 64, 32, 64, 2, 2>(
    const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*,
    const device bfloat16_t*, device bfloat16_t*, const constant int&,
    const constant int&, const constant int&, uint3, uint, uint, uint);

#endif
