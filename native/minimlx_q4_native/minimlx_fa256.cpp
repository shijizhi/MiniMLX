#include "minimlx_fa256.h"

#include <dlfcn.h>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/ops.h"

namespace minimlx::fa256 {
namespace {

using namespace mlx::core;

struct AttnChunkReduceParams {
  int C;
  int H;
  int qL;
  int D;
  int64_t o_chunk_stride;
  int64_t lse_chunk_stride;
  int64_t O_strides[3];
};

std::string module_dir() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void*>(&module_dir), &info)) {
    throw std::runtime_error("cannot locate MiniMLX FA-256 kernel module");
  }
  return std::filesystem::path(info.dli_fname).parent_path().string();
}

bool unsupported(const array& q, const array& k, const array& v) {
  if (q.dtype() != bfloat16 || k.dtype() != bfloat16 || v.dtype() != bfloat16 ||
      q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
      !q.flags().row_contiguous || !k.flags().row_contiguous ||
      !v.flags().row_contiguous) {
    return true;
  }
  return q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0) ||
      q.shape(1) % k.shape(1) != 0 || k.shape(1) != v.shape(1) ||
      k.shape(2) != v.shape(2) || q.shape(2) <= 1 || q.shape(2) > k.shape(2) ||
      q.shape(3) != 256 || k.shape(3) != 256 || v.shape(3) != 256;
}

class Primitive final : public mlx::core::Primitive {
 public:
  explicit Primitive(Stream stream, float scale, int64_t dispatch_budget)
      : mlx::core::Primitive(stream),
        scale_(scale),
        dispatch_budget_(dispatch_budget) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("MiniMLX FA-256 primitive has no CPU path");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    auto& output = outputs[0];

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int qL = q.shape(2);
    const int kL = k.shape(2);
    constexpr int BQ = 32;
    // Match OMLX's qwen35_fa256_attention default k_block=8. Keeping the
    // query tile at 32 while halving the KV tile reduces per-threadgroup
    // working-set size on pre-NAX Apple GPUs.
    constexpr int BK = 8;

    output.set_data(allocator::malloc(output.nbytes()));
    auto library = device.get_library("minimlx_q4_kernels", module_dir());
    const bool align_q = qL % BQ == 0;
    const bool align_k = kL % BK == 0;
    const bool has_mask = false;
    const bool do_causal = true;
    const bool has_sinks = false;
    metal::MTLFCList constants = {
        {&align_q, MTL::DataType::DataTypeBool, 200},
        {&align_k, MTL::DataType::DataTypeBool, 201},
        {&has_mask, MTL::DataType::DataTypeBool, 300},
        {&do_causal, MTL::DataType::DataTypeBool, 301},
        {&has_sinks, MTL::DataType::DataTypeBool, 302},
    };
    auto kernel = device.get_kernel(
        "minimlx_fa256_attention_bfloat16_bq32_bk8_bd256_wm4_wn1",
        library,
        "minimlx_fa256_attention_bfloat16_bq32_bk8_bd256_wm4_wn1",
        constants);

    mlx::steel::AttnParams params{};
    params.B = B;
    params.H = H;
    params.D = 256;
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = H / k.shape(1);
    params.scale = scale_;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.NQ_aligned = qL / BQ;
    params.NK_aligned = kL / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = kL - qL;
    auto q_strides = q.strides();
    auto k_strides = k.strides();
    auto v_strides = v.strides();
    for (int i = 0; i < 3; ++i) {
      params.Q_strides[i] = q_strides[i];
      params.K_strides[i] = k_strides[i];
      params.V_strides[i] = v_strides[i];
      params.O_strides[i] = q_strides[i];
    }

    auto& encoder = metal::get_command_encoder(stream);

    // Adapted from oMLX's long-context dispatch guard.  The stock FA kernel
    // normalizes each query row over the complete key range, so splitting the
    // key axis requires retaining one normalized partial and its logsumexp per
    // chunk, followed by a numerically stable reduce pass.
    const int64_t work = int64_t(B) * H * qL * kL;
    if (dispatch_budget_ > 0 && work > dispatch_budget_) {
      int64_t chunks = (work + dispatch_budget_ - 1) / dispatch_budget_;
      int64_t chunk_keys = (kL + chunks - 1) / chunks;
      chunk_keys = ((chunk_keys + BK - 1) / BK) * BK;
      chunk_keys = std::max<int64_t>(chunk_keys, 4LL * BQ);
      const int n_chunks = int((kL + chunk_keys - 1) / chunk_keys);

      const int64_t o_chunk_stride = int64_t(B) * H * qL * 256;
      const int64_t lse_chunk_stride = int64_t(B) * H * qL;
      array o_part(
          {n_chunks, B, H, qL, 256}, output.dtype(), nullptr,
          std::vector<array>{});
      o_part.set_data(allocator::malloc(o_part.nbytes()));
      array lse_part(
          {n_chunks, B, H, qL}, float32, nullptr, std::vector<array>{});
      lse_part.set_data(allocator::malloc(lse_part.nbytes()));
      encoder.add_temporary(o_part);
      encoder.add_temporary(lse_part);

      for (int c = 0; c < n_chunks; ++c) {
        const int64_t k_start = int64_t(c) * chunk_keys;
        const int kL_c = int(std::min<int64_t>(chunk_keys, kL - k_start));
        const bool align_k_c = kL_c % BK == 0;
        const bool partials_true = true;
        metal::MTLFCList partial_constants = {
            {&align_q, MTL::DataType::DataTypeBool, 200},
            {&align_k_c, MTL::DataType::DataTypeBool, 201},
            {&has_mask, MTL::DataType::DataTypeBool, 300},
            {&do_causal, MTL::DataType::DataTypeBool, 301},
            {&has_sinks, MTL::DataType::DataTypeBool, 302},
            {&partials_true, MTL::DataType::DataTypeBool, 306}};
        auto partial_kernel = device.get_kernel(
            "minimlx_fa256_attention_bfloat16_bq32_bk8_bd256_wm4_wn1",
            library,
            "minimlx_fa256_attention_partial_bfloat16_bq32_bk8_bd256_wm4_wn1",
            partial_constants);
        mlx::steel::AttnParams partial_params{};
        partial_params.B = B;
        partial_params.H = H;
        partial_params.D = 256;
        partial_params.qL = qL;
        partial_params.kL = kL_c;
        partial_params.gqa_factor = H / k.shape(1);
        partial_params.scale = scale_;
        partial_params.NQ = (qL + BQ - 1) / BQ;
        partial_params.NK = (kL_c + BK - 1) / BK;
        partial_params.NQ_aligned = qL / BQ;
        partial_params.NK_aligned = kL_c / BK;
        partial_params.qL_rem = qL % BQ;
        partial_params.kL_rem = kL_c % BK;
        partial_params.qL_off = (kL - qL) - int(k_start);
        for (int i = 0; i < 3; ++i) {
          partial_params.Q_strides[i] = q_strides[i];
          partial_params.K_strides[i] = k_strides[i];
          partial_params.V_strides[i] = v_strides[i];
        }
        partial_params.O_strides[0] = int64_t(H) * qL * 256;
        partial_params.O_strides[1] = int64_t(qL) * 256;
        partial_params.O_strides[2] = 256;

        encoder.set_compute_pipeline_state(partial_kernel);
        encoder.set_input_array(q, 0);
        encoder.set_input_array(
            k, 1, k_start * k.strides(2) * k.itemsize());
        encoder.set_input_array(
            v, 2, k_start * v.strides(2) * v.itemsize());
        encoder.set_output_array(output, 3);
        encoder.set_bytes(partial_params, 4);
        encoder.set_output_array(
            o_part, 14, c * o_chunk_stride * o_part.itemsize());
        encoder.set_output_array(
            lse_part, 15, c * lse_chunk_stride * lse_part.itemsize());
        encoder.dispatch_threadgroups(
            MTL::Size(partial_params.NQ, H, B), MTL::Size(128, 1, 1));
      }

      auto reduce_kernel = device.get_kernel(
          "minimlx_fa256_chunk_reduce_bfloat16", library,
          "minimlx_fa256_chunk_reduce_bfloat16");
      AttnChunkReduceParams reduce_params{
          n_chunks, H, qL, 256, o_chunk_stride, lse_chunk_stride,
          {output.strides(0), output.strides(1), output.strides(2)}};
      encoder.set_compute_pipeline_state(reduce_kernel);
      encoder.set_input_array(o_part, 0);
      encoder.set_input_array(lse_part, 1);
      encoder.set_output_array(output, 2);
      encoder.set_bytes(reduce_params, 3);
      encoder.dispatch_threads(
          MTL::Size(64, qL, int64_t(B) * H), MTL::Size(64, 4, 1));
      return;
    }

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(params, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(params.NQ, H, B), MTL::Size(128, 1, 1));
  }

  DEFINE_NAME(MiniMLXFA256Attention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const mlx::core::Primitive& other) const override {
    const auto& rhs = static_cast<const Primitive&>(other);
    return scale_ == rhs.scale_ && dispatch_budget_ == rhs.dispatch_budget_;
  }
  auto state() const { return std::make_tuple(scale_, dispatch_budget_); }

 private:
  float scale_;
  int64_t dispatch_budget_;
};

}  // namespace

array attention(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    int64_t dispatch_budget,
    StreamOrDevice stream_arg) {
  // See the Q4 primitive: GPUI's worker threads need a stream owned by the
  // evaluating thread when no explicit stream was supplied.
  auto stream = to_stream(stream_arg);
  if (unsupported(q, k, v)) {
    throw std::invalid_argument("unsupported MiniMLX FA-256 attention shape");
  }
  Shape output_shape = q.shape();
  return array(
      std::move(output_shape),
      q.dtype(),
      std::make_shared<Primitive>(stream, scale, dispatch_budget),
      std::vector<array>{q, k, v});
}

}  // namespace minimlx::fa256
