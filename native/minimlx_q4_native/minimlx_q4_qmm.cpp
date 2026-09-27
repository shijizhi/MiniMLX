#include "minimlx_q4_qmm.h"

#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace minimlx::q4 {
namespace {

using namespace mlx::core;
std::atomic<bool> nax_runtime_ok{true};
std::atomic<bool> nax_logged{false};
std::atomic<bool> nax_status_logged{false};

// MLX 0.32 declares is_nax_available() but does not export it from libmlx.
// Mirror the version/Apple-GPU-generation gate used by oMLX instead of
// linking against a private or unavailable symbol.
bool minimlx_nax_available() {
  static const bool available = [] {
    if (!metal::is_available()) {
      return false;
    }
    if (!__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      return false;
    }
    auto& device = metal::device(Device::gpu);
    const auto& architecture = device.get_architecture();
    if (architecture.empty()) {
      return false;
    }
    const char suffix = architecture.back();
    const int generation = device.get_architecture_gen();
    const bool available = generation >= (suffix == 'p' ? 18 : 17);
    if (!nax_status_logged.exchange(true)) {
      std::fprintf(
          stderr,
          "[q4-nax] availability arch=%s generation=%d available=%s\\n",
          architecture.c_str(),
          generation,
          available ? "true" : "false");
    }
    return available;
  }();
  return available;
}

std::string module_dir() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void*>(&module_dir), &info)) {
    throw std::runtime_error("cannot locate MiniMLX Q4 kernel module");
  }
  return std::filesystem::path(info.dli_fname).parent_path().string();
}

class Primitive final : public mlx::core::Primitive {
 public:
  explicit Primitive(Stream stream, int variant)
      : mlx::core::Primitive(stream), variant_(variant) {}

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& biases) {
    if (x.dtype() != bfloat16 || weight.dtype() != uint32 ||
        scales.dtype() != bfloat16 || biases.dtype() != bfloat16 ||
        x.ndim() < 2 || weight.ndim() != 2 || scales.ndim() != 2 ||
        biases.ndim() != 2 || !x.flags().row_contiguous ||
        !weight.flags().row_contiguous || !scales.flags().row_contiguous ||
        !biases.flags().row_contiguous) {
      return true;
    }
    const int K = x.shape(-1);
    const int N = weight.shape(0);
    return K <= 0 || N <= 0 || K % 64 != 0 || K % 32 != 0 || N % 64 != 0 ||
        weight.shape(1) * 32 != K * 4 || scales.shape(0) != N ||
        scales.shape(1) != K / 64 || biases.shape() != scales.shape();
  }

  void eval_cpu(
      const std::vector<array>&,
      std::vector<array>&) override {
    throw std::runtime_error("MiniMLX Q4 primitive has no CPU path");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const auto& biases = inputs[3];
    auto& output = outputs[0];
    const int K = x.shape(-1);
    const int N = weight.shape(0);
    const int M = x.size() / K;

    output.set_data(allocator::malloc(output.nbytes()));

    // Match oMLX's dispatch order: NAX is the primary long-prefill path when
    // the device and the separately built metallib support it.  Any runtime
    // incompatibility permanently demotes this process to classic QMM.
    if (minimlx_nax_available() && nax_runtime_ok.load()) {
      try {
        auto nax_library =
            device.get_library("minimlx_q4_nax_kernels", module_dir());
        auto nax_kernel = device.get_kernel(
            "minimlx_q4_affine_qmm_t_nax_bfloat16_t_bm_64_bk_32_bn_64_wm_2_wn_2",
            nax_library);
        auto& nax_encoder = metal::get_command_encoder(stream);
        nax_encoder.set_compute_pipeline_state(nax_kernel);
        nax_encoder.set_input_array(weight, 0);
        nax_encoder.set_input_array(scales, 1);
        nax_encoder.set_input_array(biases, 2);
        nax_encoder.set_input_array(x, 3);
        nax_encoder.set_output_array(output, 4);
        nax_encoder.set_bytes(K, 5);
        nax_encoder.set_bytes(N, 6);
        nax_encoder.set_bytes(M, 7);
        nax_encoder.dispatch_threadgroups(
            MTL::Size((N + 63) / 64, (M + 63) / 64, 1),
            MTL::Size(32, 2, 2));
        if (!nax_logged.exchange(true)) {
          std::fprintf(
              stderr,
              "[q4-nax] route M=%d K=%d N=%d tile=64x32x64\\n",
              M,
              K,
              N);
        }
        return;
      } catch (const std::exception&) {
        if (!nax_logged.exchange(true)) {
          std::fprintf(stderr, "[q4-nax] metallib/pipeline load failed; fallback=classic\\n");
        }
        nax_runtime_ok.store(false);
      }
    }

    auto library = device.get_library("minimlx_q4_kernels", module_dir());
    auto kernel = device.get_kernel(
        "minimlx_q4_affine_qmm_t_bfloat16_t_bm_64_bk_32_bn_64", library);
    auto& encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(scales, 1);
    encoder.set_input_array(biases, 2);
    encoder.set_input_array(x, 3);
    encoder.set_output_array(output, 4);
    encoder.set_bytes(K, 5);
    encoder.set_bytes(N, 6);
    encoder.set_bytes(M, 7);
    encoder.dispatch_threadgroups(
        MTL::Size(N / 64, (M + 63) / 64, 1), MTL::Size(128, 1, 1));
  }

  DEFINE_NAME(MiniMLXQ4AffineQmm)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const mlx::core::Primitive& other) const override {
    const auto& rhs = static_cast<const Primitive&>(other);
    return variant_ == rhs.variant_;
  }
  auto state() const { return std::make_tuple(4, variant_, 64); }

 private:
  int variant_;
};

}  // namespace

array affine_qmm(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    StreamOrDevice stream_arg) {
  // MiniMLX implements the oMLX long-prefill tile (variant 8:
  // BM=64, BK=32, BN=64).  Keep 0 as a compatibility alias for older
  // probes, but production dispatch uses the explicit variant.
  if ((variant != 0 && variant != 8) ||
      Primitive::unsupported(x, weight, scales, biases)) {
    throw std::invalid_argument("unsupported MiniMLX Q4 affine QMM shape");
  }
  // GPUI evaluates this primitive on a worker thread whose MLX default
  // stream may refer to a stream index that was created on another thread.
  // Use a thread-local stream for the implicit case; explicit callers still
  // retain full control over the stream/device.
  auto stream = to_stream(stream_arg);
  Shape output_shape = x.shape();
  output_shape.back() = weight.shape(0);
  return array(
      std::move(output_shape),
      x.dtype(),
      std::make_shared<Primitive>(stream, variant),
      std::vector<array>{x, weight, scales, biases});
}

}  // namespace minimlx::q4
