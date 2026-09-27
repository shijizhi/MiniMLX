#include <nanobind/nanobind.h>

#include "minimlx_q4_qmm.h"
#include "minimlx_fa256.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "MiniMLX adapted Q4 prefill primitive";
  m.def(
      "affine_qmm",
      [](nb::handle x_h,
         nb::handle weight_h,
         nb::handle scales_h,
         nb::handle biases_h,
         int variant) {
        auto& x = *nb::inst_ptr<mlx::core::array>(x_h);
        auto& weight = *nb::inst_ptr<mlx::core::array>(weight_h);
        auto& scales = *nb::inst_ptr<mlx::core::array>(scales_h);
        auto& biases = *nb::inst_ptr<mlx::core::array>(biases_h);
        return minimlx::q4::affine_qmm(
            x, weight, scales, biases, variant);
      },
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8);
  m.def(
      "fa256_attention",
      [](nb::handle q_h, nb::handle k_h, nb::handle v_h, float scale,
         int64_t dispatch_budget) {
        auto& q = *nb::inst_ptr<mlx::core::array>(q_h);
        auto& k = *nb::inst_ptr<mlx::core::array>(k_h);
        auto& v = *nb::inst_ptr<mlx::core::array>(v_h);
        return minimlx::fa256::attention(q, k, v, scale, dispatch_budget);
      },
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "dispatch_budget"_a = 250000000);
}
