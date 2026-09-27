#include <nanobind/nanobind.h>

#include "q4_reference.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "MiniMLX isolated Q4 extension probe";
  m.def(
      "reference_qmm_eval",
      [](nb::handle x_h,
         nb::handle weight_h,
         nb::handle scales_h,
         nb::handle biases_h,
         int group_size,
         int bits) {
        // MLX's array type lives in a nanobind domain that is not always
        // discoverable by nb::cast from an out-of-tree extension.  Retrieve
        // the existing Python instances directly instead of asking nanobind
        // to construct/cast a second array type.
        auto& x = *nb::inst_ptr<mlx::core::array>(x_h);
        auto& weight = *nb::inst_ptr<mlx::core::array>(weight_h);
        auto& scales = *nb::inst_ptr<mlx::core::array>(scales_h);
        auto& biases = *nb::inst_ptr<mlx::core::array>(biases_h);
        auto result = minimlx_q4_probe::reference_qmm(
            x, weight, scales, biases,
            group_size,
            bits);
        result.eval();
      },
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "group_size"_a,
      "bits"_a);

  m.def(
      "reference_qmm_max_abs_error",
      [](nb::handle x_h,
         nb::handle weight_h,
         nb::handle scales_h,
         nb::handle biases_h,
         nb::handle expected_h,
         int group_size,
         int bits) {
        auto& x = *nb::inst_ptr<mlx::core::array>(x_h);
        auto& weight = *nb::inst_ptr<mlx::core::array>(weight_h);
        auto& scales = *nb::inst_ptr<mlx::core::array>(scales_h);
        auto& biases = *nb::inst_ptr<mlx::core::array>(biases_h);
        auto& expected = *nb::inst_ptr<mlx::core::array>(expected_h);
        auto result = minimlx_q4_probe::reference_qmm(
            x, weight, scales, biases, group_size, bits);
        auto diff = mlx::core::max(
            mlx::core::abs(
                mlx::core::astype(result, mlx::core::float32) -
                mlx::core::astype(expected, mlx::core::float32)));
        diff.eval();
        return static_cast<double>(diff.item<float>());
      },
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "expected"_a,
      "group_size"_a,
      "bits"_a);
}
