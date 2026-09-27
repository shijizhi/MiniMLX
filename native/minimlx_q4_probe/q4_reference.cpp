#include "q4_reference.h"

namespace minimlx_q4_probe {

mlx::core::array reference_qmm(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int group_size,
    int bits,
    mlx::core::StreamOrDevice stream) {
  return mlx::core::quantized_matmul(
      x, weight, scales, std::optional<mlx::core::array>(biases), true,
      group_size, bits, "affine", stream);
}

}  // namespace minimlx_q4_probe
