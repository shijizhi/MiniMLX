#pragma once

#include "mlx/ops.h"

namespace minimlx_q4_probe {

mlx::core::array reference_qmm(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int group_size,
    int bits,
    mlx::core::StreamOrDevice stream = {});

}  // namespace minimlx_q4_probe
