#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace minimlx::q4 {

mlx::core::array affine_qmm(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int variant = 0,
    mlx::core::StreamOrDevice stream = {});

}  // namespace minimlx::q4
