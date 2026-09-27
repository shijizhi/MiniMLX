#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace minimlx::fa256 {

mlx::core::array attention(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    float scale,
    int64_t dispatch_budget = 250000000,
    mlx::core::StreamOrDevice stream = {});

}  // namespace minimlx::fa256
