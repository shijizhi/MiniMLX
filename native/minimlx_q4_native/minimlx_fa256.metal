#include "mlx/backend/metal/kernels/utils.h"
#include "minimlx_fa256_block_token.metal.h"

using namespace metal;

// MiniMLX owns this entry point and keeps the supported family deliberately
// narrow: Qwen3.5 BF16 FA-256 with OMLX-aligned 32x8 tiles and GQA.
instantiate_kernel(
    "minimlx_fa256_attention_bfloat16_bq32_bk8_bd256_wm4_wn1",
    attention,
    bfloat16_t,
    32,
    8,
    256,
    4,
    1,
    bfloat16_t,
    float);

instantiate_kernel(
    "minimlx_fa256_chunk_reduce_bfloat16",
    attention_chunk_reduce,
    bfloat16_t);
