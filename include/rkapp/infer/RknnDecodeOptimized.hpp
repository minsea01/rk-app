/**
 * SIMD-optimized DFL decoding for ARM NEON (RK3588)
 */

#pragma once

#include <array>

namespace rkapp::infer {

/**
 * NEON-optimized single-channel DFL decode
 */
float dfl_decode_neon_single(const float* logits, int reg_max, float* probs_buf);

/**
 * Decode all 4 sides (l, t, r, b) using SIMD acceleration
 */
std::array<float, 4> dfl_decode_4sides_optimized(
    const float* logits,
    int anchor_idx,
    int N,
    int reg_max,
    float* probs_buf
);

} // namespace rkapp::infer
