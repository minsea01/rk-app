/**
 * ARM NEON（RK3588）下的 DFL SIMD 解码接口
 */

#pragma once

#include <array>

namespace rkapp::infer {

/**
 * 单侧分布（一个边）DFL 解码
 */
float dfl_decode_neon_single(const float* logits, int reg_max, float* probs_buf);

/**
 * 一次解码四个边（l/t/r/b）的 DFL 距离
 */
std::array<float, 4> dfl_decode_4sides_optimized(
    const float* logits,
    int anchor_idx,
    int N,
    int reg_max,
    float* probs_buf
);

} // namespace rkapp::infer
