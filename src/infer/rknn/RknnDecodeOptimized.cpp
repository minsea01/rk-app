/**
 * ARM NEON（RK3588）下的 DFL SIMD 解码实现。
 *
 * 优化点：
 * - max/sum/projection 使用 NEON 向量化
 * - exp 保持标量实现以优先保证数值稳定
 *
 * 约束：
 * - reg_max 受限于 <= 32
 */

#include "rkapp/infer/RknnDecodeOptimized.hpp"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define RKAPP_HAS_NEON 1
#endif
#include <array>
#include <cmath>
#include <algorithm>

namespace rkapp::infer {

/**
 * NEON 优化的 DFL softmax + 期望投影
 *
 * 计算分布期望：
 * sum_k( softmax(logits[ch:ch+reg_max]) * k )
 */
float dfl_decode_neon_single(const float* logits, int reg_max, float* probs_buf) {
    // 先求最大值，提升 softmax 数值稳定性。
    float max_val = -1e30f;

#if RKAPP_HAS_NEON
    if (reg_max == 16) {
        // reg_max=16 的专用快速路径（YOLOv8 常见）。
        float32x4_t vmax = vdupq_n_f32(-1e30f);

        // 4x4 SIMD 处理 16 个元素。
        float32x4_t v0 = vld1q_f32(logits + 0);
        float32x4_t v1 = vld1q_f32(logits + 4);
        float32x4_t v2 = vld1q_f32(logits + 8);
        float32x4_t v3 = vld1q_f32(logits + 12);

        vmax = vmaxq_f32(vmax, v0);
        vmax = vmaxq_f32(vmax, v1);
        vmax = vmaxq_f32(vmax, v2);
        vmax = vmaxq_f32(vmax, v3);

        // 向量内归约最大值。
        float32x2_t vmax_high = vget_high_f32(vmax);
        float32x2_t vmax_low = vget_low_f32(vmax);
        float32x2_t vmax_pair = vpmax_f32(vmax_low, vmax_high);
        vmax_pair = vpmax_f32(vmax_pair, vmax_pair);
        max_val = vget_lane_f32(vmax_pair, 0);

        // 计算 exp(x-max) 并累计和。
        float32x4_t vmax_dup = vdupq_n_f32(max_val);

        v0 = vsubq_f32(v0, vmax_dup);
        v1 = vsubq_f32(v1, vmax_dup);
        v2 = vsubq_f32(v2, vmax_dup);
        v3 = vsubq_f32(v3, vmax_dup);

        // exp 仍用标量实现，优先精度。
        vst1q_f32(probs_buf + 0, v0);
        vst1q_f32(probs_buf + 4, v1);
        vst1q_f32(probs_buf + 8, v2);
        vst1q_f32(probs_buf + 12, v3);

        for (int k = 0; k < 16; ++k) {
            probs_buf[k] = std::exp(probs_buf[k]);
        }

        // 重新载入 exp 结果用于向量求和。
        v0 = vld1q_f32(probs_buf + 0);
        v1 = vld1q_f32(probs_buf + 4);
        v2 = vld1q_f32(probs_buf + 8);
        v3 = vld1q_f32(probs_buf + 12);

        // 求和得到 softmax 分母。
        float32x4_t vsum = vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3));
        float32x2_t vsum_high = vget_high_f32(vsum);
        float32x2_t vsum_low = vget_low_f32(vsum);
        float32x2_t vsum_pair = vadd_f32(vsum_low, vsum_high);
        vsum_pair = vpadd_f32(vsum_pair, vsum_pair);
        float sum = vget_lane_f32(vsum_pair, 0);

        if (sum < 1e-10f) sum = 1e-10f;

        // 归一化后与索引权重做点积。
        float32x4_t vsum_inv = vdupq_n_f32(1.0f / sum);
        v0 = vmulq_f32(v0, vsum_inv);
        v1 = vmulq_f32(v1, vsum_inv);
        v2 = vmulq_f32(v2, vsum_inv);
        v3 = vmulq_f32(v3, vsum_inv);

        // 与索引权重 [0..15] 做点积。
        alignas(16) static const float weights0[4] = {0.0f, 1.0f, 2.0f, 3.0f};
        alignas(16) static const float weights1[4] = {4.0f, 5.0f, 6.0f, 7.0f};
        alignas(16) static const float weights2[4] = {8.0f, 9.0f, 10.0f, 11.0f};
        alignas(16) static const float weights3[4] = {12.0f, 13.0f, 14.0f, 15.0f};

        float32x4_t w0 = vld1q_f32(weights0);
        float32x4_t w1 = vld1q_f32(weights1);
        float32x4_t w2 = vld1q_f32(weights2);
        float32x4_t w3 = vld1q_f32(weights3);

        float32x4_t vdot = vmulq_f32(v0, w0);
        vdot = vmlaq_f32(vdot, v1, w1);
        vdot = vmlaq_f32(vdot, v2, w2);
        vdot = vmlaq_f32(vdot, v3, w3);

        // 水平求和。
        float32x2_t vdot_high = vget_high_f32(vdot);
        float32x2_t vdot_low = vget_low_f32(vdot);
        float32x2_t vdot_pair = vadd_f32(vdot_low, vdot_high);
        vdot_pair = vpadd_f32(vdot_pair, vdot_pair);

        return vget_lane_f32(vdot_pair, 0);
    }
#endif

    // 非 reg_max=16 或无 NEON 时走标量回退。
    for (int k = 0; k < reg_max; ++k) {
        max_val = std::max(max_val, logits[k]);
    }

    float sum = 0.0f;
    for (int k = 0; k < reg_max; ++k) {
        probs_buf[k] = std::exp(logits[k] - max_val);
        sum += probs_buf[k];
    }

    sum = std::max(sum, 1e-10f);
    float proj = 0.0f;
    for (int k = 0; k < reg_max; ++k) {
        proj += (probs_buf[k] / sum) * static_cast<float>(k);
    }

    return proj;
}

/**
 * 四边（l/t/r/b）DFL 解码
 *
 * 输入 logits 为 (C,N) 布局，其中 C=4*reg_max+num_classes。
 * 返回值为 [l,t,r,b]（网格单位）。
 */
std::array<float, 4> dfl_decode_4sides_optimized(
    const float* logits,
    int anchor_idx,
    int N,
    int reg_max,
    float* probs_buf
) {
    std::array<float, 4> out{};
    constexpr int kMaxRegMax = 32;
    if (reg_max <= 0 || reg_max > kMaxRegMax) {
        return out;
    }

    for (int side = 0; side < 4; ++side) {
        int ch_base = side * reg_max;

        // 提取当前边在当前 anchor 的分布向量。
        // 布局：logits[(ch + k) * N + anchor_idx]
        alignas(16) float dist[32];  // Max reg_max = 32
        for (int k = 0; k < reg_max; ++k) {
            dist[k] = logits[(ch_base + k) * N + anchor_idx];
        }

        out[side] = dfl_decode_neon_single(dist, reg_max, probs_buf);
    }

    return out;
}

} // namespace rkapp::infer
