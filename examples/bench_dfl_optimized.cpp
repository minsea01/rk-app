/**
 * DFL解码性能对比测试
 *
 * 对比：
 * - 原版标量实现
 * - NEON SIMD优化版本
 *
 * 编译（板端）：
 *   g++ -std=c++17 -O3 -march=armv8-a+fp+simd \
 *       -I../include \
 *       examples/bench_dfl_optimized.cpp \
 *       src/infer/rknn/RknnDecodeOptimized.cpp \
 *       -o bench_dfl_opt
 *
 * 运行：
 *   ./bench_dfl_opt
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <array>

#include "rkapp/infer/RknnDecodeOptimized.hpp"

// 原版标量DFL解码
std::array<float, 4> dfl_decode_scalar(
    const float* logits,
    int anchor_idx,
    int N,
    int reg_max,
    float* probs_buf
) {
    std::array<float, 4> out{};

    for (int side = 0; side < 4; ++side) {
        int ch_base = side * reg_max;

        // Find max
        float max_val = -1e30f;
        for (int k = 0; k < reg_max; ++k) {
            max_val = std::max(max_val, logits[(ch_base + k) * N + anchor_idx]);
        }

        // Softmax
        float sum = 0.0f;
        for (int k = 0; k < reg_max; ++k) {
            float v = std::exp(logits[(ch_base + k) * N + anchor_idx] - max_val);
            probs_buf[k] = v;
            sum += v;
        }

        // Project
        float proj = 0.0f;
        if (sum > 1e-10f) {
            for (int k = 0; k < reg_max; ++k) {
                proj += (probs_buf[k] / sum) * static_cast<float>(k);
            }
        }

        out[side] = proj;
    }

    return out;
}

int main() {
    const int N = 3549;        // 416×416 @ stride [8,16,32]
    const int reg_max = 16;
    const int num_classes = 80;
    const int C = 4 * reg_max + num_classes;

    std::cout << "========================================\n";
    std::cout << "DFL解码性能对比测试\n";
    std::cout << "========================================\n";
    std::cout << "配置:\n";
    std::cout << "  Anchors: " << N << "\n";
    std::cout << "  reg_max: " << reg_max << "\n";
    std::cout << "  Classes: " << num_classes << "\n";
    std::cout << "\n";

    // 生成随机logits (模拟RKNN输出)
    std::vector<float> logits(C * N);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : logits) {
        v = dist(gen);
    }

    std::vector<float> probs_buf(reg_max);

    const int iterations = 1000;

    // 测试标量版本
    std::cout << "[1/2] 测试标量版本...\n";
    auto t0_scalar = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < N; ++i) {
            auto dfl = dfl_decode_scalar(logits.data(), i, N, reg_max, probs_buf.data());
            (void)dfl;  // Prevent optimization
        }
    }
    auto t1_scalar = std::chrono::high_resolution_clock::now();
    auto duration_scalar = std::chrono::duration_cast<std::chrono::microseconds>(t1_scalar - t0_scalar).count();

    // 测试NEON优化版本
    std::cout << "[2/2] 测试NEON优化版本...\n";
    auto t0_neon = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < N; ++i) {
            auto dfl = rkapp::infer::dfl_decode_4sides_optimized(logits.data(), i, N, reg_max, probs_buf.data());
            (void)dfl;
        }
    }
    auto t1_neon = std::chrono::high_resolution_clock::now();
    auto duration_neon = std::chrono::duration_cast<std::chrono::microseconds>(t1_neon - t0_neon).count();

    // 结果
    double time_per_frame_scalar = duration_scalar / 1000.0 / iterations;  // ms
    double time_per_frame_neon = duration_neon / 1000.0 / iterations;
    double speedup = static_cast<double>(duration_scalar) / duration_neon;

    std::cout << "\n========================================\n";
    std::cout << "性能对比结果\n";
    std::cout << "========================================\n";
    std::cout << "标量版本:   " << time_per_frame_scalar << " ms/frame\n";
    std::cout << "NEON优化:   " << time_per_frame_neon << " ms/frame\n";
    std::cout << "加速比:     " << speedup << "x\n";
    std::cout << "提升:       " << (speedup - 1.0) * 100 << "%\n";
    std::cout << "\n";

    // 验证结果一致性
    auto res_scalar = dfl_decode_scalar(logits.data(), 0, N, reg_max, probs_buf.data());
    auto res_neon = rkapp::infer::dfl_decode_4sides_optimized(logits.data(), 0, N, reg_max, probs_buf.data());

    std::cout << "结果验证 (anchor 0):\n";
    std::cout << "  标量: [" << res_scalar[0] << ", " << res_scalar[1] << ", "
              << res_scalar[2] << ", " << res_scalar[3] << "]\n";
    std::cout << "  NEON: [" << res_neon[0] << ", " << res_neon[1] << ", "
              << res_neon[2] << ", " << res_neon[3] << "]\n";

    float max_diff = 0.0f;
    for (int i = 0; i < 4; ++i) {
        max_diff = std::max(max_diff, std::abs(res_scalar[i] - res_neon[i]));
    }
    std::cout << "  最大误差: " << max_diff << "\n";
    std::cout << (max_diff < 0.01f ? "  ✅ 结果一致\n" : "  ⚠️  结果差异较大\n");

    std::cout << "\n========================================\n";

    return 0;
}
