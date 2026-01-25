#include "rkapp/infer/RknnDecodeUtils.hpp"
#include "rkapp/infer/RknnDecodeOptimized.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <vector>

using rkapp::infer::AnchorLayout;
using rkapp::infer::build_anchor_layout;
using rkapp::infer::dfl_decode_4sides_optimized;
using rkapp::infer::dfl_decode_neon_single;
using rkapp::infer::resolve_stride_set;

namespace {

float softmax_expected_value(const float* logits, int reg_max) {
  float maxv = -1e30f;
  for (int k = 0; k < reg_max; ++k) {
    maxv = std::max(maxv, logits[k]);
  }
  float sum = 0.0f;
  float proj = 0.0f;
  for (int k = 0; k < reg_max; ++k) {
    float p = std::exp(logits[k] - maxv);
    sum += p;
    proj += p * static_cast<float>(k);
  }
  if (sum < 1e-10f) sum = 1e-10f;
  return proj / sum;
}

}  // namespace

TEST(RknnDecodeUtils, ResolveStrideSetP5) {
  std::vector<int> strides;
  const bool resolved = resolve_stride_set(640, 8400, strides);
  ASSERT_TRUE(resolved);
  ASSERT_EQ(strides.size(), 3u);
  EXPECT_EQ(strides[0], 8);
  EXPECT_EQ(strides[1], 16);
  EXPECT_EQ(strides[2], 32);

  AnchorLayout layout = build_anchor_layout(640, 8400, strides);
  EXPECT_TRUE(layout.valid);
  EXPECT_EQ(layout.stride_map.size(), 8400u);
  EXPECT_NEAR(layout.stride_map.front(), 8.0f, 1e-5f);
  EXPECT_NEAR(layout.stride_map.back(), 32.0f, 1e-5f);
  EXPECT_NEAR(layout.anchor_cx.front(), 4.0f, 1e-5f);
  EXPECT_NEAR(layout.anchor_cy.front(), 4.0f, 1e-5f);
}

TEST(RknnDecodeUtils, FallbackLayoutWhenUnresolved) {
  std::vector<int> strides;
  const bool resolved = resolve_stride_set(640, 1000, strides);
  EXPECT_FALSE(resolved);
  AnchorLayout layout = build_anchor_layout(640, 1000, {8, 16, 32});
  EXPECT_FALSE(layout.valid);
  EXPECT_EQ(layout.stride_map.size(), 1000u);
  EXPECT_GT(layout.anchor_cx.front(), 0.0f);
  EXPECT_GT(layout.anchor_cy.front(), 0.0f);
}

TEST(RknnDecodeOptimized, SingleDecodeMatchesScalarReference) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);

  for (int reg_max : {4, 8, 16, 32}) {
    std::vector<float> logits(reg_max);
    std::vector<float> probs(reg_max);
    for (int t = 0; t < 64; ++t) {
      for (int k = 0; k < reg_max; ++k) {
        logits[k] = dist(rng);
      }
      const float got = dfl_decode_neon_single(logits.data(), reg_max, probs.data());
      const float want = softmax_expected_value(logits.data(), reg_max);
      EXPECT_NEAR(got, want, 1e-4f) << "reg_max=" << reg_max;
    }
  }
}

TEST(RknnDecodeOptimized, Decode4SidesExtractMatchesScalarReference) {
  std::mt19937 rng(321);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);

  constexpr int reg_max = 16;
  constexpr int N = 7;
  std::vector<float> logits(4 * reg_max * N);
  for (float& v : logits) v = dist(rng);

  std::array<float, 32> probs{};

  for (int anchor_idx = 0; anchor_idx < N; ++anchor_idx) {
    const auto out = dfl_decode_4sides_optimized(logits.data(), anchor_idx, N, reg_max, probs.data());
    for (int side = 0; side < 4; ++side) {
      std::array<float, reg_max> d{};
      const int ch_base = side * reg_max;
      for (int k = 0; k < reg_max; ++k) {
        d[k] = logits[(ch_base + k) * N + anchor_idx];
      }
      const float want = softmax_expected_value(d.data(), reg_max);
      EXPECT_NEAR(out[side], want, 1e-4f) << "anchor_idx=" << anchor_idx << " side=" << side;
    }
  }
}
