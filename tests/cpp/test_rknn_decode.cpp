#include "rkapp/infer/RknnDecodeUtils.hpp"

#include <gtest/gtest.h>

using rkapp::infer::AnchorLayout;
using rkapp::infer::build_anchor_layout;
using rkapp::infer::resolve_stride_set;

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

