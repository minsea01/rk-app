#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "rkapp/common/DmaBuf.hpp"

namespace {

using rkapp::common::DmaBuf;

TEST(DmaBufTest, Nv12CopyRoundTripPreservesData) {
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;

  DmaBuf buf;
  ASSERT_TRUE(buf.allocate(kWidth, kHeight, DmaBuf::PixelFormat::NV12, DmaBuf::MemType::SYSTEM));

  cv::Mat src(kHeight + kHeight / 2, kWidth, CV_8UC1);
  for (int r = 0; r < src.rows; ++r) {
    for (int c = 0; c < src.cols; ++c) {
      src.at<uint8_t>(r, c) = static_cast<uint8_t>((r * 13 + c * 7) & 0xFF);
    }
  }

  ASSERT_TRUE(buf.copyFrom(src));

  cv::Mat dst;
  ASSERT_TRUE(buf.copyTo(dst));
  ASSERT_EQ(dst.rows, src.rows);
  ASSERT_EQ(dst.cols, src.cols);
  ASSERT_EQ(dst.type(), src.type());

  const int diff = cv::countNonZero(src != dst);
  EXPECT_EQ(diff, 0);
}

TEST(DmaBufTest, Nv12CopyFromRejectsInvalidLayout) {
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;

  DmaBuf buf;
  ASSERT_TRUE(buf.allocate(kWidth, kHeight, DmaBuf::PixelFormat::NV12, DmaBuf::MemType::SYSTEM));

  cv::Mat invalid_rgb(kHeight, kWidth, CV_8UC3, cv::Scalar(0, 0, 0));
  EXPECT_FALSE(buf.copyFrom(invalid_rgb));
}

TEST(DmaBufTest, AsMatReturnsFullNv12View) {
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;

  DmaBuf buf;
  ASSERT_TRUE(buf.allocate(kWidth, kHeight, DmaBuf::PixelFormat::NV12, DmaBuf::MemType::SYSTEM));

  cv::Mat view = buf.asMat();
  ASSERT_FALSE(view.empty());
  EXPECT_EQ(view.rows, kHeight + kHeight / 2);
  EXPECT_EQ(view.cols, kWidth);
  EXPECT_EQ(view.type(), CV_8UC1);
}

}  // namespace

