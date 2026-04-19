#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rkapp/capture/FrameOps.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

namespace {

bool matEquals(const cv::Mat& lhs, const cv::Mat& rhs) {
  if (lhs.empty() || rhs.empty()) {
    return lhs.empty() == rhs.empty();
  }
  if (lhs.size() != rhs.size() || lhs.type() != rhs.type()) {
    return false;
  }
  return cv::countNonZero(lhs.reshape(1) != rhs.reshape(1)) == 0;
}

TEST(PreprocessTest, LetterboxPreservesAspectAndReturnsInfo) {
  cv::Mat src(1080, 1920, CV_8UC3, cv::Scalar(10, 20, 30));
  rkapp::preprocess::LetterboxInfo info{};

  const cv::Mat out = rkapp::preprocess::Preprocess::letterbox(
      src, 640, info, rkapp::preprocess::AccelBackend::OPENCV);

  ASSERT_FALSE(out.empty());
  EXPECT_EQ(out.rows, 640);
  EXPECT_EQ(out.cols, 640);
  EXPECT_NEAR(info.scale, 640.0f / 1920.0f, 1e-4f);
  EXPECT_EQ(info.new_width, 640);
  EXPECT_EQ(info.new_height, 360);
  EXPECT_NEAR(info.dx, 0.0f, 1e-4f);
  EXPECT_NEAR(info.dy, 140.0f, 1e-4f);
}

TEST(PreprocessTest, LetterboxRejectsEmptyInput) {
  rkapp::preprocess::LetterboxInfo info{};
  const cv::Mat out = rkapp::preprocess::Preprocess::letterbox(
      cv::Mat(), 640, info, rkapp::preprocess::AccelBackend::OPENCV);

  EXPECT_TRUE(out.empty());
  EXPECT_EQ(info.scale, 0.0f);
}

TEST(PreprocessTest, NormalizeConvertsToFloatAndScalesValues) {
  cv::Mat src(1, 1, CV_8UC3, cv::Scalar(255, 127, 0));
  const cv::Mat norm = rkapp::preprocess::Preprocess::normalize(src, 1.0f / 255.0f);

  ASSERT_EQ(norm.type(), CV_32FC3);
  const cv::Vec3f px = norm.at<cv::Vec3f>(0, 0);
  EXPECT_NEAR(px[0], 1.0f, 1e-5f);
  EXPECT_NEAR(px[1], 127.0f / 255.0f, 1e-5f);
  EXPECT_NEAR(px[2], 0.0f, 1e-5f);
}

TEST(PreprocessTest, Hwc2ChwRejectsNonFloatInput) {
  cv::Mat src(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
  const cv::Mat chw = rkapp::preprocess::Preprocess::hwc2chw(src);
  EXPECT_TRUE(chw.empty());
}

TEST(PreprocessTest, Hwc2ChwConvertsThreeChannelFloatImage) {
  cv::Mat src(1, 2, CV_32FC3);
  src.at<cv::Vec3f>(0, 0) = cv::Vec3f(1.0f, 2.0f, 3.0f);
  src.at<cv::Vec3f>(0, 1) = cv::Vec3f(4.0f, 5.0f, 6.0f);

  const cv::Mat chw = rkapp::preprocess::Preprocess::hwc2chw(src);

  ASSERT_FALSE(chw.empty());
  ASSERT_EQ(chw.type(), CV_32F);
  ASSERT_EQ(chw.rows, 1);
  ASSERT_EQ(chw.cols, 6);

  const float* data = chw.ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 4.0f);
  EXPECT_FLOAT_EQ(data[2], 2.0f);
  EXPECT_FLOAT_EQ(data[3], 5.0f);
  EXPECT_FLOAT_EQ(data[4], 3.0f);
  EXPECT_FLOAT_EQ(data[5], 6.0f);
}

TEST(PreprocessTest, BlobReturnsExpectedFlatTensor) {
  cv::Mat src(2, 2, CV_8UC3, cv::Scalar(20, 40, 60));
  const cv::Mat blob = rkapp::preprocess::Preprocess::blob(src);

  ASSERT_FALSE(blob.empty());
  EXPECT_EQ(blob.rows, 1);
  EXPECT_EQ(blob.cols, 12);
  EXPECT_EQ(blob.type(), CV_32F);
}

TEST(PreprocessTest, ConvertYuv420spToBgrRejectsOddHeight) {
  cv::Mat src(481 + 481 / 2, 640, CV_8UC1, cv::Scalar(128));
  const cv::Mat out = rkapp::preprocess::Preprocess::convertYuv420spToBgr(
      src, cv::Size(640, 481), false, rkapp::preprocess::AccelBackend::OPENCV);
  EXPECT_TRUE(out.empty());
}

TEST(PreprocessTest, ConvertYuv420spToBgrSupportsPaddedCpuPath) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 4;
  constexpr int kRows = kHeight + kHeight / 2;

  cv::Mat tight(kRows, kWidth, CV_8UC1);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      tight.at<uint8_t>(y, x) = static_cast<uint8_t>(32 + y * 16 + x * 8);
    }
  }
  for (int x = 0; x < kWidth; x += 2) {
    tight.at<uint8_t>(kHeight, x) = 180;
    tight.at<uint8_t>(kHeight, x + 1) = 90;
    tight.at<uint8_t>(kHeight + 1, x) = 120;
    tight.at<uint8_t>(kHeight + 1, x + 1) = 150;
  }

  constexpr size_t kStride = 8;
  std::vector<uint8_t> padded_storage(static_cast<size_t>(kRows) * kStride, 0);
  cv::Mat padded(kRows, kWidth, CV_8UC1, padded_storage.data(), kStride);
  for (int y = 0; y < kRows; ++y) {
    std::memcpy(padded.ptr<uint8_t>(y), tight.ptr<uint8_t>(y), kWidth);
  }

  const cv::Mat tight_bgr = rkapp::preprocess::Preprocess::convertYuv420spToBgr(
      tight, cv::Size(kWidth, kHeight), false, rkapp::preprocess::AccelBackend::OPENCV);
  const cv::Mat padded_bgr = rkapp::preprocess::Preprocess::convertYuv420spToBgr(
      padded, cv::Size(kWidth, kHeight), false, rkapp::preprocess::AccelBackend::OPENCV);

  ASSERT_FALSE(tight_bgr.empty());
  ASSERT_FALSE(padded_bgr.empty());
  EXPECT_TRUE(matEquals(tight_bgr, padded_bgr));
}

TEST(PreprocessTest, ConvertYuv420spToBgrDistinguishesNv12AndNv21) {
  cv::Mat yuv(3, 2, CV_8UC1, cv::Scalar(128));
  yuv.at<uint8_t>(2, 0) = 255;
  yuv.at<uint8_t>(2, 1) = 0;

  const cv::Mat out_nv12 = rkapp::preprocess::Preprocess::convertYuv420spToBgr(
      yuv, cv::Size(2, 2), false, rkapp::preprocess::AccelBackend::OPENCV);
  const cv::Mat out_nv21 = rkapp::preprocess::Preprocess::convertYuv420spToBgr(
      yuv, cv::Size(2, 2), true, rkapp::preprocess::AccelBackend::OPENCV);

  cv::Mat expected_nv12;
  cv::Mat expected_nv21;
  cv::cvtColor(yuv, expected_nv12, cv::COLOR_YUV2BGR_NV12);
  cv::cvtColor(yuv, expected_nv21, cv::COLOR_YUV2BGR_NV21);

  ASSERT_FALSE(out_nv12.empty());
  ASSERT_FALSE(out_nv21.empty());
  EXPECT_TRUE(matEquals(out_nv12, expected_nv12));
  EXPECT_TRUE(matEquals(out_nv21, expected_nv21));
  EXPECT_FALSE(matEquals(out_nv12, out_nv21));
}

TEST(PreprocessTest, CaptureConvertToBgrRejectsNv12WithoutImageSize) {
  rkapp::capture::CaptureFrame frame;
  frame.setMatFrame(cv::Mat(6, 4, CV_8UC1, cv::Scalar(0)), rkapp::capture::PixelFormat::NV12);
  frame.image_size = cv::Size();

  const auto converted = rkapp::capture::convertToBgr(
      frame, rkapp::preprocess::AccelBackend::OPENCV);
  EXPECT_TRUE(converted.empty());
}

TEST(PreprocessTest, DetachFrameIfAliasedOnlyClonesSharedBacking) {
  cv::Mat mapped(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
  rkapp::capture::CaptureFrame shared_frame;
  shared_frame.setMatFrame(mapped, rkapp::capture::PixelFormat::BGR888,
                           rkapp::capture::StorageKind::SHARED_MAPPED,
                           std::make_shared<int>(42));

  const auto shared_view = rkapp::capture::convertToBgr(shared_frame);
  ASSERT_FALSE(shared_view.empty());
  ASSERT_TRUE(shared_view.aliases_source);
  const cv::Mat detached = rkapp::capture::detachFrameIfAliased(
      shared_view.image, shared_frame, shared_view.aliases_source);
  EXPECT_NE(detached.data, shared_frame.mat.data);

  rkapp::capture::CaptureFrame cpu_frame;
  cpu_frame.setMatFrame(mapped.clone(), rkapp::capture::PixelFormat::BGR888);
  const auto cpu_view = rkapp::capture::convertToBgr(cpu_frame);
  ASSERT_FALSE(cpu_view.empty());
  ASSERT_TRUE(cpu_view.aliases_source);
  const cv::Mat passthrough = rkapp::capture::detachFrameIfAliased(
      cpu_view.image, cpu_frame, cpu_view.aliases_source);
  EXPECT_EQ(passthrough.data, cpu_frame.mat.data);
}

}  // namespace
