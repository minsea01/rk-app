#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "rkapp/preprocess/Preprocess.hpp"

namespace {

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

}  // namespace
