#include <gtest/gtest.h>

#include <algorithm>
#include <array>

#include <opencv2/opencv.hpp>

#include "rkapp/preprocess/Preprocess.hpp"

namespace {

double channelVariance(const cv::Mat& image, int channel_idx) {
  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(channels.at(channel_idx), mean, stddev);
  return stddev[0] * stddev[0];
}

std::array<double, 3> channelMeans(const cv::Mat& image) {
  const cv::Scalar mean = cv::mean(image);
  return {mean[0], mean[1], mean[2]};
}

double meanGap(const std::array<double, 3>& means) {
  const auto [min_it, max_it] = std::minmax_element(means.begin(), means.end());
  return *max_it - *min_it;
}

}  // namespace

TEST(PreprocessEnhancementTest, ResolveRoiRectNormalizedClamp) {
  cv::Rect roi;
  const bool ok = rkapp::preprocess::Preprocess::resolveRoiRect(
      cv::Size(200, 100), true, cv::Rect2f(0.1f, 0.2f, 0.5f, 0.5f), cv::Rect(),
      true, 8, roi);
  ASSERT_TRUE(ok);
  EXPECT_EQ(roi.x, 20);
  EXPECT_EQ(roi.y, 20);
  EXPECT_EQ(roi.width, 100);
  EXPECT_EQ(roi.height, 50);
}

TEST(PreprocessEnhancementTest, ResolveRoiRectPixelNoClampRejectsOutOfBounds) {
  cv::Rect roi;
  const bool ok = rkapp::preprocess::Preprocess::resolveRoiRect(
      cv::Size(128, 128), false, cv::Rect2f(), cv::Rect(-10, 0, 64, 64), false,
      8, roi);
  EXPECT_FALSE(ok);
}

TEST(PreprocessEnhancementTest, ResolveRoiRectPixelValidRectSucceeds) {
  cv::Rect roi;
  const bool ok = rkapp::preprocess::Preprocess::resolveRoiRect(
      cv::Size(192, 108), false, cv::Rect2f(), cv::Rect(16, 12, 96, 48), false,
      8, roi);
  ASSERT_TRUE(ok);
  EXPECT_EQ(roi.x, 16);
  EXPECT_EQ(roi.y, 12);
  EXPECT_EQ(roi.width, 96);
  EXPECT_EQ(roi.height, 48);
}

TEST(PreprocessEnhancementTest, CropRoiFullFrameFastPathReturnsOriginalView) {
  cv::Mat src(32, 48, CV_8UC3, cv::Scalar(10, 20, 30));
  const cv::Mat cropped = rkapp::preprocess::Preprocess::cropRoi(
      src, cv::Rect(0, 0, src.cols, src.rows));
  ASSERT_FALSE(cropped.empty());
  EXPECT_EQ(cropped.size(), src.size());
  EXPECT_EQ(cropped.type(), src.type());
  EXPECT_EQ(cropped.data, src.data);
}

TEST(PreprocessEnhancementTest, CropRoiSubRectReturnsClonedRegion) {
  cv::Mat src(40, 60, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::rectangle(src, cv::Rect(10, 8, 20, 12), cv::Scalar(20, 120, 240), cv::FILLED);
  const cv::Rect roi(10, 8, 20, 12);
  const cv::Mat cropped = rkapp::preprocess::Preprocess::cropRoi(src, roi);
  ASSERT_FALSE(cropped.empty());
  EXPECT_EQ(cropped.rows, roi.height);
  EXPECT_EQ(cropped.cols, roi.width);
  EXPECT_NE(cropped.data, src.data);
  EXPECT_EQ(cropped.at<cv::Vec3b>(0, 0), cv::Vec3b(20, 120, 240));
}

TEST(PreprocessEnhancementTest, GammaIdentityIsStable) {
  cv::Mat src(32, 32, CV_8UC3);
  cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

  const cv::Mat gamma_identity =
      rkapp::preprocess::Preprocess::applyGammaLut(src, 1.0f);
  ASSERT_FALSE(gamma_identity.empty());
  EXPECT_EQ(cv::countNonZero(src.reshape(1) != gamma_identity.reshape(1)), 0);
}

TEST(PreprocessEnhancementTest, GammaAdjustsMidTone) {
  cv::Mat src(1, 1, CV_8UC3, cv::Scalar(64, 64, 64));
  const cv::Mat gamma_corrected =
      rkapp::preprocess::Preprocess::applyGammaLut(src, 0.5f);
  ASSERT_FALSE(gamma_corrected.empty());
  EXPECT_GT(gamma_corrected.at<cv::Vec3b>(0, 0)[0], src.at<cv::Vec3b>(0, 0)[0]);
}

TEST(PreprocessEnhancementTest, GammaRejectsNonPositiveInput) {
  cv::Mat src(8, 8, CV_8UC3, cv::Scalar(64, 64, 64));
  EXPECT_TRUE(rkapp::preprocess::Preprocess::applyGammaLut(src, -0.5f).empty());
  EXPECT_TRUE(rkapp::preprocess::Preprocess::applyGammaLut(src, 0.0f).empty());
}

TEST(PreprocessEnhancementTest, GrayWorldWhiteBalanceReducesChannelGap) {
  cv::Mat src(96, 96, CV_8UC3, cv::Scalar(220, 120, 80));
  cv::Mat gradient(src.size(), src.type());
  for (int y = 0; y < gradient.rows; ++y) {
    for (int x = 0; x < gradient.cols; ++x) {
      gradient.at<cv::Vec3b>(y, x) = cv::Vec3b(
          static_cast<uint8_t>(x % 32), static_cast<uint8_t>(y % 32),
          static_cast<uint8_t>((x + y) % 32));
    }
  }
  cv::add(src, gradient, src);

  const auto before_means = channelMeans(src);
  const cv::Mat balanced =
      rkapp::preprocess::Preprocess::whiteBalanceGrayWorld(src, 0.0f);
  ASSERT_FALSE(balanced.empty());
  const auto after_means = channelMeans(balanced);

  EXPECT_LT(meanGap(after_means), meanGap(before_means));
}

TEST(PreprocessEnhancementTest, GrayWorldWhiteBalanceWithClipPercentWorks) {
  cv::Mat src(96, 96, CV_8UC3, cv::Scalar(225, 130, 70));
  cv::Mat gradient(src.size(), src.type());
  for (int y = 0; y < gradient.rows; ++y) {
    for (int x = 0; x < gradient.cols; ++x) {
      gradient.at<cv::Vec3b>(y, x) = cv::Vec3b(
          static_cast<uint8_t>((x * 3) % 64), static_cast<uint8_t>((y * 2) % 64),
          static_cast<uint8_t>((x + y) % 64));
    }
  }
  cv::add(src, gradient, src);
  src.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 255, 255);
  src.at<cv::Vec3b>(1, 1) = cv::Vec3b(0, 0, 0);

  const auto before_means = channelMeans(src);
  const cv::Mat balanced =
      rkapp::preprocess::Preprocess::whiteBalanceGrayWorld(src, 8.0f);
  ASSERT_FALSE(balanced.empty());
  const auto after_means = channelMeans(balanced);
  EXPECT_LT(meanGap(after_means), meanGap(before_means));
}

TEST(PreprocessEnhancementTest, BilateralDenoiseReducesVariance) {
  cv::Mat clean(128, 128, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat noise(clean.size(), clean.type());
  cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(25));

  cv::Mat clean16;
  cv::Mat noise16;
  clean.convertTo(clean16, CV_16SC3);
  noise.convertTo(noise16, CV_16SC3);
  cv::Mat noisy16 = clean16 + noise16;
  cv::Mat noisy;
  noisy16.convertTo(noisy, CV_8UC3);

  const double var_before = channelVariance(noisy, 0);
  const cv::Mat denoised =
      rkapp::preprocess::Preprocess::denoiseBilateral(noisy, 5, 35.0, 35.0);
  ASSERT_FALSE(denoised.empty());
  const double var_after = channelVariance(denoised, 0);

  EXPECT_LT(var_after, var_before);
}
