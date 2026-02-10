#include <gtest/gtest.h>

#include "rkapp/capture/CsiSource.hpp"

using rkapp::capture::CsiSource;

TEST(CsiSourceTest, ParseDefaultUri) {
  const auto cfg = CsiSource::parseUri("");
  EXPECT_EQ(cfg.device, "/dev/video0");
  EXPECT_EQ(cfg.width, 640);
  EXPECT_EQ(cfg.height, 480);
  EXPECT_EQ(cfg.framerate, 30);
  EXPECT_EQ(cfg.format, "NV12");
  EXPECT_TRUE(cfg.use_videoconvert);
  EXPECT_EQ(cfg.pull_timeout.count(), 200);
  EXPECT_EQ(cfg.max_consecutive_failures, 5);
}

TEST(CsiSourceTest, ParseFullUri) {
  const auto cfg = CsiSource::parseUri(
      "device=/dev/video2,width=1920,height=1080,framerate=60,format=NV12,pull-timeout-ms=350,"
      "max-failures=9");
  EXPECT_EQ(cfg.device, "/dev/video2");
  EXPECT_EQ(cfg.width, 1920);
  EXPECT_EQ(cfg.height, 1080);
  EXPECT_EQ(cfg.framerate, 60);
  EXPECT_EQ(cfg.format, "NV12");
  EXPECT_TRUE(cfg.use_videoconvert);
  EXPECT_EQ(cfg.pull_timeout.count(), 350);
  EXPECT_EQ(cfg.max_consecutive_failures, 9);
}

TEST(CsiSourceTest, ParseDevicePath) {
  const auto cfg = CsiSource::parseUri("device=/dev/video11");
  EXPECT_EQ(cfg.device, "/dev/video11");
}

TEST(CsiSourceTest, ParseFormatNv12UsesVideoConvert) {
  const auto cfg = CsiSource::parseUri("format=NV12");
  EXPECT_EQ(cfg.format, "NV12");
  EXPECT_TRUE(cfg.use_videoconvert);
}

TEST(CsiSourceTest, ParseFormatBgrSkipsVideoConvert) {
  const auto cfg = CsiSource::parseUri("format=BGR");
  EXPECT_EQ(cfg.format, "BGR");
  EXPECT_FALSE(cfg.use_videoconvert);
}

TEST(CsiSourceTest, ParseBayerFormatSkipsVideoConvert) {
  const auto cfg = CsiSource::parseUri("format=BayerRG8");
  EXPECT_EQ(cfg.format, "rggb");
  EXPECT_FALSE(cfg.use_videoconvert);
  const auto pipeline = CsiSource::buildPipelineDescription(cfg);
  EXPECT_NE(pipeline.find("video/x-bayer"), std::string::npos);
  EXPECT_EQ(pipeline.find("videoconvert"), std::string::npos);
}

TEST(CsiSourceTest, BuildPipelineContainsV4l2src) {
  const auto cfg = CsiSource::parseUri("device=/dev/video0,format=NV12");
  const auto pipeline = CsiSource::buildPipelineDescription(cfg);
  EXPECT_NE(pipeline.find("v4l2src device=/dev/video0"), std::string::npos);
  EXPECT_NE(pipeline.find("appsink name=sink"), std::string::npos);
}

TEST(CsiSourceTest, BuildPipelineContainsResolutionAndFps) {
  const auto cfg = CsiSource::parseUri("width=1280,height=720,framerate=50,format=BGR");
  const auto pipeline = CsiSource::buildPipelineDescription(cfg);
  EXPECT_NE(pipeline.find("width=1280"), std::string::npos);
  EXPECT_NE(pipeline.find("height=720"), std::string::npos);
  EXPECT_NE(pipeline.find("framerate=50/1"), std::string::npos);
}
