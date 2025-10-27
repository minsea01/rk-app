#include <algorithm>
#include <gtest/gtest.h>
#include "rkapp/capture/GigeSource.hpp"

using rkapp::capture::GigeSource;

namespace {

bool containsCaps(const std::vector<std::string>& caps, const std::string& value) {
    return std::find(caps.begin(), caps.end(), value) != caps.end();
}

}  // namespace

TEST(GigeSourceTest, ParseDefaults) {
    auto cfg = GigeSource::parseUri("");
    EXPECT_EQ(cfg.camera_name, "Aravis-Fake-GV01");
    EXPECT_FALSE(cfg.use_videoconvert);
    EXPECT_EQ(cfg.desired_format, "GRAY8");
    EXPECT_EQ(cfg.pull_timeout.count(), 200);
    EXPECT_EQ(cfg.max_consecutive_failures, 5);
    EXPECT_TRUE(containsCaps(cfg.caps_kv, "format=GRAY8"));

    const auto pipeline = GigeSource::buildPipelineDescription(cfg);
    EXPECT_NE(pipeline.find("format=GRAY8"), std::string::npos);
    EXPECT_EQ(pipeline.find("videoconvert"), std::string::npos);
}

TEST(GigeSourceTest, ParseColorPipeline) {
    auto cfg = GigeSource::parseUri(
        "camera-name=Cam One,width=1920,height=1080,color=true,pull-timeout-ms=450,max-failures=3");
    EXPECT_EQ(cfg.camera_name, "Cam One");
    EXPECT_TRUE(cfg.use_videoconvert);
    EXPECT_EQ(cfg.desired_format, "BGR");
    EXPECT_EQ(cfg.pull_timeout.count(), 450);
    EXPECT_EQ(cfg.max_consecutive_failures, 3);
    EXPECT_TRUE(containsCaps(cfg.caps_kv, "width=1920"));
    EXPECT_TRUE(containsCaps(cfg.caps_kv, "height=1080"));
    EXPECT_TRUE(containsCaps(cfg.caps_kv, "format=BGR"));

    const auto pipeline = GigeSource::buildPipelineDescription(cfg);
    EXPECT_NE(pipeline.find("videoconvert"), std::string::npos);
    EXPECT_NE(pipeline.find("format=BGR"), std::string::npos);
    EXPECT_NE(pipeline.find("camera-name=\"Cam One\""), std::string::npos);
}

TEST(GigeSourceTest, SanitizesCameraName) {
    auto cfg = GigeSource::parseUri("camera-name=\";rm -rf;,format=GRAY8");
    const auto pipeline = GigeSource::buildPipelineDescription(cfg);
    EXPECT_EQ(cfg.desired_format, "GRAY8");
    EXPECT_NE(pipeline.find("rm -rf"), std::string::npos);
    EXPECT_EQ(pipeline.find(';'), std::string::npos);
}
