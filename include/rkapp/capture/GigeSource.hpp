#pragma once

#include "rkapp/capture/GstSourceBase.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace rkapp::capture {

// 基于 Aravis + GStreamer 的千兆网相机输入源。
class GigeSource : public GstSourceBase {
 public:
  // URI 配置键值解析结果（如 camera-name/format/pull-timeout-ms）。
  struct UriConfig {
    std::string camera_name = "Aravis-Fake-GV01";
    std::vector<std::string> caps_kv;
    std::string desired_format = "GRAY8";
    bool use_videoconvert = false;
    std::chrono::milliseconds pull_timeout{200};
    int max_consecutive_failures = 5;
  };

  GigeSource();
  ~GigeSource() override;

  bool open(const std::string& uri) override;
  SourceType getType() const override { return SourceType::GIGE; }

  // 解析 URI 形如 "camera-name=...,format=...,pull-timeout-ms=..."。
  static UriConfig parseUri(const std::string& uri);
  // 根据解析配置拼装 GStreamer pipeline 描述串。
  static std::string buildPipelineDescription(const UriConfig& config);

 private:
  static std::string sanitizeCameraName(const std::string& name);
  static std::string sanitizeCaps(const std::string& caps);
};

}  // namespace rkapp::capture
