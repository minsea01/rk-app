#pragma once

#include "rkapp/capture/GstSourceBase.hpp"

#include <chrono>
#include <string>

namespace rkapp::capture {

// MIPI CSI 相机输入源：通过 v4l2src + appsink 拉取图像。
class CsiSource : public GstSourceBase {
 public:
  // URI 配置键值解析结果（device/width/height/fps/format 等）。
  struct UriConfig {
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int framerate = 30;
    std::string format = "NV12";
    bool use_videoconvert = true;
    std::chrono::milliseconds pull_timeout{200};
    int max_consecutive_failures = 5;
  };

  CsiSource();
  ~CsiSource() override;

  bool open(const std::string& uri) override;
  SourceType getType() const override { return SourceType::CSI; }

  // 解析 URI 形如 "device=/dev/video0,width=1280,height=720,fps=30"。
  static UriConfig parseUri(const std::string& uri);
  // 根据配置拼装 v4l2src pipeline。
  static std::string buildPipelineDescription(const UriConfig& config);

 private:
  static std::string sanitizeDevicePath(const std::string& value);
  static std::string sanitizeCapsToken(const std::string& value);
};

}  // namespace rkapp::capture
