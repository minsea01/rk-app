#pragma once

#include "rkapp/capture/GstSourceBase.hpp"

#include <chrono>
#include <string>

namespace rkapp::capture {

class CsiSource : public GstSourceBase {
 public:
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

  static UriConfig parseUri(const std::string& uri);
  static std::string buildPipelineDescription(const UriConfig& config);

 private:
  static std::string sanitizeDevicePath(const std::string& value);
  static std::string sanitizeCapsToken(const std::string& value);
};

}  // namespace rkapp::capture
