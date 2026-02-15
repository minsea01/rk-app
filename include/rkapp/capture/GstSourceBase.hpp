#pragma once

#include "rkapp/capture/ISource.hpp"

#include <chrono>
#include <mutex>
#include <string>

#if defined(RKAPP_WITH_GIGE) || defined(RKAPP_WITH_CSI)
typedef struct _GstElement GstElement;
typedef struct _GstAppSink GstAppSink;
#endif

namespace rkapp::capture {

class GstSourceBase : public ISource {
 public:
  bool read(cv::Mat& frame) override;
  bool readFrame(CaptureFrame& frame) override;
  void release() override;
  bool isOpened() const override;

  double getFPS() const override;
  cv::Size getSize() const override;
  int getTotalFrames() const override;
  int getCurrentFrame() const override;

 protected:
  struct OpenConfig {
    std::string uri;
    std::string pipeline_desc;
    std::chrono::milliseconds pull_timeout{200};
    int max_consecutive_failures = 5;
    double fps = 30.0;
    std::string unknown_format_fallback;
    bool reconnect_immediately_on_failure = true;
  };

  explicit GstSourceBase(const char* source_name);
  bool openWithConfig(const OpenConfig& config);

  const std::string& sourceName() const { return source_name_; }

 private:
#if defined(RKAPP_WITH_GIGE) || defined(RKAPP_WITH_CSI)
  bool createPipelineLocked();
  void destroyPipelineLocked();
  bool ensurePipelineLocked();
  bool checkPipelineHealthLocked();
  bool attemptReconnectLocked();
  void handleReadFailureLocked();

  GstElement* pipeline_ = nullptr;
  GstElement* sink_element_ = nullptr;
  GstAppSink* appsink_ = nullptr;
  std::string source_uri_;
  std::string pipeline_desc_;
  std::chrono::steady_clock::time_point last_reconnect_{};
  std::chrono::milliseconds reconnect_backoff_{500};
  std::chrono::milliseconds pull_timeout_{200};
  int max_consecutive_failures_ = 5;
  int consecutive_failures_ = 0;
  std::string unknown_format_fallback_;
  bool reconnect_immediately_on_failure_ = true;
#endif

  std::string source_name_;
  bool opened_ = false;
  mutable std::mutex mtx_;
  cv::Size size_{0, 0};
  double fps_ = 30.0;
  int count_ = 0;
};

}  // namespace rkapp::capture
