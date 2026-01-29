#pragma once

#include "rkapp/capture/ISource.hpp"
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#ifdef RKAPP_WITH_GIGE
typedef struct _GstElement GstElement;
typedef struct _GstAppSink GstAppSink;
#endif

namespace rkapp::capture {

class GigeSource : public ISource {
public:
  struct UriConfig {
    std::string camera_name = "Aravis-Fake-GV01";
    std::vector<std::string> caps_kv;
    std::string desired_format = "GRAY8";
    bool use_videoconvert = false;
    std::chrono::milliseconds pull_timeout{200};
    int max_consecutive_failures = 5;
  };

  GigeSource() = default;
  ~GigeSource() override;

  bool open(const std::string& uri) override;
  bool read(cv::Mat& frame) override;
  bool readFrame(CaptureFrame& frame) override;
  void release() override;
  bool isOpened() const override;

  double getFPS() const override;
  cv::Size getSize() const override;
  int getTotalFrames() const override;
  int getCurrentFrame() const override;

  SourceType getType() const override { return SourceType::GIGE; }

  static UriConfig parseUri(const std::string& uri);
  static std::string buildPipelineDescription(const UriConfig& config);

private:
  static std::string sanitizeCameraName(const std::string& name);
  static std::string sanitizeCaps(const std::string& caps);

  bool opened_ = false;
#ifdef RKAPP_WITH_GIGE
  void destroyPipeline();
  bool createPipeline();
  bool ensurePipelineLocked();
  bool checkPipelineHealthLocked();
  bool attemptReconnectLocked();
  void handleReadFailureLocked();

  GstElement* pipeline_ = nullptr;
  GstElement* sink_element_ = nullptr;
  GstAppSink* appsink_ = nullptr;
  UriConfig uri_config_;
  std::string pipeline_desc_;
  std::string camera_uri_;
  std::chrono::steady_clock::time_point last_reconnect_;
  std::chrono::milliseconds reconnect_backoff_{500};
#endif
  mutable std::mutex mtx_;
  cv::Size size_{0,0};
  double fps_ = 30.0;
  int count_ = 0;
#ifdef RKAPP_WITH_GIGE
  std::chrono::milliseconds pull_timeout_{200};
  int max_consecutive_failures_ = 5;
  int consecutive_failures_ = 0;
#endif
};

} // namespace rkapp::capture
