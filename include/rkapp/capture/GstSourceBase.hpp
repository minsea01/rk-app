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

// GStreamer 输入源基类：封装 pipeline 生命周期、拉流与重连逻辑。
class GstSourceBase : public ISource {
 public:
  // 便捷方法（非接口）：读取一帧并转换为 BGR；多态调用方使用 readFrameEx。
  bool read(cv::Mat& frame);
  bool readFrame(CaptureFrame& frame);
  ReadStatus readFrameEx(CaptureFrame& frame) override;
  void release() override;
  bool isOpened() const override;

  double getFPS() const override;
  cv::Size getSize() const override;
  int getTotalFrames() const override;
  int getCurrentFrame() const override;

 protected:
  // 打开配置：由派生类按来源（GIGE/CSI）组装 pipeline 参数。
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
  // 以下函数要求在持有 mtx_ 的情况下调用。
  bool createPipelineLocked();
  void destroyPipelineLocked();
  bool ensurePipelineLocked();
  bool checkPipelineHealthLocked();
  bool attemptReconnectLocked();
  void handleReadFailureLocked();

  // GStreamer 对象与重连状态。
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

  // 通用采集统计状态。
  std::string source_name_;
  bool opened_ = false;
  mutable std::mutex mtx_;
  cv::Size size_{0, 0};
  double fps_ = 30.0;
  int count_ = 0;
};

}  // namespace rkapp::capture
