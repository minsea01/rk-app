#pragma once

#include "rkapp/capture/ISource.hpp"

namespace rkapp::capture {

// 通用视频输入源：支持本地文件与 RTSP/RTMP/HTTP 流。
class VideoSource : public ISource {
public:
  VideoSource();
  ~VideoSource() override;

  bool open(const std::string& video_path) override;
  // 便捷方法（非接口）：读取一帧 BGR 图像；多态调用方使用 readFrameEx。
  bool read(cv::Mat& frame);
  ReadStatus readFrameEx(CaptureFrame& frame) override;
  void release() override;
  bool isOpened() const override;

  double getFPS() const override;
  cv::Size getSize() const override;
  int getTotalFrames() const override;
  int getCurrentFrame() const override;
  SourceType getType() const override;

private:
  bool tryReconnect();
  bool isStreamSource() const;

  cv::VideoCapture cap_;
  std::string video_path_;
  double fps_ = 30.0;
  int total_frames_ = 0;
  int current_frame_ = 0;
  int width_ = 0;
  int height_ = 0;

  // 重连机制参数
  int reconnect_attempts_ = 0;
  static constexpr int kMaxReconnectAttempts = 5;
  static constexpr int kInitialReconnectDelayMs = 500;
};

} // namespace rkapp::capture
