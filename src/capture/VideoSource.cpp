#include "rkapp/capture/VideoSource.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace rkapp::capture {

VideoSource::VideoSource() = default;
VideoSource::~VideoSource() { release(); }

bool VideoSource::open(const std::string& video_path) {
  video_path_ = video_path;

  // Handle RTSP URLs and video files
  bool result = cap_.open(video_path_);

  if (!result) {
    std::cerr << "Failed to open video source: " << video_path_ << std::endl;
    return false;
  }

  // Reduce capture buffering for streaming sources to minimize latency/jitter.
  // Note: not all backends honor this setting (FFmpeg/GStreamer may ignore it).
  if (isStreamSource()) {
    (void)cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
  }

  // Get video properties
  fps_ = cap_.get(cv::CAP_PROP_FPS);
  total_frames_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
  width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
  height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

  // For RTSP streams, frame count might be -1
  if (total_frames_ < 0) {
    total_frames_ = INT_MAX;
  }

  std::cout << "Opened video source: " << video_path_ << std::endl;
  std::cout << "Resolution: " << width_ << "x" << height_ << std::endl;
  std::cout << "FPS: " << fps_ << ", Frames: " << total_frames_ << std::endl;

  return true;
}

bool VideoSource::read(cv::Mat& frame) {
  if (!cap_.isOpened()) {
    // 尝试重连流媒体
    if (isStreamSource()) {
      return tryReconnect() && cap_.read(frame);
    }
    return false;
  }

  bool result = cap_.read(frame);

  if (!result && isStreamSource()) {
    // 流媒体断开，尝试重连
    std::cerr << "VideoSource: stream read failed, attempting reconnect..." << std::endl;
    return tryReconnect() && cap_.read(frame);
  }

  if (result) {
    current_frame_++;
    reconnect_attempts_ = 0;  // 成功读取后重置重连计数
  }

  return result;
}

bool VideoSource::tryReconnect() {
  if (reconnect_attempts_ >= kMaxReconnectAttempts) {
    std::cerr << "VideoSource: max reconnect attempts (" << kMaxReconnectAttempts
              << ") reached, giving up" << std::endl;
    return false;
  }

  // 指数退避延迟: 500ms, 1s, 2s, 4s, 8s
  int delay_ms = kInitialReconnectDelayMs * (1 << reconnect_attempts_);
  reconnect_attempts_++;

  std::cerr << "VideoSource: reconnect attempt " << reconnect_attempts_
            << "/" << kMaxReconnectAttempts
            << ", waiting " << delay_ms << "ms..." << std::endl;

  std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

  cap_.release();

  bool success = cap_.open(video_path_);
  if (success) {
    if (isStreamSource()) {
      (void)cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    std::cout << "VideoSource: reconnected successfully to " << video_path_ << std::endl;
    reconnect_attempts_ = 0;
  }

  return success;
}

bool VideoSource::isStreamSource() const {
  return video_path_.find("rtsp://") == 0 ||
         video_path_.find("rtmp://") == 0 ||
         video_path_.find("http://") == 0 ||
         video_path_.find("https://") == 0;
}

void VideoSource::release() {
  if (cap_.isOpened()) {
    cap_.release();
  }
  current_frame_ = 0;
}

bool VideoSource::isOpened() const { return cap_.isOpened(); }
double VideoSource::getFPS() const { return fps_; }
cv::Size VideoSource::getSize() const { return cv::Size(width_, height_); }
int VideoSource::getTotalFrames() const { return total_frames_; }
int VideoSource::getCurrentFrame() const { return current_frame_; }

SourceType VideoSource::getType() const {
  // Simple heuristic to distinguish RTSP from video file
  if (video_path_.find("rtsp://") == 0 || video_path_.find("rtmp://") == 0) {
    return SourceType::RTSP;
  }
  return SourceType::VIDEO;
}

} // namespace rkapp::capture
