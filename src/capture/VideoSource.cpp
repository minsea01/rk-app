#include "rkapp/capture/VideoSource.hpp"
#include <iostream>

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
    return false;
  }

  bool result = cap_.read(frame);
  if (result) {
    current_frame_++;
  }

  return result;
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
