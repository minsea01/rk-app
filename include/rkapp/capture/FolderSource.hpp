#pragma once

#include <vector>
#include "rkapp/capture/ISource.hpp"

namespace rkapp::capture {

class FolderSource : public ISource {
public:
  FolderSource();
  ~FolderSource() override;

  bool open(const std::string& folder_path) override;
  bool read(cv::Mat& frame) override;
  void release() override;
  bool isOpened() const override;

  double getFPS() const override;
  cv::Size getSize() const override;
  int getTotalFrames() const override;
  int getCurrentFrame() const override;
  SourceType getType() const override;

private:
  std::string folder_path_;
  std::vector<std::string> image_files_;
  size_t current_index_ = 0;
  bool is_opened_ = false;
  cv::Size cached_size_{0, 0};  // 缓存图像尺寸，避免重复读取
};

} // namespace rkapp::capture
