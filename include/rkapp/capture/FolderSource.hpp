#pragma once

#include <vector>
#include "rkapp/capture/ISource.hpp"

namespace rkapp::capture {

// 文件夹输入源：按文件名顺序逐张读取图片用于离线推理。
class FolderSource : public ISource {
public:
  FolderSource();
  ~FolderSource() override;

  bool open(const std::string& folder_path) override;
  ReadStatus readFrameEx(CaptureFrame& frame) override;
  // 便捷方法（非接口）：读取下一张可解码的图片，列表耗尽时返回 false。
  bool read(cv::Mat& frame);
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
