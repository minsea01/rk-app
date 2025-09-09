#pragma once

#include "rkapp/capture/ISource.hpp"

namespace rkapp::capture {

class GigeSource : public ISource {
public:
  GigeSource() = default;
  ~GigeSource() override = default;

  bool open(const std::string& uri) override;
  bool read(cv::Mat& frame) override;
  void release() override;
  bool isOpened() const override;

  double getFPS() const override;
  cv::Size getSize() const override;
  int getTotalFrames() const override;
  int getCurrentFrame() const override;

  SourceType getType() const override { return SourceType::GIGE; }

private:
  bool opened_ = false;
  cv::Size size_{0,0};
  double fps_ = 30.0;
  int count_ = 0;
};

} // namespace rkapp::capture

