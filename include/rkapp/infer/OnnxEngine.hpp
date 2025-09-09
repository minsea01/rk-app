// Minimal header for ONNXRuntime engine (pimpl to avoid ORT headers in consumers)
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rkapp/infer/IInferEngine.hpp"

namespace rkapp::infer {

class OnnxEngine : public IInferEngine {
public:
  OnnxEngine();
  ~OnnxEngine() override;

  bool init(const std::string& model_path, int img_size = 640) override;
  std::vector<Detection> infer(const cv::Mat& image) override;
  void warmup() override;
  void release() override;

  int getInputWidth() const override;
  int getInputHeight() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string model_path_;
  int input_size_ = 640;
  bool is_initialized_ = false;
};

} // namespace rkapp::infer

