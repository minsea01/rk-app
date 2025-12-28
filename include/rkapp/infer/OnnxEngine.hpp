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

  void setDecodeParams(const DecodeParams& params) override;

  int getInputWidth() const override;
  int getInputHeight() const override;

  /// Set CUDA device ID for inference. Must be called before init().
  /// @param device_id CUDA device index (0 = first GPU, 1 = second, etc.)
  void setCudaDeviceId(int device_id) { cuda_device_id_ = device_id; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string model_path_;
  int input_size_ = 640;
  bool is_initialized_ = false;
  DecodeParams decode_params_;
  bool unsupported_model_ = false;  // set when encountering unsupported output layout
  int cuda_device_id_ = 0;  // CUDA device index (configurable, default: 0)
};

} // namespace rkapp::infer
