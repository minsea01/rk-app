// ONNXRuntime 推理引擎头文件（PImpl 以减少头文件依赖）。
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rkapp/infer/IInferEngine.hpp"
#include "rkapp/infer/ModelMeta.hpp"

namespace rkapp::infer {

class OnnxEngine : public IInferEngine {
public:
  OnnxEngine();
  ~OnnxEngine() override;

  bool init(const ModelSpec& model_spec) override;
  std::vector<Detection> infer(const cv::Mat& image) override;
  std::vector<Detection> inferPreprocessed(
      const cv::Mat& preprocessed_image,
      const cv::Size& original_size,
      const preprocess::LetterboxInfo& letterbox_info) override;
  void warmup() override;
  void release() override;

  void setDecodeParams(const DecodeParams& params) override;

  int getInputWidth() const override;
  int getInputHeight() const override;

  /// 设置推理 CUDA 设备号，需在 init() 前调用。
  void setCudaDeviceId(int device_id) { cuda_device_id_ = device_id; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string model_path_;
  int input_size_ = 640;
  bool is_initialized_ = false;
  DecodeParams decode_params_;
  bool unsupported_model_ = false;  // 遇到不支持的输出布局后会置位
  int cuda_device_id_ = 0;  // CUDA 设备索引（默认 0）
  // warmup() 内部会调用 infer()，因此使用可重入锁保护共享状态。
  mutable std::recursive_mutex engine_mtx_;
};

} // namespace rkapp::infer
