// Minimal header for RKNN engine (pimpl to avoid RKNN headers in consumers)
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rkapp/infer/IInferEngine.hpp"

// Forward declaration for zero-copy input
namespace rkapp::common {
class DmaBuf;
}

namespace rkapp::infer {

class RknnEngine : public IInferEngine {
public:
  RknnEngine();
  ~RknnEngine() override;

  bool init(const std::string& model_path, int img_size = 640) override;
  std::vector<Detection> infer(const cv::Mat& image) override;
  void warmup() override;
  void release() override;

  /**
   * @brief Zero-copy inference from DMA-BUF backed memory
   *
   * This method avoids CPU memory copies by directly importing
   * the DMA-BUF fd into RKNN for NPU access.
   *
   * Supported input formats:
   * - RGB888 (3-channel, already letterboxed to input_size_)
   * - NV12/NV21 (will use RGA for hardware color conversion)
   *
   * @param input DMA-BUF containing preprocessed image
   * @param letterbox_info Letterbox parameters for coordinate mapping
   * @return Detection results
   */
  std::vector<Detection> inferDmaBuf(
      rkapp::common::DmaBuf& input,
      const struct rkapp::preprocess::LetterboxInfo& letterbox_info);

  // 可选：设置NPU核心掩码（例如：1<<0, 1<<1, 1<<2）。若运行库不支持，将被忽略。
  void setCoreMask(uint32_t core_mask) { core_mask_ = core_mask; }
  void setDecodeParams(const DecodeParams& params) override;

  // 添加类别数访问接口
  int num_classes() const { return num_classes_; }
  bool has_objness() const { return has_objness_; }
  const std::vector<std::string>& class_names() const { return class_names_; }

  int getInputWidth() const override;
  int getInputHeight() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string model_path_;
  int input_size_ = 640;
  bool is_initialized_ = false;
  uint32_t core_mask_ = 0x7; // 默认三核并行；可被 setCoreMask 覆盖
  
  // Auto-inferred detection parameters
  int num_classes_ = -1;
  bool has_objness_ = true;  // Most YOLO exports include objectness score
  std::vector<std::string> class_names_;
  DecodeParams decode_params_;

  struct ModelMeta {
    int reg_max = -1;
    std::vector<int> strides;
    std::string head; // "dfl" / "raw" / ""
  } model_meta_;
};

} // namespace rkapp::infer
