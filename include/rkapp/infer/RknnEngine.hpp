// Minimal header for RKNN engine (pimpl to avoid RKNN headers in consumers)
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rkapp/infer/IInferEngine.hpp"

// Forward declaration for zero-copy input
namespace rkapp::common {
class DmaBuf;
}

namespace rkapp::preprocess {
struct LetterboxInfo;
}

namespace rkapp::infer {

// Model metadata for inference optimization
struct ModelMeta {
  int reg_max = -1;
  std::vector<int> strides;
  std::string head; // "dfl" / "raw" / ""
  int output_index = -1;  // Optional override for multi-output models
  int num_classes = -1;   // Optional: explicit class count
  int has_objectness = -1;  // Optional: -1 unknown, 0 false, 1 true
};

class RknnEngine : public IInferEngine {
public:
  RknnEngine();
  ~RknnEngine() override;

  bool init(const std::string& model_path, int img_size = 640) override;
  std::vector<Detection> infer(const cv::Mat& image) override;

  /**
   * @brief Inference on already-preprocessed (letterboxed) input
   *
   * Use this when the caller has already applied letterbox preprocessing.
   * Avoids redundant letterbox in the engine.
   *
   * @param preprocessed_image Letterboxed image (size: input_size_ x input_size_)
   * @param original_size Original image size before letterbox
   * @param letterbox_info Letterbox parameters for coordinate reverse mapping
   * @return Detection results with coordinates in original image space
   */
  std::vector<Detection> inferPreprocessed(
      const cv::Mat& preprocessed_image,
      const cv::Size& original_size,
      const struct rkapp::preprocess::LetterboxInfo& letterbox_info);

  void warmup() override;
  void release() override;

  /**
   * @brief Zero-copy inference from DMA-BUF backed memory
   *
   * This method avoids CPU memory copies by directly importing
   * the DMA-BUF fd into RKNN for NPU access.
   * Requires ENABLE_RKNN_DMA_FD to be enabled at build time.
   * Optional: ENABLE_RKNN_IO_MEM enables rknn_set_io_mem (RKNN SDK >= 1.5.0).
   *
   * Supported input formats:
   * - RGB888 (3-channel, already letterboxed to input_size_) for direct DMA-FD path
   * - BGR888/NV12/NV21/RGBA/BGRA via fallback copy + color conversion path
   *
   * @param input DMA-BUF containing preprocessed image
   * @param original_size Original image size before letterbox
   * @param letterbox_info Letterbox parameters for coordinate mapping
   * @return Detection results with coordinates in original image space
   */
  std::vector<Detection> inferDmaBuf(
      rkapp::common::DmaBuf& input,
      const cv::Size& original_size,
      const struct rkapp::preprocess::LetterboxInfo& letterbox_info);

  // Optional: set NPU core mask (e.g. 1<<0, 1<<1, 1<<2).
  void setCoreMask(uint32_t core_mask);
  void setDecodeParams(const DecodeParams& params) override;

  // Thread-safe accessors for inferred decode metadata.
  int num_classes() const;
  bool has_objness() const;
  const std::vector<std::string>& class_names() const { return class_names_; }

  int getInputWidth() const override;
  int getInputHeight() const override;

private:
  struct Impl;
  mutable std::mutex state_mutex_;
  std::shared_ptr<Impl> impl_;
  std::string model_path_;
  int input_size_ = 640;
  bool is_initialized_ = false;
  uint32_t core_mask_ = 0x7; // 默认三核并行；可被 setCoreMask 覆盖
  
  // Auto-inferred detection parameters
  int num_classes_ = -1;
  bool has_objness_ = true;  // Most YOLO exports include objectness score
  std::vector<std::string> class_names_;
  DecodeParams decode_params_;
  ModelMeta model_meta_;
};

} // namespace rkapp::infer
