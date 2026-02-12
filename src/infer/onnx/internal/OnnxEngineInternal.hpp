#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "rkapp/infer/IInferEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

namespace rkapp::infer::onnx_internal {

struct QuantParams {
  float scale = 1.0f;
  int32_t zero_point = 0;
  bool has_scale = false;
  bool has_zero_point = false;
};

struct DecodeMeta {
  int reg_max = -1;
  std::vector<int> strides;
  std::string head;  // "dfl" / "raw" / ""
  int num_classes = -1;
  int has_objectness = -1;  // -1 unknown, 0 false, 1 true

  bool hasAny() const {
    return reg_max > 0 || !strides.empty() || !head.empty() || num_classes > 0 ||
           has_objectness >= 0;
  }
};

QuantParams resolveOutputQuantParams(const std::string& model_path,
                                     const std::string& output_name,
                                     const Ort::Session& session,
                                     Ort::AllocatorWithDefaultOptions& allocator);

DecodeMeta parseDecodeMetaFromSidecar(const std::string& model_path);

std::vector<Detection> parseOutput(Ort::Value& output,
                                   const rkapp::preprocess::LetterboxInfo& letterbox_info,
                                   cv::Size original_size,
                                   int input_size,
                                   const DecodeParams& params,
                                   const DecodeMeta& decode_meta,
                                   const QuantParams& quant_params,
                                   bool& unsupported_model);

}  // namespace rkapp::infer::onnx_internal
