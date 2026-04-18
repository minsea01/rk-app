#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "rkapp/infer/IInferEngine.hpp"
#include "rkapp/infer/ModelMeta.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

namespace rkapp::infer::onnx_internal {

// ONNX 输出量化参数（主要用于 int8/uint8 输出反量化）。
struct QuantParams {
  float scale = 1.0f;
  int32_t zero_point = 0;
  bool has_scale = false;
  bool has_zero_point = false;
};

// 优先从模型 metadata 解析，必要时回退 sidecar。
QuantParams resolveOutputQuantParams(const std::string& model_path,
                                     const std::string& output_name,
                                     const Ort::Session& session,
                                     Ort::AllocatorWithDefaultOptions& allocator);

// 把输出张量解码为 Detection 列表（支持 RAW/DFL）。
std::vector<Detection> parseOutput(Ort::Value& output,
                                   const rkapp::preprocess::LetterboxInfo& letterbox_info,
                                   cv::Size original_size,
                                   int input_size,
                                   const DecodeParams& params,
                                   const rkapp::infer::ModelMeta& decode_meta,
                                   const QuantParams& quant_params,
                                   bool& unsupported_model);

}  // namespace rkapp::infer::onnx_internal
