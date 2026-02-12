#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rkapp/infer/RknnDecodeUtils.hpp"
#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

namespace rkapp::infer::rknn_internal {

inline constexpr int kMaxSupportedRegMax = 32;

bool readFile(const std::string& path, std::vector<uint8_t>& out, std::string& err);

ModelMeta loadModelMeta(const std::string& model_path);

std::vector<Detection> decodeOutputAndNms(
    const float* logits_data,
    int out_n,
    int out_c,
    int out_elems,
    int out_n_dims,
    int out_dim1,
    int out_dim2,
    int& num_classes,
    const ModelMeta& model_meta,
    const DecodeParams& decode_params,
    const cv::Size& original_size,
    const rkapp::preprocess::LetterboxInfo& letterbox_info,
    const AnchorLayout* dfl_layout,
    const char* log_tag);

}  // namespace rkapp::infer::rknn_internal
