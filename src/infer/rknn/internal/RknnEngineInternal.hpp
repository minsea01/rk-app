#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rkapp/infer/RknnDecodeUtils.hpp"
#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

namespace rkapp::infer::rknn_internal {

// 当前实现支持的 reg_max 上限（与解码缓冲区大小一致）。
inline constexpr int kMaxSupportedRegMax = 32;

// 读取二进制模型文件。
bool readFile(const std::string& path, std::vector<uint8_t>& out, std::string& err);

// 从 sidecar/metadata 加载模型解码元信息。
ModelMeta loadModelMeta(const std::string& model_path);

// 统一输出解码 + NMS 入口，内部根据元信息分发 RAW/DFL。
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

// RAW 头的量化输出解码入口：直接按 scale/zp 反量化，避免 runtime 先整块转 float。
std::vector<Detection> decodeOutputAndNmsQuantizedRaw(
    const int8_t* logits_data,
    float quant_scale,
    int32_t zero_point,
    int out_n,
    int out_w_stride,
    int out_c,
    int buffer_elems,
    int& num_classes,
    const ModelMeta& model_meta,
    const DecodeParams& decode_params,
    const cv::Size& original_size,
    const rkapp::preprocess::LetterboxInfo& letterbox_info,
    const char* log_tag);

std::vector<Detection> decodeOutputAndNmsQuantizedRaw(
    const uint8_t* logits_data,
    float quant_scale,
    int32_t zero_point,
    int out_n,
    int out_w_stride,
    int out_c,
    int buffer_elems,
    int& num_classes,
    const ModelMeta& model_meta,
    const DecodeParams& decode_params,
    const cv::Size& original_size,
    const rkapp::preprocess::LetterboxInfo& letterbox_info,
    const char* log_tag);

}  // namespace rkapp::infer::rknn_internal
