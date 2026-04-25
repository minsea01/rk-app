#include "RknnEngineInternal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>

#include "rkapp/common/StringUtils.hpp"
#include "rkapp/common/log.hpp"
#include "rkapp/infer/RknnDecodeOptimized.hpp"
#include "rkapp/post/Postprocess.hpp"

namespace rkapp::infer::rknn_internal {

// RKNN 内部工具：sidecar 解析、输出布局适配、RAW/DFL 解码与 NMS。
namespace {

std::vector<std::string> modelSidecarCandidates(const std::string& model_path) {
  return {model_path + ".json", model_path + ".meta"};
}

bool modelMetaHasAny(const ModelMeta& meta) {
  return meta.reg_max > 0 || !meta.strides.empty() || !meta.head.empty() || meta.output_index >= 0 ||
         meta.num_classes > 0 || meta.has_objectness >= 0 || meta.score_is_probability >= 0 ||
         meta.coords_are_normalized >= 0 ||
         meta.task != "detect" || meta.num_keypoints > 0;
}

void mergeModelMetaMissingFields(ModelMeta& dst, const ModelMeta& src) {
  if (dst.reg_max <= 0 && src.reg_max > 0) {
    dst.reg_max = src.reg_max;
  }
  if (dst.strides.empty() && !src.strides.empty()) {
    dst.strides = src.strides;
  }
  if (dst.head.empty() && !src.head.empty()) {
    dst.head = src.head;
  }
  if (dst.output_index < 0 && src.output_index >= 0) {
    dst.output_index = src.output_index;
  }
  if (dst.num_classes <= 0 && src.num_classes > 0) {
    dst.num_classes = src.num_classes;
  }
  if (dst.has_objectness < 0 && src.has_objectness >= 0) {
    dst.has_objectness = src.has_objectness;
  }
  if (dst.score_is_probability < 0 && src.score_is_probability >= 0) {
    dst.score_is_probability = src.score_is_probability;
  }
  if (dst.coords_are_normalized < 0 && src.coords_are_normalized >= 0) {
    dst.coords_are_normalized = src.coords_are_normalized;
  }
  if (dst.task == "detect" && src.task != "detect") {
    dst.task = src.task;
  }
  if (dst.num_keypoints <= 0 && src.num_keypoints > 0) {
    dst.num_keypoints = src.num_keypoints;
  }
}

// 移除 sidecar 中的行注释/块注释，降低正则解析噪声。
std::string stripComments(const std::string& content) {
  std::string out;
  out.reserve(content.size());
  bool in_string = false;
  bool escape = false;
  for (size_t i = 0; i < content.size(); ++i) {
    char c = content[i];
    if (in_string) {
      out.push_back(c);
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      out.push_back(c);
      continue;
    }

    if (c == '/' && i + 1 < content.size()) {
      char next = content[i + 1];
      if (next == '/') {
        i += 1;
        while (i + 1 < content.size() && content[i + 1] != '\n') {
          ++i;
        }
        continue;
      }
      if (next == '*') {
        i += 1;
        while (i + 1 < content.size()) {
          if (content[i] == '*' && content[i + 1] == '/') {
            i += 1;
            break;
          }
          ++i;
        }
        continue;
      }
    }

    out.push_back(c);
  }
  return out;
}

std::vector<Detection> decodeAndPostprocess(const float* logits,
                                            int out_n,
                                            int out_c,
                                            int out_elems,
                                            int& num_classes,
                                            const ModelMeta& model_meta,
                                            const DecodeParams& decode_params,
                                            const cv::Size& original_size,
                                            const rkapp::preprocess::LetterboxInfo& letterbox_info,
                                            const AnchorLayout* dfl_layout,
                                            const char* log_tag) {
  // 解码主流程：先按 head 做 RAW/DFL 解析，再统一 NMS。
  std::vector<Detection> dets;
  int N = out_n;
  int C = out_c;
  if (N <= 0 || C <= 0) {
    LOGE(log_tag, ": Invalid output shape (C=", C, ", N=", N, ")");
    return {};
  }

  const float img_w = static_cast<float>(original_size.width);
  const float img_h = static_cast<float>(original_size.height);

  auto clamp_det = [&](Detection& d) {
    // 把框裁剪到原图边界内，避免越界。
    d.x = std::max(0.0f, std::min(d.x, img_w));
    d.y = std::max(0.0f, std::min(d.y, img_h));
    d.w = std::max(0.0f, std::min(d.w, img_w - d.x));
    d.h = std::max(0.0f, std::min(d.h, img_h - d.y));
  };

  auto sigmoid = [](float x) {
    return (x >= 0) ? (1.0f / (1.0f + std::exp(-x))) : (std::exp(x) / (1.0f + std::exp(x)));
  };
  const bool score_is_probability = (model_meta.score_is_probability == 1);
  const bool coords_are_normalized = (model_meta.coords_are_normalized == 1);
  auto activate_score = [&](float x) {
    return score_is_probability ? std::clamp(x, 0.0f, 1.0f) : sigmoid(x);
  };
  const float model_w = std::max(1.0f, letterbox_info.new_width + 2.0f * letterbox_info.dx);
  const float model_h = std::max(1.0f, letterbox_info.new_height + 2.0f * letterbox_info.dy);

  const bool head_raw = (model_meta.head == "raw");
  const bool head_dfl = (model_meta.head == "dfl");
  if (!head_raw && !head_dfl) {
    LOGE(log_tag, ": Missing or invalid head metadata (expected raw/dfl)");
    return {};
  }

  const int reg_max = model_meta.reg_max;
  const int cls_ch_meta = model_meta.num_classes;
  if (cls_ch_meta <= 0) {
    LOGE(log_tag, ": Missing or invalid num_classes in metadata");
    return {};
  }

  bool use_dfl = head_dfl;
  if (use_dfl) {
    if (reg_max <= 0 || model_meta.strides.empty()) {
      LOGE(log_tag, ": DFL decode requires reg_max and strides metadata");
      return {};
    }
  } else {
    if (model_meta.has_objectness < 0) {
      LOGE(log_tag, ": RAW decode requires has_objectness metadata");
      return {};
    }
  }

  if (use_dfl && reg_max > kMaxSupportedRegMax) {
    LOGE(log_tag, ": reg_max=", reg_max, " exceeds buffer size (", kMaxSupportedRegMax, ")");
    return {};
  }

  const int kpt_ch = model_meta.num_keypoints > 0 ? model_meta.num_keypoints * 3 : 0;
  const int expected_c = use_dfl ? (4 * reg_max + cls_ch_meta + kpt_ch)
                                 : (4 + cls_ch_meta + (model_meta.has_objectness == 1 ? 1 : 0));
  if (C != expected_c) {
    LOGE(log_tag, ": Output channels mismatch (C=", C, ", expected=", expected_c, ")");
    return {};
  }

  auto decode_raw = [&]() {
    // RAW 头：通道布局 [cx,cy,w,h,(obj),cls...]
    if (C < 4) {
      LOGW(log_tag, ": raw decode aborted due to insufficient channels (C=", C, ")");
      return;
    }

    const bool has_objectness = (model_meta.has_objectness == 1);
    const int cls_offset = has_objectness ? 5 : 4;
    const int cls_ch = cls_ch_meta;
    if (num_classes < 0) num_classes = cls_ch;

    if (cls_ch <= 0) {
      LOGW(log_tag, ": raw decode aborted due to invalid class channels (C=", C, ")");
      return;
    }

    const int max_idx = (cls_offset + cls_ch - 1) * N + (N - 1);
    if (max_idx >= out_elems) {
      LOGE(log_tag, ": raw decode aborted - index out of bounds (max_idx=", max_idx,
           ", out_elems=", out_elems, ")");
      return;
    }

    LOGI(log_tag, ": RAW decode with ", (has_objectness ? "objectness" : "no objectness"),
         ", cls_ch=", cls_ch, ", score_mode=",
         score_is_probability ? "identity" : "sigmoid", ", coord_mode=",
         coords_are_normalized ? "normalized" : "pixels");

    float max_seen_conf = 0.0f;
    float max_seen_cx = 0.0f;
    float max_seen_cy = 0.0f;
    float max_seen_w = 0.0f;
    float max_seen_h = 0.0f;
    int max_seen_idx = -1;
    for (int i = 0; i < N; i++) {
      float cx = logits[0 * N + i];
      float cy = logits[1 * N + i];
      float w = logits[2 * N + i];
      float h = logits[3 * N + i];
      if (coords_are_normalized) {
        cx *= model_w;
        cy *= model_h;
        w *= model_w;
        h *= model_h;
      }

      float obj = has_objectness ? activate_score(logits[4 * N + i]) : 1.0f;

      float max_conf = 0.f;
      int best = 0;
      for (int c = 0; c < cls_ch; c++) {
        float conf = activate_score(logits[(cls_offset + c) * N + i]);
        if (conf > max_conf) {
          max_conf = conf;
          best = c;
        }
      }
      float conf = obj * max_conf;
      if (conf > max_seen_conf) {
        max_seen_conf = conf;
        max_seen_cx = cx;
        max_seen_cy = cy;
        max_seen_w = w;
        max_seen_h = h;
        max_seen_idx = i;
      }
      if (conf >= decode_params.conf_thres) {
        Detection d;
        float scale = letterbox_info.scale, dx = letterbox_info.dx, dy = letterbox_info.dy;
        d.x = (cx - w / 2 - dx) / scale;
        d.y = (cy - h / 2 - dy) / scale;
        d.w = w / scale;
        d.h = h / scale;
        d.confidence = conf;
        d.class_id = best;
        d.class_name = "class_" + std::to_string(best);
        clamp_det(d);
        dets.push_back(d);
      }
    }
    if (dets.empty()) {
      LOGI(log_tag, ": RAW top candidate before threshold/NMS idx=", max_seen_idx,
           ", conf=", max_seen_conf, ", box=[", max_seen_cx, ",", max_seen_cy, ",",
           max_seen_w, ",", max_seen_h, "]");
    }
  };

  if (use_dfl) {
    // DFL 头：前 4*reg_max 为边分布，后续为类别（可选关键点）。
    const int cls_ch = cls_ch_meta;
    if (num_classes < 0) num_classes = cls_ch;
    if (!dfl_layout || !dfl_layout->valid ||
        dfl_layout->stride_map.size() != static_cast<size_t>(N) ||
        dfl_layout->anchor_cx.size() != static_cast<size_t>(N) ||
        dfl_layout->anchor_cy.size() != static_cast<size_t>(N)) {
      LOGE(log_tag, ": DFL decode requires a valid cached anchor layout");
      return {};
    }

    const AnchorLayout& layout = *dfl_layout;
    std::array<float, kMaxSupportedRegMax> probs_buf{};
    for (int i = 0; i < N; ++i) {
      auto dfl = dfl_decode_4sides_optimized(logits, i, N, reg_max, probs_buf.data());
      float s = layout.stride_map[i];
      float l = dfl[0] * s, t = dfl[1] * s, r = dfl[2] * s, b = dfl[3] * s;
      float x1 = layout.anchor_cx[i] - l;
      float y1 = layout.anchor_cy[i] - t;
      float x2 = layout.anchor_cx[i] + r;
      float y2 = layout.anchor_cy[i] + b;
      float best_conf = 0.f;
      int best_cls = 0;
      for (int c = 0; c < cls_ch; ++c) {
        float conf = sigmoid(logits[(4 * reg_max + c) * N + i]);
        if (conf > best_conf) {
          best_conf = conf;
          best_cls = c;
        }
      }
      if (best_conf >= decode_params.conf_thres) {
        Detection d;
        float scale = letterbox_info.scale, dx = letterbox_info.dx, dy = letterbox_info.dy;
        float w = x2 - x1, h = y2 - y1;
        d.x = (x1 - dx) / scale;
        d.y = (y1 - dy) / scale;
        d.w = w / scale;
        d.h = h / scale;
        d.confidence = best_conf;
        d.class_id = best_cls;
        d.class_name = "class_" + std::to_string(best_cls);

        // 姿态模型：额外解码关键点 (x,y,visibility)。
        if (model_meta.num_keypoints > 0) {
          const int kp_base = 4 * reg_max + cls_ch;
          d.keypoints.reserve(model_meta.num_keypoints);
          for (int k = 0; k < model_meta.num_keypoints; ++k) {
            float raw_x = logits[(kp_base + k * 3 + 0) * N + i];
            float raw_y = logits[(kp_base + k * 3 + 1) * N + i];
            float vis = sigmoid(logits[(kp_base + k * 3 + 2) * N + i]);
            float kx = layout.anchor_cx[i] + raw_x * 2.0f * s;
            float ky = layout.anchor_cy[i] + raw_y * 2.0f * s;
            // 把关键点从 letterbox 坐标还原到原图坐标。
            kx = (kx - dx) / scale;
            ky = (ky - dy) / scale;
            kx = std::max(0.0f, std::min(kx, img_w));
            ky = std::max(0.0f, std::min(ky, img_h));
            d.keypoints.push_back({kx, ky, vis});
          }
        }

        clamp_det(d);
        dets.push_back(d);
      }
    }
  } else {
    decode_raw();
  }

  rkapp::post::NMSConfig nms_cfg;
  nms_cfg.conf_thres = decode_params.conf_thres;
  nms_cfg.iou_thres = decode_params.iou_thres;
  if (decode_params.max_boxes > 0) {
    nms_cfg.max_det = decode_params.max_boxes;
    nms_cfg.topk = decode_params.max_boxes;
  }
  return rkapp::post::Postprocess::nms(dets, nms_cfg);
}

const float* maybeTransposeOutput(const float* logits_data,
                                  int out_n,
                                  int out_c,
                                  int n_dims,
                                  int dim1,
                                  int dim2,
                                  std::vector<float>& transpose_buf) {
  // 某些导出为 [N,C]，这里转为统一的 [C,N] 访问布局。
  if (n_dims >= 3 && dim1 == out_n && dim2 == out_c) {
    size_t total_elems = static_cast<size_t>(out_n) * out_c;
    transpose_buf.resize(total_elems);
    for (int n = 0; n < out_n; n++) {
      for (int c = 0; c < out_c; c++) {
        transpose_buf[c * out_n + n] = logits_data[n * out_c + c];
      }
    }
    return transpose_buf.data();
  }
  return logits_data;
}

template <typename QuantT>
std::vector<Detection> decodeRawAndPostprocessQuantized(
    const QuantT* logits,
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
    const char* log_tag) {
  if (model_meta.head != "raw") {
    LOGE(log_tag, ": Quantized fast path only supports raw heads");
    return {};
  }
  if (model_meta.num_classes <= 0) {
    LOGE(log_tag, ": Missing or invalid num_classes in metadata");
    return {};
  }
  if (model_meta.has_objectness < 0) {
    LOGE(log_tag, ": RAW decode requires has_objectness metadata");
    return {};
  }
  if (quant_scale <= 0.0f) {
    LOGE(log_tag, ": Invalid quantized output scale ", quant_scale);
    return {};
  }

  const int N = out_n;
  const int C = out_c;
  const int channel_stride = out_w_stride > 0 ? out_w_stride : out_n;
  if (N <= 0 || C <= 0) {
    LOGE(log_tag, ": Invalid quantized output shape (C=", C, ", N=", N, ")");
    return {};
  }

  const bool has_objectness = (model_meta.has_objectness == 1);
  const int cls_offset = has_objectness ? 5 : 4;
  const int cls_ch = model_meta.num_classes;
  if (num_classes < 0) num_classes = cls_ch;
  const int expected_c = 4 + cls_ch + (has_objectness ? 1 : 0);
  if (C != expected_c) {
    LOGE(log_tag, ": Quantized RAW output channels mismatch (C=", C, ", expected=", expected_c,
         ")");
    return {};
  }

  const int max_idx = (cls_offset + cls_ch - 1) * channel_stride + (N - 1);
  if (max_idx >= buffer_elems) {
    LOGE(log_tag, ": Quantized RAW decode aborted - index out of bounds (max_idx=", max_idx,
         ", buffer_elems=", buffer_elems, ", stride=", channel_stride, ")");
    return {};
  }

  const float img_w = static_cast<float>(original_size.width);
  const float img_h = static_cast<float>(original_size.height);
  auto clamp_det = [&](Detection& d) {
    d.x = std::max(0.0f, std::min(d.x, img_w));
    d.y = std::max(0.0f, std::min(d.y, img_h));
    d.w = std::max(0.0f, std::min(d.w, img_w - d.x));
    d.h = std::max(0.0f, std::min(d.h, img_h - d.y));
  };
  auto sigmoid = [](float x) {
    return (x >= 0) ? (1.0f / (1.0f + std::exp(-x))) : (std::exp(x) / (1.0f + std::exp(x)));
  };
  const bool score_is_probability = (model_meta.score_is_probability == 1);
  const bool coords_are_normalized = (model_meta.coords_are_normalized == 1);
  auto activate_score = [&](float x) {
    return score_is_probability ? std::clamp(x, 0.0f, 1.0f) : sigmoid(x);
  };
  auto load = [&](int idx) -> float {
    return (static_cast<float>(logits[idx]) - static_cast<float>(zero_point)) * quant_scale;
  };
  const float model_w = std::max(1.0f, letterbox_info.new_width + 2.0f * letterbox_info.dx);
  const float model_h = std::max(1.0f, letterbox_info.new_height + 2.0f * letterbox_info.dy);

  static std::once_flag quantized_raw_log_once;
  std::call_once(quantized_raw_log_once, [&]() {
    LOGI(log_tag, ": Quantized RAW decode fast path (cls_ch=", cls_ch, ", has_objectness=",
         has_objectness ? "true" : "false", ", score_mode=",
         score_is_probability ? "identity" : "sigmoid", ", coord_mode=",
         coords_are_normalized ? "normalized" : "pixels", ", scale=", quant_scale,
         ", zp=", zero_point, ", stride=", channel_stride, ")");
  });

  std::vector<Detection> dets;
  float max_seen_conf = 0.0f;
  float max_seen_cx = 0.0f;
  float max_seen_cy = 0.0f;
  float max_seen_w = 0.0f;
  float max_seen_h = 0.0f;
  int max_seen_idx = -1;
  for (int i = 0; i < N; ++i) {
    float cx = load(0 * channel_stride + i);
    float cy = load(1 * channel_stride + i);
    float w = load(2 * channel_stride + i);
    float h = load(3 * channel_stride + i);
    if (coords_are_normalized) {
      cx *= model_w;
      cy *= model_h;
      w *= model_w;
      h *= model_h;
    }
    const float obj =
        has_objectness ? activate_score(load(4 * channel_stride + i)) : 1.0f;

    float max_conf = 0.0f;
    int best = 0;
    for (int c = 0; c < cls_ch; ++c) {
      const float conf = activate_score(load((cls_offset + c) * channel_stride + i));
      if (conf > max_conf) {
        max_conf = conf;
        best = c;
      }
    }
    const float conf = obj * max_conf;
    if (conf > max_seen_conf) {
      max_seen_conf = conf;
      max_seen_cx = cx;
      max_seen_cy = cy;
      max_seen_w = w;
      max_seen_h = h;
      max_seen_idx = i;
    }
    if (conf < decode_params.conf_thres) {
      continue;
    }

    Detection d;
    const float scale = letterbox_info.scale;
    const float dx = letterbox_info.dx;
    const float dy = letterbox_info.dy;
    d.x = (cx - w / 2.0f - dx) / scale;
    d.y = (cy - h / 2.0f - dy) / scale;
    d.w = w / scale;
    d.h = h / scale;
    d.confidence = conf;
    d.class_id = best;
    d.class_name = "class_" + std::to_string(best);
    clamp_det(d);
    dets.push_back(d);
  }

  if (dets.empty()) {
    LOGI(log_tag, ": Quantized RAW top candidate before threshold/NMS idx=", max_seen_idx,
         ", conf=", max_seen_conf, ", box=[", max_seen_cx, ",", max_seen_cy, ",",
         max_seen_w, ",", max_seen_h, "]");
  }

  rkapp::post::NMSConfig nms_cfg;
  nms_cfg.conf_thres = decode_params.conf_thres;
  nms_cfg.iou_thres = decode_params.iou_thres;
  if (decode_params.max_boxes > 0) {
    nms_cfg.max_det = decode_params.max_boxes;
    nms_cfg.topk = decode_params.max_boxes;
  }
  return rkapp::post::Postprocess::nms(dets, nms_cfg);
}

}  // namespace

bool readFile(const std::string& path, std::vector<uint8_t>& out, std::string& err) {
  out.clear();
  err.clear();

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    err = "open failed";
    return false;
  }

  out.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  if (ifs.bad()) {
    out.clear();
    err = "read failed";
    return false;
  }
  return true;
}

ModelMeta loadModelMeta(const std::string& model_path) {
  return rkapp::infer::loadModelMetaFromPath(model_path).meta;
}

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
    const char* log_tag) {
  thread_local std::vector<float> transpose_local;
  const float* logits = maybeTransposeOutput(
      logits_data, out_n, out_c, out_n_dims, out_dim1, out_dim2, transpose_local);
  return decodeAndPostprocess(logits, out_n, out_c, out_elems, num_classes, model_meta,
                              decode_params, original_size, letterbox_info, dfl_layout,
                              log_tag);
}

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
    const char* log_tag) {
  return decodeRawAndPostprocessQuantized(logits_data, quant_scale, zero_point, out_n,
                                          out_w_stride, out_c, buffer_elems, num_classes,
                                          model_meta, decode_params, original_size,
                                          letterbox_info, log_tag);
}

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
    const char* log_tag) {
  return decodeRawAndPostprocessQuantized(logits_data, quant_scale, zero_point, out_n,
                                          out_w_stride, out_c, buffer_elems, num_classes,
                                          model_meta, decode_params, original_size,
                                          letterbox_info, log_tag);
}

}  // namespace rkapp::infer::rknn_internal
