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

namespace {

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
    d.x = std::max(0.0f, std::min(d.x, img_w));
    d.y = std::max(0.0f, std::min(d.y, img_h));
    d.w = std::max(0.0f, std::min(d.w, img_w - d.x));
    d.h = std::max(0.0f, std::min(d.h, img_h - d.y));
  };

  auto sigmoid = [](float x) {
    return (x >= 0) ? (1.0f / (1.0f + std::exp(-x))) : (std::exp(x) / (1.0f + std::exp(x)));
  };

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

    static std::once_flag raw_decode_log_once;
    std::call_once(raw_decode_log_once, [&]() {
      LOGI(log_tag, ": RAW decode with ", (has_objectness ? "objectness" : "no objectness"),
           ", cls_ch=", cls_ch);
    });

    for (int i = 0; i < N; i++) {
      float cx = logits[0 * N + i];
      float cy = logits[1 * N + i];
      float w = logits[2 * N + i];
      float h = logits[3 * N + i];

      float obj = has_objectness ? sigmoid(logits[4 * N + i]) : 1.0f;

      float max_conf = 0.f;
      int best = 0;
      for (int c = 0; c < cls_ch; c++) {
        float conf = sigmoid(logits[(cls_offset + c) * N + i]);
        if (conf > max_conf) {
          max_conf = conf;
          best = c;
        }
      }
      float conf = obj * max_conf;
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
  };

  if (use_dfl) {
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

        // Decode keypoints for pose models
        if (model_meta.num_keypoints > 0) {
          const int kp_base = 4 * reg_max + cls_ch;
          d.keypoints.reserve(model_meta.num_keypoints);
          for (int k = 0; k < model_meta.num_keypoints; ++k) {
            float raw_x = logits[(kp_base + k * 3 + 0) * N + i];
            float raw_y = logits[(kp_base + k * 3 + 1) * N + i];
            float vis = sigmoid(logits[(kp_base + k * 3 + 2) * N + i]);
            float kx = (layout.anchor_cx[i] + raw_x * 2.0f) * s;
            float ky = (layout.anchor_cy[i] + raw_y * 2.0f) * s;
            // Reverse letterbox transform to original image coordinates
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
  ModelMeta meta;
  const std::vector<std::string> candidates = {
      model_path + ".json",
      model_path + ".meta",
      "artifacts/models/decode_meta.json"};

  auto parse_int = [](const std::string& content, const std::string& key) -> int {
    try {
      std::smatch m;
      std::regex re("(^|[^A-Za-z0-9_])\"?" + key + "\"?\\s*[:=]\\s*([0-9]+)");
      if (std::regex_search(content, m, re) && m.size() > 2) {
        return std::stoi(m[2].str());
      }
    } catch (const std::exception&) {
    }
    return -1;
  };

  auto parse_bool = [](const std::string& content, const std::string& key) -> int {
    try {
      std::smatch m;
      std::regex re("(^|[^A-Za-z0-9_])\"?" + key + "\"?\\s*[:=]\\s*(true|false|0|1)",
                    std::regex::icase);
      if (std::regex_search(content, m, re) && m.size() > 2) {
        const std::string v = rkapp::common::toLowerCopy(m[2].str());
        if (v == "1" || v == "true") return 1;
        if (v == "0" || v == "false") return 0;
      }
    } catch (const std::exception&) {
    }
    return -1;
  };

  auto parse_strides = [](const std::string& content) -> std::vector<int> {
    std::vector<int> out;
    std::regex re(R"((^|[^A-Za-z0-9_])\"?strides\"?\s*[:=]\s*\[([^\]]+)\])");
    std::smatch m;
    if (!std::regex_search(content, m, re) || m.size() < 3) return out;
    std::string body = m[2].str();
    std::stringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      try {
        out.push_back(std::stoi(tok));
      } catch (...) {
      }
    }
    return out;
  };

  auto parse_head = [](const std::string& content) -> std::string {
    std::smatch m;
    std::regex re_quoted(R"("head"\s*[:=]\s*\"?([a-zA-Z]+)\"?)");
    if (std::regex_search(content, m, re_quoted) && m.size() > 1) {
      return rkapp::common::toLowerCopy(m[1].str());
    }
    std::regex re_unquoted(R"((^|[^A-Za-z0-9_])head\s*[:=]\s*\"?([a-zA-Z]+)\"?)");
    if (std::regex_search(content, m, re_unquoted) && m.size() > 2) {
      return rkapp::common::toLowerCopy(m[2].str());
    }
    return {};
  };

  for (const auto& path : candidates) {
    std::ifstream f(path);
    if (!f.is_open()) continue;

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (content.empty()) continue;

    std::string sanitized = stripComments(content);
    int reg = parse_int(sanitized, "reg_max");
    if (reg > 0) meta.reg_max = reg;

    auto strides = parse_strides(sanitized);
    if (!strides.empty()) meta.strides = std::move(strides);

    std::string head = parse_head(sanitized);
    if (!head.empty()) meta.head = std::move(head);

    int out_idx = parse_int(sanitized, "output_index");
    if (out_idx < 0) out_idx = parse_int(sanitized, "output_idx");
    if (out_idx >= 0) meta.output_index = out_idx;

    int num_classes = parse_int(sanitized, "num_classes");
    if (num_classes < 0) num_classes = parse_int(sanitized, "classes");
    if (num_classes < 0) num_classes = parse_int(sanitized, "nc");
    if (num_classes > 0) meta.num_classes = num_classes;

    int has_obj = parse_bool(sanitized, "has_objectness");
    if (has_obj < 0) has_obj = parse_bool(sanitized, "objectness");
    if (has_obj < 0) has_obj = parse_bool(sanitized, "has_obj");
    if (has_obj >= 0) meta.has_objectness = has_obj;

    std::string task = parse_head(sanitized);  // reuse string parser
    // parse_head only accepts "dfl"/"raw", so use a dedicated regex for "task"
    {
      std::smatch tm;
      std::regex re_task(R"("task"\s*[:=]\s*"?([a-zA-Z]+)"?)");
      if (std::regex_search(sanitized, tm, re_task) && tm.size() > 1) {
        std::string t = rkapp::common::toLowerCopy(tm[1].str());
        if (t == "detect" || t == "pose") {
          meta.task = std::move(t);
        }
      }
    }

    int num_kpts = parse_int(sanitized, "num_keypoints");
    if (num_kpts < 0) num_kpts = parse_int(sanitized, "kpt_shape");
    if (num_kpts > 0) meta.num_keypoints = num_kpts;

    if (meta.reg_max > 0 || !meta.strides.empty() || !meta.head.empty() ||
        meta.output_index >= 0 || meta.num_classes > 0 || meta.has_objectness >= 0 ||
        meta.task != "detect" || meta.num_keypoints > 0) {
      LOGI("RknnEngine: loaded decode metadata from ", path);
      if (meta.num_keypoints > 0) {
        LOGI("RknnEngine: pose model (", meta.num_keypoints, " keypoints)");
      }
      break;
    }
  }

  return meta;
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

}  // namespace rkapp::infer::rknn_internal
