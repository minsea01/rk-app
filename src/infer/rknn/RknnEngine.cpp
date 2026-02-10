#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/infer/RknnDecodeUtils.hpp"
#include "rkapp/infer/RknnDecodeOptimized.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "rkapp/common/DmaBuf.hpp"
#include "log.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <utility>
#include <unistd.h>  // for close()

#if RKNN_PLATFORM
#include <rknn_api.h>
  #if RKNN_USE_RGA
  #include <im2d.h>
  #include <rga.h>
  #endif
#endif

namespace rkapp::infer {

struct RknnEngine::Impl {
  std::mutex infer_mutex;
  std::atomic<bool> shutting_down{false};
#if RKNN_PLATFORM
  rknn_context ctx = 0;
  rknn_input_output_num io_num{};
  rknn_tensor_attr in_attr{};
  rknn_tensor_attr out_attr{};
  rknn_tensor_format input_fmt = RKNN_TENSOR_NHWC;
  rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
  int out_elems = 0;
  int out_c = 0, out_n = 0; // C (8), N (8400) or transposed

  AnchorLayout dfl_layout;
#endif
};

namespace {

constexpr int kMaxSupportedRegMax = 32;

struct ScopedFd {
  int fd = -1;

  ScopedFd() = default;
  explicit ScopedFd(int f) : fd(f) {}

  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  ScopedFd(ScopedFd&& other) noexcept : fd(other.fd) { other.fd = -1; }
  ScopedFd& operator=(ScopedFd&& other) noexcept {
    if (this != &other) {
      reset();
      fd = other.fd;
      other.fd = -1;
    }
    return *this;
  }

  ~ScopedFd() { reset(); }

  void reset(int new_fd = -1) {
    if (fd >= 0) close(fd);
    fd = new_fd;
  }

  int get() const { return fd; }
  bool valid() const { return fd >= 0; }
};

bool read_file(const std::string& p, std::vector<uint8_t>& out, std::string& err){
  out.clear();
  err.clear();

  std::ifstream ifs(p, std::ios::binary);
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

std::string strip_comments(const std::string& content) {
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

// Best-effort model metadata loader to avoid纯启发式解码。
// 优先读取侧car文件：<model_path>.json / <model_path>.meta / artifacts/models/decode_meta.json
rkapp::infer::ModelMeta load_model_meta(const std::string& model_path) {
  rkapp::infer::ModelMeta meta;
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
      // Catches both std::regex_error and std::invalid_argument from stoi
    }
    return -1;
  };

  auto parse_bool = [](const std::string& content, const std::string& key) -> int {
    try {
      std::smatch m;
      std::regex re("(^|[^A-Za-z0-9_])\"?" + key + "\"?\\s*[:=]\\s*(true|false|0|1)",
                    std::regex::icase);
      if (std::regex_search(content, m, re) && m.size() > 2) {
        std::string v = m[2].str();
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        if (v == "1" || v == "true") return 1;
        if (v == "0" || v == "false") return 0;
      }
    } catch (const std::exception&) {
      // Catches std::regex_error and any other parsing exceptions
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
      std::string v = m[1].str();
      std::transform(v.begin(), v.end(), v.begin(), ::tolower);
      return v;
    }
    std::regex re_unquoted(R"((^|[^A-Za-z0-9_])head\s*[:=]\s*\"?([a-zA-Z]+)\"?)");
    if (std::regex_search(content, m, re_unquoted) && m.size() > 2) {
      std::string v = m[2].str();
      std::transform(v.begin(), v.end(), v.begin(), ::tolower);
      return v;
    }
    return {};
  };

  for (const auto& p : candidates) {
    std::ifstream f(p);
    if (!f.is_open()) continue;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (content.empty()) continue;
    std::string sanitized = strip_comments(content);
    int reg = parse_int(sanitized, "reg_max");
    if (reg > 0) meta.reg_max = reg;
    auto strides = parse_strides(sanitized);
    if (!strides.empty()) meta.strides = strides;
    std::string head = parse_head(sanitized);
    if (!head.empty()) meta.head = head;
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
    if (meta.reg_max > 0 || !meta.strides.empty() || !meta.head.empty() ||
        meta.output_index >= 0 || meta.num_classes > 0 || meta.has_objectness >= 0) {
      LOGI("RknnEngine: loaded decode metadata from ", p);
      break;
    }
  }
  return meta;
}

std::vector<Detection> decode_and_postprocess(
    const float* logits,
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

  // Numerically stable sigmoid to prevent overflow
  auto sigmoid = [](float x) {
    return (x >= 0) ? (1.0f / (1.0f + std::exp(-x)))
                    : (std::exp(x) / (1.0f + std::exp(x)));
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

  // Guard against reg_max overflow in fixed-size decode buffer.
  if (use_dfl && reg_max > kMaxSupportedRegMax) {
    LOGE(log_tag, ": reg_max=", reg_max, " exceeds buffer size (", kMaxSupportedRegMax, ")");
    return {};
  }

  const int expected_c = use_dfl
      ? (4 * reg_max + cls_ch_meta)
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

    // Detect whether model has objectness channel
    // YOLOv5/v7: [cx, cy, w, h, obj, cls...]  -> C = 5 + num_classes (has obj)
    // YOLOv8/v11: [cx, cy, w, h, cls...]      -> C = 4 + num_classes (no obj)
    const bool has_objectness = (model_meta.has_objectness == 1);
    const int cls_offset = has_objectness ? 5 : 4;  // Start of class channels
    const int cls_ch = cls_ch_meta;
    if (num_classes < 0) num_classes = cls_ch;

    if (cls_ch <= 0) {
      LOGW(log_tag, ": raw decode aborted due to invalid class channels (C=", C, ")");
      return;
    }

    // 边界检查：确保最大索引不越界
    const int max_idx = (cls_offset + cls_ch - 1) * N + (N - 1);
    if (max_idx >= out_elems) {
      LOGE(log_tag, ": raw decode aborted - index out of bounds (max_idx=", max_idx,
           ", out_elems=", out_elems, ")");
      return;
    }

    LOGI(log_tag, ": RAW decode with ", (has_objectness ? "objectness" : "no objectness"),
         ", cls_ch=", cls_ch);

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
    // DFL path
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

/**
 * @brief Transpose RKNN output from [1, N, C] to [C, N] layout if needed.
 *
 * RKNN may output logits in [1, N, C] format (n_dims=3, dims[1]=N, dims[2]=C),
 * but decode expects [C, N]. This helper transposes in-place using thread_local buffer.
 *
 * @param logits_data Original logits data pointer
 * @param out_n Number of anchors (N dimension)
 * @param out_c Number of channels (C dimension)
 * @param n_dims Number of output dimensions
 * @param dim1 dims[1] from output attribute
 * @param dim2 dims[2] from output attribute
 * @param transpose_buf Thread-local buffer for transposed data (resized if needed)
 * @return Pointer to logits (original or transposed)
 */
inline const float* maybe_transpose_output(const float* logits_data, int out_n, int out_c,
                                           int n_dims, int dim1, int dim2,
                                           std::vector<float>& transpose_buf) {
  // Check if transpose is needed: [1, N, C] layout needs to become [C, N]
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

std::vector<Detection> decode_output_and_nms(
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
  const float* logits =
      maybe_transpose_output(logits_data, out_n, out_c, out_n_dims, out_dim1, out_dim2,
                             transpose_local);
  return decode_and_postprocess(
      logits, out_n, out_c, out_elems, num_classes, model_meta, decode_params, original_size,
      letterbox_info, dfl_layout, log_tag);
}

} // namespace

RknnEngine::RknnEngine() = default;
RknnEngine::~RknnEngine(){ release(); }

bool RknnEngine::init(const std::string& model_path, int img_size){
  release();

  auto new_impl = std::make_shared<Impl>();
  ModelMeta model_meta = load_model_meta(model_path);
  int inferred_num_classes = -1;
  bool inferred_has_objness = true;

#if RKNN_PLATFORM
  bool ctx_ready = false;
  auto cleanup = [&]() {
    if (ctx_ready && new_impl && new_impl->ctx) {
      rknn_destroy(new_impl->ctx);
      new_impl->ctx = 0;
    }
  };

  std::vector<uint8_t> blob;
  std::string read_err;
  if (!read_file(model_path, blob, read_err)) {
    LOGE("RknnEngine: failed to read model file: ", model_path,
         " (exists=", std::filesystem::exists(model_path), ", reason=", read_err, ")");
    cleanup();
    return false;
  }
  if(blob.empty()){
    LOGE("RknnEngine: model file empty: ", model_path);
    cleanup();
    return false;
  }

  // Step 1: Initialize RKNN context
  int ret = rknn_init(&new_impl->ctx, blob.data(), blob.size(), 0, nullptr);
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_init failed with code ", ret);
    cleanup();
    return false;
  }
  ctx_ready = true;

  // Step 2: Configure NPU core mask using official rknn_set_core_mask API
  // This is the correct API for RK3588 multi-core scheduling (SDK 1.3.0+)
  // Reference: https://github.com/rockchip-linux/rknpu2/blob/master/runtime/RK3588/Linux/librknn_api/include/rknn_api.h
  //
  // Core mask values (rknn_core_mask enum):
  //   RKNN_NPU_CORE_AUTO    = 0  (default, auto-select idle core)
  //   RKNN_NPU_CORE_0       = 1  (2 TOPS)
  //   RKNN_NPU_CORE_1       = 2  (2 TOPS)
  //   RKNN_NPU_CORE_2       = 4  (2 TOPS)
  //   RKNN_NPU_CORE_0_1     = 3  (4 TOPS, dual-core)
  //   RKNN_NPU_CORE_0_1_2   = 7  (6 TOPS, tri-core - recommended for max throughput)
  //
  // Note: Multi-core mode accelerates: Conv, DepthwiseConv, Add, Concat, Relu, Clip, Relu6,
  //       ThresholdedRelu, Prelu, LeakyRelu. Other ops fallback to Core0.
  uint32_t core_mask_snapshot = 0;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    core_mask_snapshot = core_mask_;
  }
  if (core_mask_snapshot != 0) {
    // Map our core_mask_ to rknn_core_mask enum
    rknn_core_mask npu_core_mask = RKNN_NPU_CORE_AUTO;
    switch (core_mask_snapshot) {
      case 0x1: npu_core_mask = RKNN_NPU_CORE_0; break;
      case 0x2: npu_core_mask = RKNN_NPU_CORE_1; break;
      case 0x4: npu_core_mask = RKNN_NPU_CORE_2; break;
      case 0x3: npu_core_mask = RKNN_NPU_CORE_0_1; break;
      case 0x7: npu_core_mask = RKNN_NPU_CORE_0_1_2; break;
      default:
        // For any other mask, use all three cores for maximum throughput
        LOGW("RknnEngine: Unknown core_mask 0x", std::hex, core_mask_snapshot, std::dec,
             ", defaulting to RKNN_NPU_CORE_0_1_2 (6 TOPS)");
        npu_core_mask = RKNN_NPU_CORE_0_1_2;
        break;
    }

    int mask_ret = rknn_set_core_mask(new_impl->ctx, npu_core_mask);
    if (mask_ret != RKNN_SUCC) {
      // Non-fatal: log warning but continue (single-core fallback)
      LOGW("RknnEngine: rknn_set_core_mask failed (code ", mask_ret,
           "). Running on default core. This may reduce throughput by up to 66%.");
    } else {
      const char* core_desc = "AUTO";
      switch (npu_core_mask) {
        case RKNN_NPU_CORE_0: core_desc = "Core0 (2 TOPS)"; break;
        case RKNN_NPU_CORE_1: core_desc = "Core1 (2 TOPS)"; break;
        case RKNN_NPU_CORE_2: core_desc = "Core2 (2 TOPS)"; break;
        case RKNN_NPU_CORE_0_1: core_desc = "Core0+1 (4 TOPS)"; break;
        case RKNN_NPU_CORE_0_1_2: core_desc = "Core0+1+2 (6 TOPS)"; break;
        default: break;
      }
      LOGI("RknnEngine: NPU multi-core enabled: ", core_desc);
    }
  } else {
    LOGI("RknnEngine: NPU core_mask=0, using RKNN_NPU_CORE_AUTO mode");
  }

  ret = rknn_query(new_impl->ctx, RKNN_QUERY_IN_OUT_NUM, &new_impl->io_num, sizeof(new_impl->io_num));
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: RKNN_QUERY_IN_OUT_NUM failed: ", ret);
    cleanup();
    return false;
  }

  if (new_impl->io_num.n_input == 0) {
    LOGE("RknnEngine: No inputs detected (n_input=", new_impl->io_num.n_input, ")");
    cleanup();
    return false;
  }

  std::memset(&new_impl->in_attr, 0, sizeof(new_impl->in_attr));
  new_impl->in_attr.index = 0;
  ret = rknn_query(new_impl->ctx, RKNN_QUERY_INPUT_ATTR, &new_impl->in_attr, sizeof(new_impl->in_attr));
  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine: RKNN_QUERY_INPUT_ATTR failed: ", ret);
    cleanup();
    return false;
  }

  new_impl->input_fmt = new_impl->in_attr.fmt;
  new_impl->input_type = new_impl->in_attr.type;
  if (new_impl->input_fmt != RKNN_TENSOR_NHWC && new_impl->input_fmt != RKNN_TENSOR_NCHW) {
    LOGE("RknnEngine: Unsupported input format (fmt=", new_impl->input_fmt, ")");
    cleanup();
    return false;
  }
  if (new_impl->input_type != RKNN_TENSOR_UINT8) {
    LOGE("RknnEngine: Unsupported input type (type=", new_impl->input_type, ")");
    cleanup();
    return false;
  }

  int input_c = 0;
  int input_h = 0;
  int input_w = 0;
  if (new_impl->input_fmt == RKNN_TENSOR_NHWC) {
    if (new_impl->in_attr.n_dims == 4) {
      input_h = static_cast<int>(new_impl->in_attr.dims[1]);
      input_w = static_cast<int>(new_impl->in_attr.dims[2]);
      input_c = static_cast<int>(new_impl->in_attr.dims[3]);
    } else if (new_impl->in_attr.n_dims == 3) {
      input_h = static_cast<int>(new_impl->in_attr.dims[0]);
      input_w = static_cast<int>(new_impl->in_attr.dims[1]);
      input_c = static_cast<int>(new_impl->in_attr.dims[2]);
    }
  } else if (new_impl->input_fmt == RKNN_TENSOR_NCHW) {
    if (new_impl->in_attr.n_dims == 4) {
      input_c = static_cast<int>(new_impl->in_attr.dims[1]);
      input_h = static_cast<int>(new_impl->in_attr.dims[2]);
      input_w = static_cast<int>(new_impl->in_attr.dims[3]);
    } else if (new_impl->in_attr.n_dims == 3) {
      input_c = static_cast<int>(new_impl->in_attr.dims[0]);
      input_h = static_cast<int>(new_impl->in_attr.dims[1]);
      input_w = static_cast<int>(new_impl->in_attr.dims[2]);
    }
  }

  if (input_c > 0 && input_c != 3) {
    LOGE("RknnEngine: Unsupported input channels (C=", input_c, "), expected 3");
    cleanup();
    return false;
  }
  if (input_h > 0 && input_w > 0 &&
      (input_h != img_size || input_w != img_size)) {
    LOGW("RknnEngine: Input size mismatch with model (model=", input_w, "x",
         input_h, ", cfg=", img_size, "x", img_size, ")");
  }

  // Select output tensor (metadata required for multi-output models)
  int best_output_idx = 0;
  rknn_tensor_attr best_attr{};

  auto query_output_attr = [&](uint32_t idx, rknn_tensor_attr& attr) -> bool {
    std::memset(&attr, 0, sizeof(attr));
    attr.index = idx;
    if (rknn_query(new_impl->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC) {
      LOGW("RknnEngine: RKNN_QUERY_OUTPUT_ATTR failed for output ", idx);
      return false;
    }
    return true;
  };

  if (new_impl->io_num.n_output == 1) {
    best_output_idx = 0;
    if (!query_output_attr(0, best_attr)) {
      cleanup();
      return false;
    }
    LOGI("RknnEngine: Single output detected, using index 0");
  } else if (new_impl->io_num.n_output > 1) {
    if (model_meta.output_index < 0) {
      LOGE("RknnEngine: Multiple outputs require metadata output_index");
      cleanup();
      return false;
    }
    if (model_meta.output_index >= static_cast<int>(new_impl->io_num.n_output)) {
      LOGE("RknnEngine: output_index out of range (", model_meta.output_index,
           "), n_output=", new_impl->io_num.n_output);
      cleanup();
      return false;
    }
    best_output_idx = model_meta.output_index;
    if (!query_output_attr(best_output_idx, best_attr)) {
      cleanup();
      return false;
    }
    LOGI("RknnEngine: Using output index from metadata: ", best_output_idx);
  } else {
    LOGE("RknnEngine: No outputs detected (n_output=", new_impl->io_num.n_output, ")");
    cleanup();
    return false;
  }

  new_impl->out_attr = best_attr;

  // Auto-infer classes from output tensor dimensions
  new_impl->out_elems = 1;
  for(uint32_t i=0;i<new_impl->out_attr.n_dims;i++) new_impl->out_elems *= new_impl->out_attr.dims[i];
  
  if (model_meta.head != "raw" && model_meta.head != "dfl") {
    LOGE("RknnEngine: Missing or invalid head metadata (expected raw/dfl)");
    cleanup();
    return false;
  }
  if (model_meta.num_classes <= 0) {
    LOGE("RknnEngine: Missing or invalid num_classes metadata");
    cleanup();
    return false;
  }

  int expected_c = 0;
  if (model_meta.head == "dfl") {
    if (model_meta.reg_max <= 0 || model_meta.strides.empty()) {
      LOGE("RknnEngine: DFL decode requires reg_max and strides metadata");
      cleanup();
      return false;
    }
    expected_c = 4 * model_meta.reg_max + model_meta.num_classes;
  } else {
    if (model_meta.has_objectness < 0) {
      LOGE("RknnEngine: RAW decode requires has_objectness metadata");
      cleanup();
      return false;
    }
    expected_c = 4 + model_meta.num_classes + (model_meta.has_objectness == 1 ? 1 : 0);
  }

  if (new_impl->out_attr.n_dims != 3) {
    LOGE("RknnEngine: Unsupported output dimensions: n_dims=", new_impl->out_attr.n_dims);
    cleanup();
    return false;
  }

  int64_t d1 = new_impl->out_attr.dims[1];
  int64_t d2 = new_impl->out_attr.dims[2];
  if (d1 == expected_c && d2 != expected_c) {
    new_impl->out_c = expected_c;
    new_impl->out_n = static_cast<int>(d2);
    LOGI("RknnEngine: Detected channels_first layout [1, ", d1, ", ", d2, "]");
  } else if (d2 == expected_c && d1 != expected_c) {
    new_impl->out_c = expected_c;
    new_impl->out_n = static_cast<int>(d1);
    LOGI("RknnEngine: Detected channels_last layout [1, ", d1, ", ", d2, "], will transpose");
  } else {
    LOGE("RknnEngine: Output layout mismatch (dims=[1,", d1, ",", d2,
         "], expected C=", expected_c, ")");
    cleanup();
    return false;
  }

  inferred_num_classes = (model_meta.num_classes > 0) ? model_meta.num_classes : -1;
  if (model_meta.has_objectness >= 0) {
    inferred_has_objness = (model_meta.has_objectness == 1);
  }

  const bool expect_dfl = (model_meta.head == "dfl");
  if (expect_dfl) {
    if (model_meta.reg_max <= 0 || model_meta.strides.empty()) {
      LOGE("RknnEngine: DFL decode requires reg_max and strides metadata");
      cleanup();
      return false;
    }
    if (model_meta.reg_max > kMaxSupportedRegMax) {
      LOGE("RknnEngine: reg_max=", model_meta.reg_max, " exceeds max (", kMaxSupportedRegMax, ")");
      cleanup();
      return false;
    }
    AnchorLayout layout = build_anchor_layout(img_size, new_impl->out_n, model_meta.strides);
    if (!layout.valid) {
      LOGE("RknnEngine: anchor layout invalid for provided strides");
      cleanup();
      return false;
    }
    new_impl->dfl_layout = std::move(layout);
  }

  LOGI("RknnEngine: Output elements per inference: ", new_impl->out_elems);
#else
  LOGW("RknnEngine: RKNN platform not enabled at build time");
#endif
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    model_path_ = model_path;
    input_size_ = img_size;
    model_meta_ = std::move(model_meta);
    impl_ = std::move(new_impl);
    num_classes_ = inferred_num_classes;
    has_objness_ = inferred_has_objness;
    is_initialized_ = true;
  }
  LOGI("RknnEngine: Initialized");
  return true;
}

std::vector<Detection> RknnEngine::inferPreprocessed(
    const cv::Mat& preprocessed_image,
    const cv::Size& original_size,
    const rkapp::preprocess::LetterboxInfo& letterbox_info) {

#if RKNN_PLATFORM
  std::shared_ptr<Impl> impl;
  int input_size_snapshot = 0;
  ModelMeta model_meta_snapshot;
  DecodeParams decode_params_snapshot;
  int num_classes_snapshot = -1;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!is_initialized_) {
      LOGE("RknnEngine::inferPreprocessed: Not initialized!");
      return {};
    }
    if (!impl_) {
      LOGE("RknnEngine::inferPreprocessed: Missing implementation state");
      return {};
    }
    impl = impl_;
    input_size_snapshot = input_size_;
    model_meta_snapshot = model_meta_;
    decode_params_snapshot = decode_params_;
    num_classes_snapshot = num_classes_;
  }

  // Validate input is already preprocessed
  if (preprocessed_image.cols != input_size_snapshot || preprocessed_image.rows != input_size_snapshot) {
    LOGE("RknnEngine::inferPreprocessed: Input size mismatch. Expected ", input_size_snapshot, "x",
         input_size_snapshot, ", got ", preprocessed_image.cols, "x", preprocessed_image.rows);
    return {};
  }

  // Convert BGR to RGB (input should already be letterboxed)
  cv::Mat rgb = rkapp::preprocess::Preprocess::convertColor(preprocessed_image, cv::COLOR_BGR2RGB);
  if (!rgb.isContinuous()) {
    rgb = rgb.clone();
  }

  rknn_input in{};
  in.index = 0;
  in.type = impl->input_type;
  in.fmt = impl->input_fmt;
  if (impl->input_fmt == RKNN_TENSOR_NCHW) {
    const int h = rgb.rows;
    const int w = rgb.cols;
    const int c = rgb.channels();
    const size_t needed = static_cast<size_t>(h) * w * c;
    thread_local std::vector<uint8_t> input_local;
    input_local.resize(needed);
    const int hw = h * w;
    std::array<cv::Mat, 3> planes = {
        cv::Mat(h, w, CV_8UC1, input_local.data() + 0 * hw),
        cv::Mat(h, w, CV_8UC1, input_local.data() + 1 * hw),
        cv::Mat(h, w, CV_8UC1, input_local.data() + 2 * hw),
    };
    const int from_to[] = {0, 0, 1, 1, 2, 2};
    cv::mixChannels(&rgb, 1, planes.data(), static_cast<int>(planes.size()), from_to, 3);
    in.size = static_cast<uint32_t>(needed);
    in.buf = input_local.data();
  } else {
    in.size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
    in.buf = (void*)rgb.data;
  }

  std::unique_lock<std::mutex> lock(impl->infer_mutex);
  if (impl->shutting_down.load(std::memory_order_acquire)) {
    LOGW("RknnEngine::inferPreprocessed: Engine is shutting down");
    return {};
  }

  int ret = rknn_inputs_set(impl->ctx, 1, &in);
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_inputs_set failed: ", ret);
    return {};
  }

  ret = rknn_run(impl->ctx, nullptr);
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_run failed: ", ret);
    return {};
  }

  const size_t logits_elems = static_cast<size_t>(impl->out_elems);
  thread_local std::vector<float> logits_local;
  logits_local.resize(logits_elems);

  rknn_output out{};
  out.want_float = 1;
  out.is_prealloc = 1;
  out.buf = logits_local.data();
  out.size = logits_local.size() * sizeof(float);
  out.index = impl->out_attr.index;  // Use the selected best output index
  ret = rknn_outputs_get(impl->ctx, 1, &out, nullptr);
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_outputs_get failed: ", ret);
    return {};
  }

  int num_classes = num_classes_snapshot;
  rknn_outputs_release(impl->ctx, 1, &out);
  lock.unlock();

  auto nms_result = decode_output_and_nms(
      logits_local.data(), impl->out_n, impl->out_c, impl->out_elems, impl->out_attr.n_dims,
      impl->out_attr.dims[1], impl->out_attr.dims[2], num_classes, model_meta_snapshot,
      decode_params_snapshot, original_size, letterbox_info, &impl->dfl_layout, "RknnEngine");

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (num_classes > 0 && num_classes_ < 0) {
      num_classes_ = num_classes;
    }
  }
  return nms_result;
#else
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!is_initialized_) {
      LOGE("RknnEngine::inferPreprocessed: Not initialized!");
      return {};
    }
  }
  // Non-RKNN build: return empty
  (void)original_size;  // suppress unused warning
  (void)letterbox_info;
  return {};
#endif
}

std::vector<Detection> RknnEngine::infer(const cv::Mat& image){
#if RKNN_PLATFORM
  int input_size_snapshot = 0;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if(!is_initialized_) {
      LOGE("RknnEngine: Not initialized!");
      return {};
    }
    input_size_snapshot = input_size_;
  }

  // Letterbox + BGR->RGB conversion
  // Uses RGA hardware acceleration when available (AUTO backend)
  // Performance: ~0.3ms RGA vs ~3ms OpenCV for 1080p->640x640
  rkapp::preprocess::LetterboxInfo letterbox_info;
  cv::Mat letter =
      rkapp::preprocess::Preprocess::letterbox(image, input_size_snapshot, letterbox_info);
  if (letter.empty()) {
    LOGE("RknnEngine: Preprocess failed (empty output). Input may be invalid.");
    return {};
  }

  // Delegate to inferPreprocessed to avoid code duplication
  return inferPreprocessed(letter, image.size(), letterbox_info);
#else
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if(!is_initialized_) {
      LOGE("RknnEngine: Not initialized!");
      return {};
    }
  }
  // Non-RKNN build: return empty
  return {};
#endif
}

void RknnEngine::warmup(){
  int input_size_snapshot = 0;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if(!is_initialized_) {
      LOGW("RknnEngine: Cannot warmup - not initialized!");
      return;
    }
    input_size_snapshot = input_size_;
  }
  cv::Mat dummy(input_size_snapshot, input_size_snapshot, CV_8UC3, cv::Scalar(128,128,128));
  (void)infer(dummy);
}

std::vector<Detection> RknnEngine::inferDmaBuf(
    rkapp::common::DmaBuf& input,
    const cv::Size& original_size,
    const rkapp::preprocess::LetterboxInfo& letterbox_info) {

#if RKNN_PLATFORM
  std::shared_ptr<Impl> impl;
  int input_size_snapshot = 0;
  ModelMeta model_meta_snapshot;
  DecodeParams decode_params_snapshot;
  int num_classes_snapshot = -1;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!is_initialized_) {
      LOGE("RknnEngine::inferDmaBuf: Not initialized!");
      return {};
    }
    if (!impl_) {
      LOGE("RknnEngine::inferDmaBuf: Missing implementation state");
      return {};
    }
    impl = impl_;
    input_size_snapshot = input_size_;
    model_meta_snapshot = model_meta_;
    decode_params_snapshot = decode_params_;
    num_classes_snapshot = num_classes_;
  }

  auto fallback_to_copy = [&]() -> std::vector<Detection> {
    cv::Mat mat;
    if (!input.copyTo(mat)) {
      LOGE("RknnEngine::inferDmaBuf: Failed to copy DMA-BUF to Mat");
      return {};
    }
    cv::Mat bgr;
    try {
      switch (input.format()) {
        case rkapp::common::DmaBuf::PixelFormat::RGB888:
          cv::cvtColor(mat, bgr, cv::COLOR_RGB2BGR);
          break;
        case rkapp::common::DmaBuf::PixelFormat::BGR888:
          bgr = mat;
          break;
        case rkapp::common::DmaBuf::PixelFormat::RGBA8888:
          cv::cvtColor(mat, bgr, cv::COLOR_RGBA2BGR);
          break;
        case rkapp::common::DmaBuf::PixelFormat::BGRA8888:
          cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
          break;
        case rkapp::common::DmaBuf::PixelFormat::NV12:
          cv::cvtColor(mat, bgr, cv::COLOR_YUV2BGR_NV12);
          break;
        case rkapp::common::DmaBuf::PixelFormat::NV21:
          cv::cvtColor(mat, bgr, cv::COLOR_YUV2BGR_NV21);
          break;
        default:
          LOGE("RknnEngine::inferDmaBuf: Unsupported DMA-BUF format for fallback conversion");
          return {};
      }
    } catch (const cv::Exception& e) {
      LOGE("RknnEngine::inferDmaBuf: Fallback color conversion failed: ", e.what());
      return {};
    }
    if (bgr.empty()) {
      LOGE("RknnEngine::inferDmaBuf: Fallback conversion produced empty image");
      return {};
    }
    if (bgr.cols == input_size_snapshot && bgr.rows == input_size_snapshot) {
      return inferPreprocessed(bgr, original_size, letterbox_info);
    }
    LOGW("RknnEngine::inferDmaBuf: Fallback frame is not letterboxed (",
         bgr.cols, "x", bgr.rows, "), running full infer()");
    return infer(bgr);
  };

#if !defined(RKAPP_RKNN_DMA_FD_INPUT)
  static std::once_flag warn_once;
  std::call_once(warn_once, []() {
    LOGW("RknnEngine::inferDmaBuf: DMA-FD input disabled; enable ENABLE_RKNN_DMA_FD to use it");
  });
  return fallback_to_copy();
#endif

  // Validate input dimensions
  if (input.width() != input_size_snapshot || input.height() != input_size_snapshot) {
    LOGE("RknnEngine::inferDmaBuf: Input size mismatch. Expected ", input_size_snapshot, "x",
         input_size_snapshot, ", got ", input.width(), "x", input.height());
    return {};
  }

  if (impl->input_fmt != RKNN_TENSOR_NHWC || impl->input_type != RKNN_TENSOR_UINT8) {
    LOGW("RknnEngine::inferDmaBuf: Zero-copy requires NHWC UINT8, falling back to copy");
    return fallback_to_copy();
  }
  if (input.format() != rkapp::common::DmaBuf::PixelFormat::RGB888) {
    LOGW("RknnEngine::inferDmaBuf: Direct DMA-FD path expects RGB888 input, falling back to copy");
    return fallback_to_copy();
  }

  // Export DMA-BUF fd for RKNN
  ScopedFd dma_fd(input.exportFd());
  if (!dma_fd.valid()) {
    LOGW("RknnEngine::inferDmaBuf: Failed to export DMA-BUF fd, falling back to copy path");
    return fallback_to_copy();
  }

  std::unique_lock<std::mutex> lock(impl->infer_mutex);
  if (impl->shutting_down.load(std::memory_order_acquire)) {
    LOGW("RknnEngine::inferDmaBuf: Engine is shutting down");
    return {};
  }

  int ret = RKNN_SUCC;
#if defined(RKAPP_RKNN_IO_MEM)
  rknn_tensor_mem* input_mem = nullptr;
  rknn_mem_info mem_info{};
  mem_info.fd = dma_fd.get();
  mem_info.offset = 0;
  mem_info.size = input.size();
  input_mem = rknn_create_mem_from_fd(impl->ctx, &mem_info);
  if (!input_mem) {
    LOGW("RknnEngine::inferDmaBuf: rknn_create_mem_from_fd failed, falling back to copy path");
    dma_fd.reset();
    lock.unlock();
    return fallback_to_copy();
  }

  ret = rknn_set_io_mem(impl->ctx, input_mem, &impl->in_attr);
  if (ret != RKNN_SUCC) {
    rknn_destroy_mem(impl->ctx, input_mem);
    LOGW("RknnEngine::inferDmaBuf: rknn_set_io_mem failed (code ", ret,
         "), falling back to copy path");
    dma_fd.reset();
    lock.unlock();
    return fallback_to_copy();
  }
#else
  // Set up RKNN input with DMA-BUF fd
  // RKNN SDK 1.5.0+ supports rknn_inputs_set with fd parameter
  rknn_input in{};
  in.index = 0;
  in.type = impl->input_type;
  in.fmt = impl->input_fmt;
  in.size = static_cast<uint32_t>(input.size());

  // Use pass_through mode for DMA-BUF input (zero-copy)
  // The fd is passed via the buf pointer reinterpretation
  in.pass_through = 1;  // Enable direct buffer access
  in.buf = reinterpret_cast<void*>(static_cast<intptr_t>(dma_fd.get()));

  ret = rknn_inputs_set(impl->ctx, 1, &in);

  if (ret != RKNN_SUCC) {
    LOGW("RknnEngine::inferDmaBuf: rknn_inputs_set failed (code ", ret,
         "), falling back to copy path");
    dma_fd.reset();
    lock.unlock();
    return fallback_to_copy();
  }
#endif

  // Run inference (fd must remain valid during this call)
  ret = rknn_run(impl->ctx, nullptr);
  if (ret != RKNN_SUCC) {
#if defined(RKAPP_RKNN_IO_MEM)
    if (input_mem) {
      rknn_destroy_mem(impl->ctx, input_mem);
    }
#endif
    LOGE("RknnEngine::inferDmaBuf: rknn_run failed: ", ret);
    return {};
  }

  // Get output and decode via the shared postprocess path
  const size_t logits_elems = static_cast<size_t>(impl->out_elems);
  thread_local std::vector<float> logits_local;
  logits_local.resize(logits_elems);

  rknn_output out{};
  out.want_float = 1;
  out.is_prealloc = 1;  // Use preallocated buffer
  out.buf = logits_local.data();
  out.size = logits_local.size() * sizeof(float);
  out.index = impl->out_attr.index;  // Use the selected best output index
  ret = rknn_outputs_get(impl->ctx, 1, &out, nullptr);

#if defined(RKAPP_RKNN_IO_MEM)
  if (input_mem) {
    rknn_destroy_mem(impl->ctx, input_mem);
  }
#endif

  // NOW safe to close the fd after inference and output retrieval complete
  dma_fd.reset();

  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine::inferDmaBuf: rknn_outputs_get failed: ", ret);
    return {};
  }

  int num_classes = num_classes_snapshot;
  rknn_outputs_release(impl->ctx, 1, &out);
  lock.unlock();

  auto nms_result = decode_output_and_nms(
      logits_local.data(), impl->out_n, impl->out_c, impl->out_elems, impl->out_attr.n_dims,
      impl->out_attr.dims[1], impl->out_attr.dims[2], num_classes, model_meta_snapshot,
      decode_params_snapshot, original_size, letterbox_info, &impl->dfl_layout,
      "RknnEngine::inferDmaBuf");

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (num_classes > 0 && num_classes_ < 0) {
      num_classes_ = num_classes;
    }
  }
  return nms_result;

#else
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!is_initialized_) {
      LOGE("RknnEngine::inferDmaBuf: Not initialized!");
      return {};
    }
  }
  // Non-RKNN build: fallback to copy
  cv::Mat mat;
  if (!input.copyTo(mat)) {
    LOGE("RknnEngine::inferDmaBuf: Failed to copy DMA-BUF to Mat");
    return {};
  }
  (void)letterbox_info;  // unused in non-RKNN build
  return infer(mat);
#endif
}

void RknnEngine::release(){
  std::shared_ptr<Impl> impl;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    impl = std::move(impl_);
    is_initialized_ = false;
  }

#if RKNN_PLATFORM
  if (impl) {
    impl->shutting_down.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(impl->infer_mutex);
    if (impl->ctx) {
      rknn_destroy(impl->ctx);
      impl->ctx = 0;
    }
  }
#endif
}

int RknnEngine::getInputWidth() const {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  return input_size_;
}

int RknnEngine::getInputHeight() const {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  return input_size_;
}

void RknnEngine::setCoreMask(uint32_t core_mask) {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  core_mask_ = core_mask;
}

void RknnEngine::setDecodeParams(const DecodeParams& params) {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  decode_params_ = params;
}

int RknnEngine::num_classes() const {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  return num_classes_;
}

bool RknnEngine::has_objness() const {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  return has_objness_;
}

} // namespace rkapp::infer
