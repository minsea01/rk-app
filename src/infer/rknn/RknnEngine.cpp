#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/infer/RknnDecodeUtils.hpp"
#include "rkapp/infer/RknnDecodeOptimized.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "rkapp/common/DmaBuf.hpp"
#include "log.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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
  std::mutex infer_mutex;
#endif
};

namespace {

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

std::vector<uint8_t> read_file(const std::string& p){
  std::ifstream ifs(p, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)), {});
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

  // CRITICAL: Guard against reg_max > 32 buffer overflow
  constexpr int MAX_REG_MAX = 32;
  if (use_dfl && reg_max > MAX_REG_MAX) {
    LOGE(log_tag, ": reg_max=", reg_max, " exceeds buffer size (", MAX_REG_MAX, ")");
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
    std::array<float, MAX_REG_MAX> probs_buf{};
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

} // namespace

RknnEngine::RknnEngine() = default;
RknnEngine::~RknnEngine(){ release(); }

bool RknnEngine::init(const std::string& model_path, int img_size){
  if (impl_ || is_initialized_) {
    release();
  }
  model_path_ = model_path;
  input_size_ = img_size;
  impl_ = std::make_unique<Impl>();
  model_meta_ = load_model_meta(model_path);
#if RKNN_PLATFORM
  bool ctx_ready = false;
  auto cleanup = [&]() {
    if (ctx_ready && impl_ && impl_->ctx) {
      rknn_destroy(impl_->ctx);
      impl_->ctx = 0;
    }
    impl_.reset();
    is_initialized_ = false;
  };

  auto blob = read_file(model_path);
  if(blob.empty()){
    LOGE("RknnEngine: model file empty: ", model_path);
    cleanup();
    return false;
  }

  // Step 1: Initialize RKNN context
  int ret = rknn_init(&impl_->ctx, blob.data(), blob.size(), 0, nullptr);
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
  if (core_mask_ != 0) {
    // Map our core_mask_ to rknn_core_mask enum
    rknn_core_mask npu_core_mask = RKNN_NPU_CORE_AUTO;
    switch (core_mask_) {
      case 0x1: npu_core_mask = RKNN_NPU_CORE_0; break;
      case 0x2: npu_core_mask = RKNN_NPU_CORE_1; break;
      case 0x4: npu_core_mask = RKNN_NPU_CORE_2; break;
      case 0x3: npu_core_mask = RKNN_NPU_CORE_0_1; break;
      case 0x7: npu_core_mask = RKNN_NPU_CORE_0_1_2; break;
      default:
        // For any other mask, use all three cores for maximum throughput
        LOGW("RknnEngine: Unknown core_mask 0x", std::hex, core_mask_, std::dec,
             ", defaulting to RKNN_NPU_CORE_0_1_2 (6 TOPS)");
        npu_core_mask = RKNN_NPU_CORE_0_1_2;
        break;
    }

    int mask_ret = rknn_set_core_mask(impl_->ctx, npu_core_mask);
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

  ret = rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &impl_->io_num, sizeof(impl_->io_num));
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: RKNN_QUERY_IN_OUT_NUM failed: ", ret);
    cleanup();
    return false;
  }

  if (impl_->io_num.n_input == 0) {
    LOGE("RknnEngine: No inputs detected (n_input=", impl_->io_num.n_input, ")");
    cleanup();
    return false;
  }

  std::memset(&impl_->in_attr, 0, sizeof(impl_->in_attr));
  impl_->in_attr.index = 0;
  ret = rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &impl_->in_attr, sizeof(impl_->in_attr));
  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine: RKNN_QUERY_INPUT_ATTR failed: ", ret);
    cleanup();
    return false;
  }

  impl_->input_fmt = impl_->in_attr.fmt;
  impl_->input_type = impl_->in_attr.type;
  if (impl_->input_fmt != RKNN_TENSOR_NHWC && impl_->input_fmt != RKNN_TENSOR_NCHW) {
    LOGE("RknnEngine: Unsupported input format (fmt=", impl_->input_fmt, ")");
    cleanup();
    return false;
  }
  if (impl_->input_type != RKNN_TENSOR_UINT8) {
    LOGE("RknnEngine: Unsupported input type (type=", impl_->input_type, ")");
    cleanup();
    return false;
  }

  int input_c = 0;
  int input_h = 0;
  int input_w = 0;
  if (impl_->input_fmt == RKNN_TENSOR_NHWC) {
    if (impl_->in_attr.n_dims == 4) {
      input_h = static_cast<int>(impl_->in_attr.dims[1]);
      input_w = static_cast<int>(impl_->in_attr.dims[2]);
      input_c = static_cast<int>(impl_->in_attr.dims[3]);
    } else if (impl_->in_attr.n_dims == 3) {
      input_h = static_cast<int>(impl_->in_attr.dims[0]);
      input_w = static_cast<int>(impl_->in_attr.dims[1]);
      input_c = static_cast<int>(impl_->in_attr.dims[2]);
    }
  } else if (impl_->input_fmt == RKNN_TENSOR_NCHW) {
    if (impl_->in_attr.n_dims == 4) {
      input_c = static_cast<int>(impl_->in_attr.dims[1]);
      input_h = static_cast<int>(impl_->in_attr.dims[2]);
      input_w = static_cast<int>(impl_->in_attr.dims[3]);
    } else if (impl_->in_attr.n_dims == 3) {
      input_c = static_cast<int>(impl_->in_attr.dims[0]);
      input_h = static_cast<int>(impl_->in_attr.dims[1]);
      input_w = static_cast<int>(impl_->in_attr.dims[2]);
    }
  }

  if (input_c > 0 && input_c != 3) {
    LOGE("RknnEngine: Unsupported input channels (C=", input_c, "), expected 3");
    cleanup();
    return false;
  }
  if (input_h > 0 && input_w > 0 &&
      (input_h != input_size_ || input_w != input_size_)) {
    LOGW("RknnEngine: Input size mismatch with model (model=", input_w, "x",
         input_h, ", cfg=", input_size_, "x", input_size_, ")");
  }

  // Select output tensor (metadata required for multi-output models)
  int best_output_idx = 0;
  rknn_tensor_attr best_attr{};

  auto query_output_attr = [&](uint32_t idx, rknn_tensor_attr& attr) -> bool {
    std::memset(&attr, 0, sizeof(attr));
    attr.index = idx;
    if (rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC) {
      LOGW("RknnEngine: RKNN_QUERY_OUTPUT_ATTR failed for output ", idx);
      return false;
    }
    return true;
  };

  if (impl_->io_num.n_output == 1) {
    best_output_idx = 0;
    if (!query_output_attr(0, best_attr)) {
      cleanup();
      return false;
    }
    LOGI("RknnEngine: Single output detected, using index 0");
  } else if (impl_->io_num.n_output > 1) {
    if (model_meta_.output_index < 0) {
      LOGE("RknnEngine: Multiple outputs require metadata output_index");
      cleanup();
      return false;
    }
    if (model_meta_.output_index >= static_cast<int>(impl_->io_num.n_output)) {
      LOGE("RknnEngine: output_index out of range (", model_meta_.output_index,
           "), n_output=", impl_->io_num.n_output);
      cleanup();
      return false;
    }
    best_output_idx = model_meta_.output_index;
    if (!query_output_attr(best_output_idx, best_attr)) {
      cleanup();
      return false;
    }
    LOGI("RknnEngine: Using output index from metadata: ", best_output_idx);
  } else {
    LOGE("RknnEngine: No outputs detected (n_output=", impl_->io_num.n_output, ")");
    cleanup();
    return false;
  }

  impl_->out_attr = best_attr;

  // Auto-infer classes from output tensor dimensions
  impl_->out_elems = 1;
  for(uint32_t i=0;i<impl_->out_attr.n_dims;i++) impl_->out_elems *= impl_->out_attr.dims[i];
  
  if (model_meta_.head != "raw" && model_meta_.head != "dfl") {
    LOGE("RknnEngine: Missing or invalid head metadata (expected raw/dfl)");
    cleanup();
    return false;
  }
  if (model_meta_.num_classes <= 0) {
    LOGE("RknnEngine: Missing or invalid num_classes metadata");
    cleanup();
    return false;
  }

  int expected_c = 0;
  if (model_meta_.head == "dfl") {
    if (model_meta_.reg_max <= 0 || model_meta_.strides.empty()) {
      LOGE("RknnEngine: DFL decode requires reg_max and strides metadata");
      cleanup();
      return false;
    }
    expected_c = 4 * model_meta_.reg_max + model_meta_.num_classes;
  } else {
    if (model_meta_.has_objectness < 0) {
      LOGE("RknnEngine: RAW decode requires has_objectness metadata");
      cleanup();
      return false;
    }
    expected_c = 4 + model_meta_.num_classes + (model_meta_.has_objectness == 1 ? 1 : 0);
  }

  if (impl_->out_attr.n_dims != 3) {
    LOGE("RknnEngine: Unsupported output dimensions: n_dims=", impl_->out_attr.n_dims);
    cleanup();
    return false;
  }

  int64_t d1 = impl_->out_attr.dims[1];
  int64_t d2 = impl_->out_attr.dims[2];
  if (d1 == expected_c && d2 != expected_c) {
    impl_->out_c = expected_c;
    impl_->out_n = static_cast<int>(d2);
    LOGI("RknnEngine: Detected channels_first layout [1, ", d1, ", ", d2, "]");
  } else if (d2 == expected_c && d1 != expected_c) {
    impl_->out_c = expected_c;
    impl_->out_n = static_cast<int>(d1);
    LOGI("RknnEngine: Detected channels_last layout [1, ", d1, ", ", d2, "], will transpose");
  } else {
    LOGE("RknnEngine: Output layout mismatch (dims=[1,", d1, ",", d2,
         "], expected C=", expected_c, ")");
    cleanup();
    return false;
  }

  // Seed num_classes_ from metadata when provided
  num_classes_ = (model_meta_.num_classes > 0) ? model_meta_.num_classes : -1;
  if (model_meta_.has_objectness >= 0) {
    has_objness_ = (model_meta_.has_objectness == 1);
  }

  const bool expect_dfl = (model_meta_.head == "dfl");
  if (expect_dfl) {
    if (model_meta_.reg_max <= 0 || model_meta_.strides.empty()) {
      LOGE("RknnEngine: DFL decode requires reg_max and strides metadata");
      cleanup();
      return false;
    }
    constexpr int kMaxRegMax = 32;
    if (model_meta_.reg_max > kMaxRegMax) {
      LOGE("RknnEngine: reg_max=", model_meta_.reg_max, " exceeds max (", kMaxRegMax, ")");
      cleanup();
      return false;
    }
    AnchorLayout layout = build_anchor_layout(input_size_, impl_->out_n, model_meta_.strides);
    if (!layout.valid) {
      LOGE("RknnEngine: anchor layout invalid for provided strides");
      cleanup();
      return false;
    }
    impl_->dfl_layout = std::move(layout);
  }

  LOGI("RknnEngine: Output elements per inference: ", impl_->out_elems);
#else
  LOGW("RknnEngine: RKNN platform not enabled at build time");
#endif
  is_initialized_ = true;
  LOGI("RknnEngine: Initialized");
  return true;
}

std::vector<Detection> RknnEngine::inferPreprocessed(
    const cv::Mat& preprocessed_image,
    const cv::Size& original_size,
    const rkapp::preprocess::LetterboxInfo& letterbox_info) {

  if (!is_initialized_) {
    LOGE("RknnEngine::inferPreprocessed: Not initialized!");
    return {};
  }

#if RKNN_PLATFORM
  if (!impl_) {
    LOGE("RknnEngine::inferPreprocessed: Missing implementation state");
    return {};
  }
  // Validate input is already preprocessed
  if (preprocessed_image.cols != input_size_ || preprocessed_image.rows != input_size_) {
    LOGE("RknnEngine::inferPreprocessed: Input size mismatch. Expected ", input_size_, "x",
         input_size_, ", got ", preprocessed_image.cols, "x", preprocessed_image.rows);
    return {};
  }

  // Convert BGR to RGB (input should already be letterboxed)
  cv::Mat rgb = rkapp::preprocess::Preprocess::convertColor(preprocessed_image, cv::COLOR_BGR2RGB);
  if (!rgb.isContinuous()) {
    rgb = rgb.clone();
  }

  rknn_input in{};
  in.index = 0;
  in.type = impl_->input_type;
  in.fmt = impl_->input_fmt;
  if (impl_->input_fmt == RKNN_TENSOR_NCHW) {
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
  std::unique_lock<std::mutex> lock(impl_->infer_mutex);
  int ret = rknn_inputs_set(impl_->ctx, 1, &in);
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_inputs_set failed: ", ret);
    return {};
  }

  ret = rknn_run(impl_->ctx, nullptr);
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_run failed: ", ret);
    return {};
  }

  const size_t logits_elems = static_cast<size_t>(impl_->out_elems);
  thread_local std::vector<float> logits_local;
  logits_local.resize(logits_elems);

  rknn_output out{};
  out.want_float = 1;
  out.is_prealloc = 1;
  out.buf = logits_local.data();
  out.size = logits_local.size() * sizeof(float);
  out.index = impl_->out_attr.index;  // Use the selected best output index
  ret = rknn_outputs_get(impl_->ctx, 1, &out, nullptr);
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_outputs_get failed: ", ret);
    return {};
  }

  DecodeParams decode_params = decode_params_;
  int num_classes = num_classes_;
  rknn_outputs_release(impl_->ctx, 1, &out);
  lock.unlock();

  const float* logits = logits_local.data();
  thread_local std::vector<float> transpose_local;
  if (impl_->out_attr.n_dims >= 3 &&
      impl_->out_attr.dims[1] == impl_->out_n &&
      impl_->out_attr.dims[2] == impl_->out_c) {
    transpose_local.resize(logits_elems);
    for (int n = 0; n < impl_->out_n; n++) {
      for (int c = 0; c < impl_->out_c; c++) {
        transpose_local[c * impl_->out_n + n] = logits[n * impl_->out_c + c];
      }
    }
    logits = transpose_local.data();
  }

  auto nms_result = decode_and_postprocess(
      logits, impl_->out_n, impl_->out_c, impl_->out_elems, num_classes, model_meta_,
      decode_params, original_size, letterbox_info, &impl_->dfl_layout, "RknnEngine");

  if (num_classes > 0) {
    std::lock_guard<std::mutex> update_lock(impl_->infer_mutex);
    if (num_classes_ < 0) num_classes_ = num_classes;
  }
  return nms_result;
#else
  // Non-RKNN build: return empty
  (void)original_size;  // suppress unused warning
  (void)letterbox_info;
  return {};
#endif
}

std::vector<Detection> RknnEngine::infer(const cv::Mat& image){
  if(!is_initialized_) {
    LOGE("RknnEngine: Not initialized!");
    return {};
  }

#if RKNN_PLATFORM
  // Letterbox + BGR->RGB conversion
  // Uses RGA hardware acceleration when available (AUTO backend)
  // Performance: ~0.3ms RGA vs ~3ms OpenCV for 1080p->640x640
  rkapp::preprocess::LetterboxInfo letterbox_info;
  cv::Mat letter = rkapp::preprocess::Preprocess::letterbox(image, input_size_, letterbox_info);
  if (letter.empty()) {
    LOGE("RknnEngine: Preprocess failed (empty output). Input may be invalid.");
    return {};
  }

  // Delegate to inferPreprocessed to avoid code duplication
  return inferPreprocessed(letter, image.size(), letterbox_info);
#else
  // Non-RKNN build: return empty
  return {};
#endif
}

void RknnEngine::warmup(){
  if(!is_initialized_) { LOGW("RknnEngine: Cannot warmup - not initialized!"); return; }
  cv::Mat dummy(input_size_, input_size_, CV_8UC3, cv::Scalar(128,128,128));
  (void)infer(dummy);
}

std::vector<Detection> RknnEngine::inferDmaBuf(
    rkapp::common::DmaBuf& input,
    const cv::Size& original_size,
    const rkapp::preprocess::LetterboxInfo& letterbox_info) {

  if (!is_initialized_) {
    LOGE("RknnEngine::inferDmaBuf: Not initialized!");
    return {};
  }

#if RKNN_PLATFORM
  if (!impl_) {
    LOGE("RknnEngine::inferDmaBuf: Missing implementation state");
    return {};
  }

  auto fallback_to_copy = [&]() -> std::vector<Detection> {
    cv::Mat mat;
    if (!input.copyTo(mat)) {
      LOGE("RknnEngine::inferDmaBuf: Failed to copy DMA-BUF to Mat");
      return {};
    }
    cv::Mat bgr;
    cv::cvtColor(mat, bgr, cv::COLOR_RGB2BGR);
    return inferPreprocessed(bgr, original_size, letterbox_info);
  };

#if !defined(RKAPP_RKNN_DMA_FD_INPUT)
  static std::once_flag warn_once;
  std::call_once(warn_once, []() {
    LOGW("RknnEngine::inferDmaBuf: DMA-FD input disabled; enable ENABLE_RKNN_DMA_FD to use it");
  });
  return fallback_to_copy();
#endif

  // Validate input dimensions
  if (input.width() != input_size_ || input.height() != input_size_) {
    LOGE("RknnEngine::inferDmaBuf: Input size mismatch. Expected ", input_size_, "x",
         input_size_, ", got ", input.width(), "x", input.height());
    return {};
  }

  if (impl_->input_fmt != RKNN_TENSOR_NHWC || impl_->input_type != RKNN_TENSOR_UINT8) {
    LOGW("RknnEngine::inferDmaBuf: Zero-copy requires NHWC UINT8, falling back to copy");
    return fallback_to_copy();
  }

  // Export DMA-BUF fd for RKNN
  ScopedFd dma_fd(input.exportFd());
  if (!dma_fd.valid()) {
    LOGW("RknnEngine::inferDmaBuf: Failed to export DMA-BUF fd, falling back to copy path");
    return fallback_to_copy();
  }

  std::unique_lock<std::mutex> lock(impl_->infer_mutex);

  int ret = RKNN_SUCC;
#if defined(RKAPP_RKNN_IO_MEM)
  rknn_tensor_mem* input_mem = nullptr;
  rknn_mem_info mem_info{};
  mem_info.fd = dma_fd.get();
  mem_info.offset = 0;
  mem_info.size = input.size();
  input_mem = rknn_create_mem_from_fd(impl_->ctx, &mem_info);
  if (!input_mem) {
    LOGW("RknnEngine::inferDmaBuf: rknn_create_mem_from_fd failed, falling back to copy path");
    dma_fd.reset();
    lock.unlock();
    return fallback_to_copy();
  }

  ret = rknn_set_io_mem(impl_->ctx, input_mem, &impl_->in_attr);
  if (ret != RKNN_SUCC) {
    rknn_destroy_mem(impl_->ctx, input_mem);
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
  in.type = impl_->input_type;
  in.fmt = impl_->input_fmt;
  in.size = static_cast<uint32_t>(input.size());

  // Use pass_through mode for DMA-BUF input (zero-copy)
  // The fd is passed via the buf pointer reinterpretation
  in.pass_through = 1;  // Enable direct buffer access
  in.buf = reinterpret_cast<void*>(static_cast<intptr_t>(dma_fd.get()));

  ret = rknn_inputs_set(impl_->ctx, 1, &in);

  if (ret != RKNN_SUCC) {
    LOGW("RknnEngine::inferDmaBuf: rknn_inputs_set failed (code ", ret,
         "), falling back to copy path");
    dma_fd.reset();
    lock.unlock();
    return fallback_to_copy();
  }
#endif

  // Run inference (fd must remain valid during this call)
  ret = rknn_run(impl_->ctx, nullptr);
  if (ret != RKNN_SUCC) {
#if defined(RKAPP_RKNN_IO_MEM)
    if (input_mem) {
      rknn_destroy_mem(impl_->ctx, input_mem);
    }
#endif
    LOGE("RknnEngine::inferDmaBuf: rknn_run failed: ", ret);
    return {};
  }

  // Get output and decode via the shared postprocess path
  const size_t logits_elems = static_cast<size_t>(impl_->out_elems);
  thread_local std::vector<float> logits_local;
  logits_local.resize(logits_elems);

  rknn_output out{};
  out.want_float = 1;
  out.is_prealloc = 1;  // Use preallocated buffer
  out.buf = logits_local.data();
  out.size = logits_local.size() * sizeof(float);
  out.index = impl_->out_attr.index;  // Use the selected best output index
  ret = rknn_outputs_get(impl_->ctx, 1, &out, nullptr);

#if defined(RKAPP_RKNN_IO_MEM)
  if (input_mem) {
    rknn_destroy_mem(impl_->ctx, input_mem);
  }
#endif

  // NOW safe to close the fd after inference and output retrieval complete
  dma_fd.reset();

  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine::inferDmaBuf: rknn_outputs_get failed: ", ret);
    return {};
  }

  DecodeParams decode_params = decode_params_;
  int num_classes = num_classes_;
  rknn_outputs_release(impl_->ctx, 1, &out);
  lock.unlock();

  const float* logits = logits_local.data();
  thread_local std::vector<float> transpose_local;
  if (impl_->out_attr.n_dims >= 3 &&
      impl_->out_attr.dims[1] == impl_->out_n &&
      impl_->out_attr.dims[2] == impl_->out_c) {
    transpose_local.resize(logits_elems);
    for (int n = 0; n < impl_->out_n; n++) {
      for (int c = 0; c < impl_->out_c; c++) {
        transpose_local[c * impl_->out_n + n] = logits[n * impl_->out_c + c];
      }
    }
    logits = transpose_local.data();
  }

  auto nms_result = decode_and_postprocess(
      logits, impl_->out_n, impl_->out_c, impl_->out_elems, num_classes, model_meta_,
      decode_params, original_size, letterbox_info, &impl_->dfl_layout,
      "RknnEngine::inferDmaBuf");

  if (num_classes > 0) {
    std::lock_guard<std::mutex> update_lock(impl_->infer_mutex);
    if (num_classes_ < 0) num_classes_ = num_classes;
  }
  return nms_result;

#else
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
#if RKNN_PLATFORM
  if(impl_ && impl_->ctx){
    rknn_destroy(impl_->ctx);
  }
#endif
  impl_.reset();
  is_initialized_ = false;
}

int RknnEngine::getInputWidth() const { return input_size_; }
int RknnEngine::getInputHeight() const { return input_size_; }

void RknnEngine::setDecodeParams(const DecodeParams& params) {
#if RKNN_PLATFORM
  if (impl_) {
    std::lock_guard<std::mutex> lock(impl_->infer_mutex);
    decode_params_ = params;
    return;
  }
#endif
  decode_params_ = params;
}

} // namespace rkapp::infer
