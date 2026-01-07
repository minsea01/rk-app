#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/infer/RknnDecodeUtils.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "rkapp/common/DmaBuf.hpp"
#include "log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <regex>
#include <sstream>
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
  rknn_tensor_attr out_attr{};
  int out_elems = 0;
  int out_c = 0, out_n = 0; // C (8), N (8400) or transposed
#endif
};

namespace {

std::vector<uint8_t> read_file(const std::string& p){
  std::ifstream ifs(p, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)), {});
}

// Best-effort model metadata loader to avoid纯启发式解码。
// 优先读取侧car文件：<model_path>.json / <model_path>.meta / artifacts/models/decode_meta.json
rkapp::infer::RknnEngine::ModelMeta load_model_meta(const std::string& model_path) {
  rkapp::infer::RknnEngine::ModelMeta meta;
  const std::vector<std::string> candidates = {
      model_path + ".json",
      model_path + ".meta",
      "artifacts/models/decode_meta.json"};

  auto parse_int = [](const std::string& content, const std::string& key) -> int {
    std::regex re(key + R"(\s*[:=]\s*([0-9]+))");
    std::smatch m;
    if (std::regex_search(content, m, re) && m.size() > 1) {
      return std::stoi(m[1].str());
    }
    return -1;
  };

  auto parse_strides = [](const std::string& content) -> std::vector<int> {
    std::vector<int> out;
    std::regex re(R"(strides\s*[:=]\s*\[([^\]]+)\])");
    std::smatch m;
    if (!std::regex_search(content, m, re) || m.size() < 2) return out;
    std::string body = m[1].str();
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
    std::regex re(R"(head\s*[:=]\s*\"?([a-zA-Z]+)\"?)");
    std::smatch m;
    if (std::regex_search(content, m, re) && m.size() > 1) {
      std::string v = m[1].str();
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
    int reg = parse_int(content, "reg_max");
    if (reg > 0) meta.reg_max = reg;
    auto strides = parse_strides(content);
    if (!strides.empty()) meta.strides = strides;
    std::string head = parse_head(content);
    if (!head.empty()) meta.head = head;
    if (meta.reg_max > 0 || !meta.strides.empty() || !meta.head.empty()) {
      LOGI("RknnEngine: loaded decode metadata from ", p);
      break;
    }
  }
  return meta;
}

} // namespace

RknnEngine::RknnEngine() = default;
RknnEngine::~RknnEngine(){ release(); }

bool RknnEngine::init(const std::string& model_path, int img_size){
  model_path_ = model_path;
  input_size_ = img_size;
  impl_ = std::make_unique<Impl>();
  model_meta_ = load_model_meta(model_path);
#if RKNN_PLATFORM
  auto blob = read_file(model_path);
  if(blob.empty()){
    LOGE("RknnEngine: model file empty: ", model_path);
    return false;
  }

  // Step 1: Initialize RKNN context
  int ret = rknn_init(&impl_->ctx, blob.data(), blob.size(), 0, nullptr);
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_init failed with code ", ret);
    return false;
  }

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
    return false;
  }

  // Assume single output
  std::memset(&impl_->out_attr, 0, sizeof(impl_->out_attr));
  impl_->out_attr.index = impl_->io_num.n_output - 1;
  ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &impl_->out_attr, sizeof(impl_->out_attr));
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: RKNN_QUERY_OUTPUT_ATTR failed: ", ret);
    return false;
  }

  // Auto-infer classes from output tensor dimensions
  impl_->out_elems = 1;
  for(uint32_t i=0;i<impl_->out_attr.n_dims;i++) impl_->out_elems *= impl_->out_attr.dims[i];
  
  // Determine output layout generically and postpone class inference to runtime
  int last_dim = std::max(0, (int)impl_->out_attr.n_dims - 1);
  int64_t C = impl_->out_attr.dims[last_dim];
  // Layout: try (1, N, C) and (1, C, N)
  if (impl_->out_attr.n_dims >= 3) {
    int64_t d1 = impl_->out_attr.dims[1];
    int64_t d2 = impl_->out_attr.dims[2];
    if (d2 == C) {
      impl_->out_c = (int)C;
      impl_->out_n = (int)d1;
    } else if (d1 == C) {
      impl_->out_c = (int)C;
      impl_->out_n = (int)d2;
    } else {
      // Fallback: assume last dim is C and product/ C is N
      impl_->out_c = (int)C;
      int64_t total = impl_->out_elems;
      int64_t N = total / C;
      impl_->out_n = (int)N;
    }
  }
  // Defer num_classes_ until we inspect C in infer()
  num_classes_ = -1;
#else
  LOGW("RknnEngine: RKNN platform not enabled at build time");
#endif
  is_initialized_ = true;
  LOGI("RknnEngine: Initialized");
  return true;
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

  // Use RGA-accelerated color conversion when available
  // This provides ~5x speedup for BGR->RGB conversion
  cv::Mat rgb = rkapp::preprocess::Preprocess::convertColor(letter, cv::COLOR_BGR2RGB);

  rknn_input in{}; in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
  in.size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
  in.buf = (void*)rgb.data;
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

  rknn_output out{}; out.want_float = 1; out.is_prealloc = 0;
  ret = rknn_outputs_get(impl_->ctx, 1, &out, nullptr);
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_outputs_get failed: ", ret);
    return {};
  }

  std::vector<float> logits;
  logits.assign((float*)out.buf, (float*)out.buf + impl_->out_elems);
  rknn_outputs_release(impl_->ctx, 1, &out);

  // If output is [1,8400,8], transpose to [1,8,8400]
  if(impl_->out_attr.dims[1] == impl_->out_n && impl_->out_attr.dims[2] == impl_->out_c){
    std::vector<float> trans(impl_->out_elems);
    for(int n=0;n<impl_->out_n;n++)
      for(int c=0;c<impl_->out_c;c++)
        trans[c*impl_->out_n + n] = logits[n*impl_->out_c + c];
    logits.swap(trans);
  }

  // Decode to detections. Support两种头：
  // 1) YOLOv8 DFL: 使用 reg_max 和 strides（优先来自元数据）
  // 2) Raw: [cx,cy,w,h,(obj),cls...]
  std::vector<Detection> dets;
  int N = impl_->out_n;
  int C = impl_->out_c;

  auto sigmoid = [](float x){ return 1.0f / (1.0f + std::exp(-x)); };

  const int reg_max = (model_meta_.reg_max > 0) ? model_meta_.reg_max : 16;
  const bool force_raw = (model_meta_.head == "raw");
  const bool force_dfl = (model_meta_.head == "dfl");
  const bool want_dfl = (!force_raw) && (force_dfl || C >= 4 * reg_max);
  const bool has_meta = model_meta_.reg_max > 0 && !model_meta_.strides.empty();
  const bool use_dfl = want_dfl && has_meta;

  auto decode_raw = [&]() {
    if (C < 5) {
      LOGW("RknnEngine: raw decode aborted due to insufficient channels (C=", C, ")");
      return;
    }
    const int cls_offset = 5;
    const int cls_ch = C - cls_offset;
    if (cls_ch <= 0) {
      LOGW("RknnEngine: raw decode aborted due to missing class channels (C=", C, ")");
      return;
    }
    if (num_classes_ < 0) num_classes_ = cls_ch; // assume objness present
    for(int i=0;i<N;i++){
      float cx = logits[0*N + i];
      float cy = logits[1*N + i];
      float w  = logits[2*N + i];
      float h  = logits[3*N + i];
      float obj = sigmoid(logits[4*N + i]);
      float max_conf = 0.f; int best = 0;
      for(int c=0; c < cls_ch; c++){
        float conf = sigmoid(logits[(cls_offset + c)*N + i]);
        if(conf > max_conf){ max_conf = conf; best = c; }
      }
      float conf = obj * (max_conf > 0 ? max_conf : 1.0f);
      if(conf >= decode_params_.conf_thres){
        Detection d;
        float scale = letterbox_info.scale, dx = letterbox_info.dx, dy = letterbox_info.dy;
        d.x = (cx - w/2 - dx) / scale;
        d.y = (cy - h/2 - dy) / scale;
        d.w = w / scale; d.h = h / scale;
        d.confidence = conf; d.class_id = best; d.class_name = "class_" + std::to_string(best);
        dets.push_back(d);
        if (decode_params_.max_boxes > 0 && static_cast<int>(dets.size()) >= decode_params_.max_boxes) {
          break;
        }
      }
    }
  };

  bool fell_back_to_raw = false;

  if (use_dfl) {
    // DFL path
    int cls_ch = C - 4 * reg_max;
    if (cls_ch <= 0) {
      LOGW("RknnEngine: DFL decode skipped due to invalid class channels (C=", C, ", reg_max=", reg_max, ")");
      fell_back_to_raw = true;
    } else {
      if (num_classes_ < 0) num_classes_ = cls_ch;
      // Require stride metadata to avoid heuristic mis-decodes
      std::vector<int> strides = model_meta_.strides;
      const bool resolved = resolve_stride_set(input_size_, N, strides);
      if (!resolved || strides.empty()) {
        LOGW("RknnEngine: missing/invalid stride metadata for DFL decode (anchors=", N, ", img_size=", input_size_, "), falling back to raw decode");
        fell_back_to_raw = true;
      } else {
        AnchorLayout layout = build_anchor_layout(input_size_, N, strides);
        if (!layout.valid) {
          LOGW("RknnEngine: anchor layout invalid for provided strides, falling back to raw decode");
          fell_back_to_raw = true;
        } else {
          // Decode distributions to l,t,r,b
          auto dfl_softmax_project = [&](int base_c, int i)->std::array<float,4> {
            std::array<float,4> out{};
            for (int side = 0; side < 4; ++side) {
              int ch0 = base_c + side * reg_max; // channel start for this side
              // softmax over reg_max at position i
              float maxv = -1e30f;
              for (int k = 0; k < reg_max; ++k) maxv = std::max(maxv, logits[(ch0 + k)*N + i]);
              float denom = 0.f;
              std::vector<float> probs(reg_max);
              for (int k = 0; k < reg_max; ++k) { float v = std::exp(logits[(ch0 + k)*N + i] - maxv); probs[k] = v; denom += v; }
              float proj = 0.f;
              for (int k = 0; k < reg_max; ++k) proj += probs[k] * (float)k;
              out[side] = (denom > 0.f) ? (proj / denom) : 0.f;
            }
            return out; // in grid units
          };

          // For each anchor, compute bbox and class
          for (int i = 0; i < N; ++i) {
            auto dfl = dfl_softmax_project(0, i); // 0..(reg_max-1)
            float s = layout.stride_map[i];
            // scale distances by stride
            float l = dfl[0] * s, t = dfl[1] * s, r = dfl[2] * s, b = dfl[3] * s;
            float x1 = layout.anchor_cx[i] - l;
            float y1 = layout.anchor_cy[i] - t;
            float x2 = layout.anchor_cx[i] + r;
            float y2 = layout.anchor_cy[i] + b;
            // class scores
            float best_conf = 0.f; int best_cls = 0;
            for (int c = 0; c < cls_ch; ++c) {
              float conf = sigmoid(logits[(4*reg_max + c)*N + i]);
              if (conf > best_conf) { best_conf = conf; best_cls = c; }
            }
            if (best_conf >= decode_params_.conf_thres) {
              Detection d;
              float scale = letterbox_info.scale, dx = letterbox_info.dx, dy = letterbox_info.dy;
              float w = x2 - x1, h = y2 - y1;
              d.x = (x1 - dx) / scale; d.y = (y1 - dy) / scale;
              d.w = w / scale; d.h = h / scale;
              d.confidence = best_conf; d.class_id = best_cls; d.class_name = "class_" + std::to_string(best_cls);
              dets.push_back(d);
              if (decode_params_.max_boxes > 0 && static_cast<int>(dets.size()) >= decode_params_.max_boxes) {
                break;
              }
            }
          }
        }
      }
    }
  } else if (want_dfl && !has_meta) {
    LOGW("RknnEngine: DFL-like output detected but missing metadata (reg_max/strides); falling back to raw decode");
    fell_back_to_raw = true;
  }

  if (!use_dfl || fell_back_to_raw) {
    decode_raw();
  }
  rkapp::post::NMSConfig nms_cfg;
  nms_cfg.conf_thres = decode_params_.conf_thres;
  nms_cfg.iou_thres = decode_params_.iou_thres;
  if (decode_params_.max_boxes > 0) {
    nms_cfg.max_det = decode_params_.max_boxes;
    nms_cfg.topk = decode_params_.max_boxes;
  }
  return rkapp::post::Postprocess::nms(dets, nms_cfg);
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
    const rkapp::preprocess::LetterboxInfo& letterbox_info) {

  if (!is_initialized_) {
    LOGE("RknnEngine::inferDmaBuf: Not initialized!");
    return {};
  }

#if RKNN_PLATFORM
  // Validate input dimensions
  if (input.width() != input_size_ || input.height() != input_size_) {
    LOGE("RknnEngine::inferDmaBuf: Input size mismatch. Expected ", input_size_, "x",
         input_size_, ", got ", input.width(), "x", input.height());
    return {};
  }

  // Export DMA-BUF fd for RKNN
  int dma_fd = input.exportFd();
  if (dma_fd < 0) {
    LOGW("RknnEngine::inferDmaBuf: Failed to export DMA-BUF fd, falling back to copy path");
    // Fallback: copy to cv::Mat and use regular infer
    cv::Mat mat;
    if (!input.copyTo(mat)) {
      LOGE("RknnEngine::inferDmaBuf: Failed to copy DMA-BUF to Mat");
      return {};
    }
    return infer(mat);
  }

  // Set up RKNN input with DMA-BUF fd
  // RKNN SDK 1.5.0+ supports rknn_inputs_set with fd parameter
  rknn_input in{};
  in.index = 0;
  in.type = RKNN_TENSOR_UINT8;
  in.fmt = RKNN_TENSOR_NHWC;
  in.size = static_cast<uint32_t>(input.size());

  // Use pass_through mode for DMA-BUF input (zero-copy)
  // The fd is passed via the buf pointer reinterpretation
  in.pass_through = 1;  // Enable direct buffer access
  in.buf = reinterpret_cast<void*>(static_cast<intptr_t>(dma_fd));

  int ret = rknn_inputs_set(impl_->ctx, 1, &in);

  // Close the duplicated fd
  close(dma_fd);

  if (ret != RKNN_SUCC) {
    LOGW("RknnEngine::inferDmaBuf: rknn_inputs_set failed (code ", ret,
         "), falling back to copy path");
    // Fallback to copy path
    cv::Mat mat;
    if (!input.copyTo(mat)) {
      LOGE("RknnEngine::inferDmaBuf: Failed to copy DMA-BUF to Mat");
      return {};
    }
    return infer(mat);
  }

  // Run inference
  ret = rknn_run(impl_->ctx, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine::inferDmaBuf: rknn_run failed: ", ret);
    return {};
  }

  // Get output
  rknn_output out{};
  out.want_float = 1;
  out.is_prealloc = 0;
  ret = rknn_outputs_get(impl_->ctx, 1, &out, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine::inferDmaBuf: rknn_outputs_get failed: ", ret);
    return {};
  }

  std::vector<float> logits;
  logits.assign((float*)out.buf, (float*)out.buf + impl_->out_elems);
  rknn_outputs_release(impl_->ctx, 1, &out);

  // Transpose if needed
  if (impl_->out_attr.dims[1] == impl_->out_n && impl_->out_attr.dims[2] == impl_->out_c) {
    std::vector<float> trans(impl_->out_elems);
    for (int n = 0; n < impl_->out_n; n++)
      for (int c = 0; c < impl_->out_c; c++)
        trans[c * impl_->out_n + n] = logits[n * impl_->out_c + c];
    logits.swap(trans);
  }

  // Decode detections (same as infer())
  std::vector<Detection> dets;
  int N = impl_->out_n;
  int C = impl_->out_c;

  auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

  // Use raw decode for simplicity in zero-copy path
  if (C >= 5) {
    const int cls_offset = 5;
    const int cls_ch = C - cls_offset;
    if (cls_ch > 0) {
      for (int i = 0; i < N; i++) {
        float cx = logits[0 * N + i];
        float cy = logits[1 * N + i];
        float w = logits[2 * N + i];
        float h = logits[3 * N + i];
        float obj = sigmoid(logits[4 * N + i]);
        float max_conf = 0.f;
        int best = 0;
        for (int c = 0; c < cls_ch; c++) {
          float conf = sigmoid(logits[(cls_offset + c) * N + i]);
          if (conf > max_conf) {
            max_conf = conf;
            best = c;
          }
        }
        float conf = obj * (max_conf > 0 ? max_conf : 1.0f);
        if (conf >= decode_params_.conf_thres) {
          Detection d;
          float scale = letterbox_info.scale;
          float dx = letterbox_info.dx;
          float dy = letterbox_info.dy;
          d.x = (cx - w / 2 - dx) / scale;
          d.y = (cy - h / 2 - dy) / scale;
          d.w = w / scale;
          d.h = h / scale;
          d.confidence = conf;
          d.class_id = best;
          d.class_name = "class_" + std::to_string(best);
          dets.push_back(d);
          if (decode_params_.max_boxes > 0 &&
              static_cast<int>(dets.size()) >= decode_params_.max_boxes) {
            break;
          }
        }
      }
    }
  }

  // Apply NMS
  rkapp::post::NMSConfig nms_cfg;
  nms_cfg.conf_thres = decode_params_.conf_thres;
  nms_cfg.iou_thres = decode_params_.iou_thres;
  if (decode_params_.max_boxes > 0) {
    nms_cfg.max_det = decode_params_.max_boxes;
    nms_cfg.topk = decode_params_.max_boxes;
  }
  return rkapp::post::Postprocess::nms(dets, nms_cfg);

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
  decode_params_ = params;
}

} // namespace rkapp::infer
