#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/infer/RknnDecodeUtils.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <regex>
#include <sstream>

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
  // 支持可选核心掩码：若SDK支持rknn_init_ext可使用；否则回退rknn_init
  int ret = RKNN_SUCC;
#if defined(RKNN_API_VERSION) && (RKNN_API_VERSION >= 0x010400) // 假设1.4.0+提供扩展
  // 伪代码：如存在rknn_init_with_core_mask之类扩展API，可在此调用
  // ret = rknn_init_with_core_mask(&impl_->ctx, blob.data(), blob.size(), core_mask_);
  // 暂时回退
  (void)core_mask_;
  ret = rknn_init(&impl_->ctx, blob.data(), blob.size(), 0, nullptr);
#else
  ret = rknn_init(&impl_->ctx, blob.data(), blob.size(), 0, nullptr);
#endif
  if(ret != RKNN_SUCC){
    LOGE("RknnEngine: rknn_init failed with code ", ret);
    return false;
  }

  // Try to configure NPU core mask if supported by SDK and user specified a non-zero mask
#if defined(RKNN_CONFIG_NPU_CORE_MASK)
  if (core_mask_ != 0) {
    int cm = static_cast<int>(core_mask_);
    int cfg_ret = rknn_config(impl_->ctx, RKNN_CONFIG_NPU_CORE_MASK, &cm);
    if (cfg_ret != RKNN_SUCC) {
      LOGW("RknnEngine: rknn_config(NPU_CORE_MASK) failed: ", cfg_ret);
    } else {
      LOGI("RknnEngine: NPU core mask configured: 0x", std::hex, core_mask_, std::dec);
    }
  }
#endif

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
  // Letterbox + RGB (uint8)
  rkapp::preprocess::LetterboxInfo letterbox_info;
  cv::Mat letter = rkapp::preprocess::Preprocess::letterbox(image, input_size_, letterbox_info);
  cv::Mat rgb;
  cv::cvtColor(letter, rgb, cv::COLOR_BGR2RGB);

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
  const bool use_dfl = (!force_raw) && (force_dfl || C >= 4 * reg_max);

  if (use_dfl) {
    // DFL path
    int cls_ch = C - 4 * reg_max;
    if (num_classes_ < 0) num_classes_ = cls_ch;
    // Prepare strides and per-anchor stride map for imgsz=input_size_
    std::vector<int> strides = model_meta_.strides.empty() ? std::vector<int>{8, 16, 32} : model_meta_.strides;
    const bool resolved = resolve_stride_set(input_size_, N, strides);
    AnchorLayout layout = build_anchor_layout(input_size_, N, strides);
    if (!layout.valid || !resolved) {
      LOGW("RknnEngine: Anchor layout derived heuristically; results may be less accurate (N=", N, ")");
    }

    // Decode distributions to l,t,r,b
    auto dfl_softmax_project = [&](int base_c, int i)->std::array<float,4> {
      std::array<float,4> out{};
      for (int side = 0; side < 4; ++side) {
        int ch0 = base_c + side * reg_max; // channel start for this side
        // softmax over reg_max at position i
        float maxv = -1e30f;
        for (int k = 0; k < reg_max; ++k) maxv = std::max(maxv, logits[(ch0 + k)*N + i]);
        float denom = 0.f;
        float probs[16];
        for (int k = 0; k < reg_max; ++k) { float v = std::exp(logits[(ch0 + k)*N + i] - maxv); probs[k] = v; denom += v; }
        float proj = 0.f;
        for (int k = 0; k < reg_max; ++k) proj += probs[k] * (float)k;
        out[side] = (denom > 0.f) ? (proj / denom) : 0.f;
      }
      return out; // in grid units
    };

    // For each anchor, compute bbox and class
    for (int i = 0; i < N; ++i) {
      auto dfl = dfl_softmax_project(0, i); // 0..63
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
  } else {
    // Raw path
    if (num_classes_ < 0) num_classes_ = std::max(0, C - 5); // assume objness present; if not, handled by loop
    for(int i=0;i<N;i++){
      float cx = logits[0*N + i];
      float cy = logits[1*N + i];
      float w  = logits[2*N + i];
      float h  = logits[3*N + i];
      int cls_offset = 4;
      float obj = 1.0f;
      if (C >= 5) { obj = sigmoid(logits[4*N + i]); cls_offset = 5; }
      float max_conf = 0.f; int best = 0;
      for(int c=0; c < (C - cls_offset); c++){
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
