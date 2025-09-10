#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>

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

static std::vector<uint8_t> read_file(const std::string& p){
  std::ifstream ifs(p, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)), {});
}

RknnEngine::RknnEngine() = default;
RknnEngine::~RknnEngine(){ release(); }

bool RknnEngine::init(const std::string& model_path, int img_size){
  model_path_ = model_path;
  input_size_ = img_size;
  impl_ = std::make_unique<Impl>();
#if RKNN_PLATFORM
  auto blob = read_file(model_path);
  if(blob.empty()){ std::cerr << "RknnEngine: model empty\n"; return false; }
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
  if(ret != RKNN_SUCC){ std::cerr << "rknn_init=" << ret << std::endl; return false; }

  // Try to configure NPU core mask if supported by SDK and user specified a non-zero mask
#if defined(RKNN_CONFIG_NPU_CORE_MASK)
  if (core_mask_ != 0) {
    int cm = static_cast<int>(core_mask_);
    int cfg_ret = rknn_config(impl_->ctx, RKNN_CONFIG_NPU_CORE_MASK, &cm);
    if (cfg_ret != RKNN_SUCC) {
      std::cerr << "[RknnEngine] rknn_config(NPU_CORE_MASK) failed: " << cfg_ret << std::endl;
    } else {
      std::cout << "[RknnEngine] NPU core mask configured: 0x" << std::hex << core_mask_ << std::dec << std::endl;
    }
  }
#endif

  ret = rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &impl_->io_num, sizeof(impl_->io_num));
  if(ret != RKNN_SUCC){ std::cerr << "query io_num=" << ret << std::endl; return false; }

  // Assume single output
  std::memset(&impl_->out_attr, 0, sizeof(impl_->out_attr));
  impl_->out_attr.index = impl_->io_num.n_output - 1;
  ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &impl_->out_attr, sizeof(impl_->out_attr));
  if(ret != RKNN_SUCC){ std::cerr << "query out attr=" << ret << std::endl; return false; }

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
  std::cout << "RknnEngine: RKNN platform not enabled at build time" << std::endl;
#endif
  is_initialized_ = true;
  std::cout << "RknnEngine: Initialized" << std::endl;
  return true;
}

std::vector<Detection> RknnEngine::infer(const cv::Mat& image){
  if(!is_initialized_) { std::cerr << "RknnEngine: Not initialized!" << std::endl; return {}; }

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
  if(ret != RKNN_SUCC){ std::cerr << "inputs_set=" << ret << std::endl; return {}; }

  ret = rknn_run(impl_->ctx, nullptr);
  if(ret != RKNN_SUCC){ std::cerr << "rknn_run=" << ret << std::endl; return {}; }

  rknn_output out{}; out.want_float = 1; out.is_prealloc = 0;
  ret = rknn_outputs_get(impl_->ctx, 1, &out, nullptr);
  if(ret != RKNN_SUCC){ std::cerr << "outputs_get=" << ret << std::endl; return {}; }

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

  // Decode to detections. Support two heads:
  // 1) YOLOv8 DFL head: C >= 64 (4*16 + num_classes)
  // 2) Raw head: [cx,cy,w,h,(obj),cls...]
  std::vector<Detection> dets;
  int N = impl_->out_n;
  int C = impl_->out_c;

  auto sigmoid = [](float x){ return 1.0f / (1.0f + std::exp(-x)); };

  if (C >= 64) {
    // DFL path (assume reg_max=16)
    const int reg_max = 16;
    int cls_ch = C - 4 * reg_max;
    if (num_classes_ < 0) num_classes_ = cls_ch;
    // Prepare strides and per-anchor stride map for imgsz=input_size_
    const int strides[3] = {8,16,32};
    std::vector<float> stride_map(N, 0.0f);
    {
      int idx = 0;
      for (int s : strides) {
        int fm = input_size_ / s;
        int count = fm * fm;
        for (int k = 0; k < count && idx < N; ++k) stride_map[idx++] = (float)s;
      }
      if (idx != N) {
        // try reverse order
        idx = 0;
        for (int si = 2; si >= 0; --si) {
          int s = strides[si];
          int fm = input_size_ / s;
          int count = fm * fm;
          for (int k = 0; k < count && idx < N; ++k) stride_map[idx++] = (float)s;
        }
      }
      // If still mismatch, fallback to approximate by nearest scale (uniform)
      if ((int)stride_map.size() != N || stride_map[0] == 0.0f) {
        float s = (float)strides[0];
        std::fill(stride_map.begin(), stride_map.end(), s);
      }
    }
    // Build anchor centers (cx,cy) for the same ordering assumption
    std::vector<float> anchor_cx(N, 0.0f), anchor_cy(N, 0.0f);
    {
      int idx = 0;
      bool filled = false;
      for (int pass = 0; pass < 2 && !filled; ++pass) {
        // pass 0: 8->16->32, pass 1: 32->16->8
        int order[3] = {8,16,32};
        if (pass == 1) { order[0]=32; order[1]=16; order[2]=8; }
        idx = 0;
        for (int s : order) {
          int fm = input_size_ / s;
          for (int iy = 0; iy < fm; ++iy) {
            for (int ix = 0; ix < fm; ++ix) {
              if (idx >= N) break;
              anchor_cx[idx] = (ix + 0.5f) * s;
              anchor_cy[idx] = (iy + 0.5f) * s;
              ++idx;
            }
          }
        }
        if (idx == N) filled = true;
      }
      if (!filled) {
        // Fallback fill with grid of stride 8
        idx = 0; int s = 8; int fm = input_size_ / s;
        for (int iy = 0; iy < fm && idx < N; ++iy)
          for (int ix = 0; ix < fm && idx < N; ++ix) {
            anchor_cx[idx] = (ix + 0.5f) * s;
            anchor_cy[idx] = (iy + 0.5f) * s;
            ++idx;
          }
      }
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
      float s = stride_map[i];
      // scale distances by stride
      float l = dfl[0] * s, t = dfl[1] * s, r = dfl[2] * s, b = dfl[3] * s;
      float x1 = anchor_cx[i] - l;
      float y1 = anchor_cy[i] - t;
      float x2 = anchor_cx[i] + r;
      float y2 = anchor_cy[i] + b;
      // class scores
      float best_conf = 0.f; int best_cls = 0;
      for (int c = 0; c < cls_ch; ++c) {
        float conf = sigmoid(logits[(4*reg_max + c)*N + i]);
        if (conf > best_conf) { best_conf = conf; best_cls = c; }
      }
      if (best_conf >= 0.25f) {
        Detection d;
        float scale = letterbox_info.scale, dx = letterbox_info.dx, dy = letterbox_info.dy;
        float w = x2 - x1, h = y2 - y1;
        d.x = (x1 - dx) / scale; d.y = (y1 - dy) / scale;
        d.w = w / scale; d.h = h / scale;
        d.confidence = best_conf; d.class_id = best_cls; d.class_name = "class_" + std::to_string(best_cls);
        dets.push_back(d);
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
      if(conf > 0.25f){
        Detection d;
        float scale = letterbox_info.scale, dx = letterbox_info.dx, dy = letterbox_info.dy;
        d.x = (cx - w/2 - dx) / scale;
        d.y = (cy - h/2 - dy) / scale;
        d.w = w / scale; d.h = h / scale;
        d.confidence = conf; d.class_id = best; d.class_name = "class_" + std::to_string(best);
        dets.push_back(d);
      }
    }
  }
  return dets;
#else
  // Non-RKNN build: return empty
  return {};
#endif
}

void RknnEngine::warmup(){
  if(!is_initialized_) { std::cerr << "RknnEngine: Cannot warmup - not initialized!" << std::endl; return; }
  cv::Mat dummy(input_size_, input_size_, CV_8UC3, cv::Scalar(128,128,128));
  (void)infer(dummy);
}

void RknnEngine::release(){
#if RKNN_PLATFORM
  if(impl_ && impl_->ctx){ rknn_destroy(impl_->ctx); }
#endif
  impl_.reset();
  is_initialized_ = false;
}

int RknnEngine::getInputWidth() const { return input_size_; }
int RknnEngine::getInputHeight() const { return input_size_; }

} // namespace rkapp::infer
