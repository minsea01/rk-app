#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>

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
  
  // Get the channel dimension (assuming last dim contains features: x,y,w,h,[obj,]cls0,cls1,...)
  int last_dim = std::max(0, (int)impl_->out_attr.n_dims - 1);
  int64_t C = impl_->out_attr.dims[last_dim];
  // Infer number of classes: default to C-4 (x,y,w,h + classes)
  // If your export includes objness, set has_objness_=true and use C-5.
  has_objness_ = false;
  num_classes_ = (int)(C - 4);
  
  if (num_classes_ <= 0 || num_classes_ > 1024) {
    std::cerr << "[RknnEngine] Invalid classes inferred: " << num_classes_ << std::endl;
    return false;
  }
  
  // Determine output layout (N,C or C,N)
  if(impl_->out_attr.dims[1] == C){ 
    impl_->out_c = C; 
    impl_->out_n = impl_->out_attr.dims[2]; 
  } else if(impl_->out_attr.dims[2] == C){ 
    impl_->out_c = C; 
    impl_->out_n = impl_->out_attr.dims[1]; 
  } else { 
    std::cerr << "Unexpected output shape for C=" << C << std::endl; 
    return false; 
  }
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

  // Decode to detections (assume [1,8,8400])
  std::vector<Detection> dets;
  int N = impl_->out_n; // 8400
  int C = impl_->out_c; // 8
  for(int i=0;i<N;i++){
    float cx = logits[0*N + i];
    float cy = logits[1*N + i];
    float w  = logits[2*N + i];
    float h  = logits[3*N + i];
    float max_conf = 0.f; int best = 0;
    for(int c=0;c<C-4;c++){
      float conf = logits[(4+c)*N + i];
      if(conf > max_conf){ max_conf = conf; best = c; }
    }
    if(max_conf > 0.25f){
      Detection d;
      float scale = letterbox_info.scale, dx = letterbox_info.dx, dy = letterbox_info.dy;
      d.x = (cx - w/2 - dx) / scale;
      d.y = (cy - h/2 - dy) / scale;
      d.w = w / scale; d.h = h / scale;
      d.confidence = max_conf; d.class_id = best; d.class_name = "class_" + std::to_string(best);
      dets.push_back(d);
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
