#include "rkapp/infer/OnnxEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "log.hpp"
#include "rkapp/post/Postprocess.hpp"
#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>

namespace rkapp::infer {

struct OnnxEngine::Impl {
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
  std::unique_ptr<Ort::MemoryInfo> memory_info;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  bool use_cuda = false;

  static bool resolve_layout(const std::vector<int64_t>& shape, int& N, int& C) {
    if (shape.size() == 3) {
      const int64_t d1 = shape[1];
      const int64_t d2 = shape[2];
      if (d1 >= 4 && d1 <= 4096 && d2 > d1) { C = static_cast<int>(d1); N = static_cast<int>(d2); return true; }
      if (d2 >= 4 && d2 <= 4096 && d1 > d2) { C = static_cast<int>(d2); N = static_cast<int>(d1); return true; }
      if (d1 >= 4 && d2 >= 4) { C = static_cast<int>(std::min(d1, d2)); N = static_cast<int>(std::max(d1, d2)); return true; }
      return false;
    }
    if (shape.size() == 2) {
      const int64_t d1 = shape[0];
      const int64_t d2 = shape[1];
      if (d2 >= 4) { N = static_cast<int>(d1); C = static_cast<int>(d2); return true; }
      if (d1 >= 4) { N = static_cast<int>(d2); C = static_cast<int>(d1); return true; }
    }
    return false;
  }

  static std::vector<Detection> parseOutput(Ort::Value& output,
                                            const rkapp::preprocess::LetterboxInfo& letterbox_info,
                                            cv::Size original_size,
                                            const DecodeParams& params,
                                            bool& unsupported_model) {
    std::vector<Detection> detections;
    auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    float* data = output.GetTensorMutableData<float>();

    int N = 0, C = 0;
    if (!resolve_layout(shape, N, C)) {
      LOGW("OnnxEngine: unsupported output shape");
      return detections;
    }

    bool channels_first = false;
    if (shape.size() == 3) {
      const int64_t d1 = shape[1], d2 = shape[2];
      channels_first = (d1 == C && d2 == N); // (1, C, N)
    } else if (shape.size() == 2) {
      channels_first = false; // assume (N, C)
    }

    auto at = [&](int c, int i) -> float {
      if (channels_first) return data[c * N + i];
      return data[i * C + c];
    };

    const bool has_objness = (C - 5) >= 1;
    const int cls_offset = has_objness ? 5 : 4;
    const int num_classes = std::max(0, C - cls_offset);
    if (num_classes <= 0) {
      LOGW("OnnxEngine: invalid class channel count");
      return detections;
    }
    // DFL-style outputs (e.g., YOLOv8/YOLO11) are not supported by this lightweight decoder
    if (C >= 64) {
      LOGE("OnnxEngine: detected large channel dimension (C=", C, ") that likely corresponds to a DFL head; "
           "export a raw head or add DFL decoding before using this engine");
      unsupported_model = true;
      return {};
    }

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

    int limit = N;
    if (params.max_boxes > 0) limit = std::min(limit, params.max_boxes);

    for (int i = 0; i < limit; i++) {
      // layout: [cx, cy, w, h, (obj), cls...]
      const float cx = at(0, i);
      const float cy = at(1, i);
      const float w  = at(2, i);
      const float h  = at(3, i);

      float obj = 1.0f;
      if (has_objness) obj = sigmoid(at(4, i));

      float best_conf = 0.0f;
      int best_cls = 0;
      for (int c = 0; c < num_classes; ++c) {
        float conf = sigmoid(at(cls_offset + c, i));
        if (conf > best_conf) { best_conf = conf; best_cls = c; }
      }

      const float combined = obj * best_conf;
      if (combined < params.conf_thres) continue;

      Detection det;
      const float scale = letterbox_info.scale;
      const float dx = letterbox_info.dx;
      const float dy = letterbox_info.dy;
      det.x = (cx - w / 2.0f - dx) / scale;
      det.y = (cy - h / 2.0f - dy) / scale;
      det.w = w / scale;
      det.h = h / scale;
      det.x = std::max(0.0f, std::min(det.x, static_cast<float>(original_size.width)));
      det.y = std::max(0.0f, std::min(det.y, static_cast<float>(original_size.height)));
      det.w = std::max(0.0f, std::min(det.w, static_cast<float>(original_size.width) - det.x));
      det.h = std::max(0.0f, std::min(det.h, static_cast<float>(original_size.height) - det.y));
      det.confidence = combined;
      det.class_id = best_cls;
      det.class_name = "class_" + std::to_string(best_cls);
      detections.push_back(det);
    }
    return detections;
  }
};

OnnxEngine::OnnxEngine() = default;
OnnxEngine::~OnnxEngine() { release(); }

bool OnnxEngine::init(const std::string& model_path, int img_size) {
  try {
    model_path_ = model_path;
    input_size_ = img_size;
    impl_ = std::make_unique<Impl>();

    LOGI("OnnxEngine: Initializing with model ", model_path_,
         " (size: ", input_size_, ")");

    impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxEngine");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(0);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try CUDA if available
    LOGI("OnnxEngine: Attempting CUDA provider...");
    try {
      OrtCUDAProviderOptions cuda_opts{};
      cuda_opts.device_id = 0;
      cuda_opts.arena_extend_strategy = 1;
      cuda_opts.do_copy_in_default_stream = 1;
      session_options.AppendExecutionProvider_CUDA(cuda_opts);
      impl_->use_cuda = true;
      LOGI("OnnxEngine: CUDA provider added");
    } catch (const std::exception& e) {
      impl_->use_cuda = false;
      LOGW("OnnxEngine: CUDA provider unavailable, CPU fallback: ", e.what());
    }

    impl_->session = std::make_unique<Ort::Session>(*impl_->env, model_path_.c_str(), session_options);
    impl_->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

    // I/O names
    size_t num_input_nodes = impl_->session->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
      auto name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
      impl_->input_names.push_back(name.get());
    }
    size_t num_output_nodes = impl_->session->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
      auto name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
      impl_->output_names.push_back(name.get());
    }

    is_initialized_ = true;
    LOGI("OnnxEngine: Initialized successfully");
    return true;
  } catch (const Ort::Exception& e) {
    LOGE("OnnxEngine init error: ", e.what());
    return false;
  }
}

std::vector<Detection> OnnxEngine::infer(const cv::Mat& image) {
  if (!is_initialized_) { LOGE("OnnxEngine: Not initialized!"); return {}; }
  if (unsupported_model_) { LOGE("OnnxEngine: Model output layout unsupported (DFL). Aborting inference."); return {}; }
  try {
    rkapp::preprocess::LetterboxInfo letterbox_info;
    cv::Mat processed = rkapp::preprocess::Preprocess::letterbox(image, input_size_, letterbox_info);
    if (processed.empty()) {
      LOGE("OnnxEngine: Preprocess failed (empty output). Input may be invalid.");
      return {};
    }
    cv::Mat rgb = rkapp::preprocess::Preprocess::convertColor(processed, cv::COLOR_BGR2RGB);
    cv::Mat normalized = rkapp::preprocess::Preprocess::normalize(rgb, 1.0f/255.0f);
    cv::Mat blob = rkapp::preprocess::Preprocess::hwc2chw(normalized);

    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        *impl_->memory_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size()));

    std::vector<const char*> in_names, out_names;
    for (auto& n : impl_->input_names) in_names.push_back(n.c_str());
    for (auto& n : impl_->output_names) out_names.push_back(n.c_str());

    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr},
                                       in_names.data(), inputs.data(), inputs.size(),
                                       out_names.data(), out_names.size());
    if (outputs.empty()) {
      LOGW("OnnxEngine inference returned no outputs");
      return {};
    }
    if (outputs.size() > 1) {
      LOGW("OnnxEngine: multiple outputs detected (", outputs.size(), "), using first tensor only");
    }
    bool unsupported = false;
    auto dets = Impl::parseOutput(outputs[0], letterbox_info, image.size(), decode_params_, unsupported);
    if (unsupported) {
      unsupported_model_ = true;
      LOGE("OnnxEngine: Unsupported DFL-style output; stopping further inference attempts");
      return {};
    }
    rkapp::post::NMSConfig nms_cfg;
    nms_cfg.conf_thres = decode_params_.conf_thres;
    nms_cfg.iou_thres = decode_params_.iou_thres;
    if (decode_params_.max_boxes > 0) {
      nms_cfg.max_det = decode_params_.max_boxes;
      nms_cfg.topk = decode_params_.max_boxes;
    }
    return rkapp::post::Postprocess::nms(dets, nms_cfg);
  } catch (const Ort::Exception& e) {
    LOGE("OnnxEngine inference error: ", e.what());
    return {};
  }
}

void OnnxEngine::warmup() {
  if (!is_initialized_) { LOGW("OnnxEngine: Cannot warmup - not initialized!"); return; }
  cv::Mat dummy(input_size_, input_size_, CV_8UC3, cv::Scalar(128,128,128));
  (void)infer(dummy);
}

void OnnxEngine::release() {
  if (!impl_) return;
  impl_->session.reset();
  impl_->env.reset();
  impl_->memory_info.reset();
  impl_.reset();
  is_initialized_ = false;
  LOGI("OnnxEngine: Released");
}

int OnnxEngine::getInputWidth() const { return input_size_; }
int OnnxEngine::getInputHeight() const { return input_size_; }

void OnnxEngine::setDecodeParams(const DecodeParams& params) {
  decode_params_ = params;
}

} // namespace rkapp::infer
