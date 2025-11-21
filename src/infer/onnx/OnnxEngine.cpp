#include "rkapp/infer/OnnxEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "log.hpp"
#include "rkapp/post/Postprocess.hpp"
#include <algorithm>
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

  static std::vector<Detection> parseOutput(Ort::Value& output,
                                            const rkapp::preprocess::LetterboxInfo& letterbox_info,
                                            cv::Size original_size,
                                            const DecodeParams& params) {
    std::vector<Detection> detections;
    auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    float* data = output.GetTensorMutableData<float>();

    // Handle different output formats
    int num_detections = 0;
    int num_classes = 0;
    if (shape.size() == 3) {
      if (shape[1] < 20) { // e.g., (1,8,8400)
        num_detections = static_cast<int>(shape[2]);
        num_classes = static_cast<int>(shape[1] - 4);
      } else if (shape[1] > shape[2]) { // (1,84,8400)
        num_detections = static_cast<int>(shape[2]);
        num_classes = static_cast<int>(shape[1] - 4);
      } else { // (1,8400,84)
        num_detections = static_cast<int>(shape[1]);
        num_classes = static_cast<int>(shape[2] - 4);
      }
    }

    if (num_detections <= 0 || num_classes < 0) {
      return detections;
    }

    int limit = num_detections;
    if (params.max_boxes > 0) {
      limit = std::min(limit, params.max_boxes);
    }

    for (int i = 0; i < limit; i++) {
      float cx, cy, w, h;
      if (shape[1] < 20) {
        cx = data[0 * num_detections + i];
        cy = data[1 * num_detections + i];
        w = data[2 * num_detections + i];
        h = data[3 * num_detections + i];
      } else if (shape[1] > shape[2]) {
        cx = data[0 * num_detections + i];
        cy = data[1 * num_detections + i];
        w = data[2 * num_detections + i];
        h = data[3 * num_detections + i];
      } else {
        int offset = i * (num_classes + 4);
        cx = data[offset + 0];
        cy = data[offset + 1];
        w = data[offset + 2];
        h = data[offset + 3];
      }

      float max_conf = 0.0f;
      int best_class = 0;
      for (int c = 0; c < num_classes; c++) {
        float conf;
        if (shape[1] < 20) conf = data[(4 + c) * num_detections + i];
        else if (shape[1] > shape[2]) conf = data[(4 + c) * num_detections + i];
        else { int offset = i * (num_classes + 4); conf = data[offset + 4 + c]; }
        if (conf > max_conf) { max_conf = conf; best_class = c; }
      }

      if (max_conf >= params.conf_thres) {
        Detection det;
        float scale = letterbox_info.scale, dx = letterbox_info.dx, dy = letterbox_info.dy;
        det.x = (cx - w/2 - dx) / scale;
        det.y = (cy - h/2 - dy) / scale;
        det.w = w / scale;
        det.h = h / scale;
        det.x = std::max(0.0f, std::min(det.x, (float)original_size.width));
        det.y = std::max(0.0f, std::min(det.y, (float)original_size.height));
        det.w = std::max(0.0f, std::min(det.w, (float)original_size.width - det.x));
        det.h = std::max(0.0f, std::min(det.h, (float)original_size.height - det.y));
        det.confidence = max_conf;
        det.class_id = best_class;
        det.class_name = "class_" + std::to_string(best_class);
        detections.push_back(det);
        if (params.max_boxes > 0 && static_cast<int>(detections.size()) >= params.max_boxes) {
          break;
        }
      }
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
  try {
    rkapp::preprocess::LetterboxInfo letterbox_info;
    cv::Mat processed = rkapp::preprocess::Preprocess::letterbox(image, input_size_, letterbox_info);
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
    auto dets = Impl::parseOutput(outputs[0], letterbox_info, image.size(), decode_params_);
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
