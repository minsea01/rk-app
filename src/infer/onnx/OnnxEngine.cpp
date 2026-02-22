#include "rkapp/infer/OnnxEngine.hpp"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <vector>

#include "internal/OnnxEngineInternal.hpp"
#include "rkapp/common/log.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

namespace rkapp::infer {

struct OnnxEngine::Impl {
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
  std::unique_ptr<Ort::MemoryInfo> memory_info;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  bool use_cuda = false;
  onnx_internal::QuantParams output_quant;
  onnx_internal::DecodeMeta decode_meta;
};

OnnxEngine::OnnxEngine() = default;
OnnxEngine::~OnnxEngine() { release(); }

bool OnnxEngine::init(const std::string& model_path, int img_size) {
  try {
    model_path_ = model_path;
    input_size_ = img_size;
    impl_ = std::make_unique<Impl>();

    LOGI("OnnxEngine: Initializing with model ", model_path_, " (size: ", input_size_, ")");

    impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxEngine");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(0);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    LOGI("OnnxEngine: Attempting CUDA provider (device ", cuda_device_id_, ")...");
    try {
      OrtCUDAProviderOptions cuda_opts{};
      cuda_opts.device_id = cuda_device_id_;
      cuda_opts.arena_extend_strategy = 1;
      cuda_opts.do_copy_in_default_stream = 1;
      session_options.AppendExecutionProvider_CUDA(cuda_opts);
      impl_->use_cuda = true;
      LOGI("OnnxEngine: CUDA provider added (device ", cuda_device_id_, ")");
    } catch (const std::exception& e) {
      impl_->use_cuda = false;
      LOGW("OnnxEngine: CUDA provider unavailable, CPU fallback: ", e.what());
    }

    impl_->session = std::make_unique<Ort::Session>(*impl_->env, model_path_.c_str(), session_options);
    impl_->memory_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

    const size_t num_input_nodes = impl_->session->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
      auto name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
      impl_->input_names.push_back(name.get());
    }
    const size_t num_output_nodes = impl_->session->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
      auto name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
      impl_->output_names.push_back(name.get());
    }

    if (!impl_->output_names.empty()) {
      impl_->output_quant = onnx_internal::resolveOutputQuantParams(
          model_path_, impl_->output_names[0], *impl_->session, impl_->allocator);
      auto out_type_info = impl_->session->GetOutputTypeInfo(0);
      auto out_tensor_info = out_type_info.GetTensorTypeAndShapeInfo();
      auto out_type = out_tensor_info.GetElementType();
      if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
          out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        const bool is_int8 = out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        const float default_scale = 1.0f / 127.0f;
        const int32_t default_zero = is_int8 ? 0 : 128;
        const float used_scale =
            (impl_->output_quant.has_scale && impl_->output_quant.scale > 0.0f)
                ? impl_->output_quant.scale
                : default_scale;
        const int32_t used_zero =
            impl_->output_quant.has_zero_point ? impl_->output_quant.zero_point : default_zero;
        if (impl_->output_quant.has_scale && impl_->output_quant.has_zero_point) {
          LOGI("OnnxEngine: Using output quant params scale=", used_scale,
               ", zero_point=", used_zero);
        } else {
          LOGW("OnnxEngine: Quantized output missing scale/zero-point; using fallback scale=",
               used_scale, ", zero_point=", used_zero);
        }
      }
    }

    impl_->decode_meta = onnx_internal::parseDecodeMetaFromSidecar(model_path_);
    if (impl_->decode_meta.hasAny()) {
      LOGI("OnnxEngine: Loaded decode metadata (head=",
           impl_->decode_meta.head.empty() ? "auto" : impl_->decode_meta.head,
           ", reg_max=", impl_->decode_meta.reg_max, ", num_classes=",
           impl_->decode_meta.num_classes, ", has_objectness=",
           impl_->decode_meta.has_objectness, ")");
    } else {
      LOGW("OnnxEngine: No decode metadata found; using conservative auto decode");
    }

    unsupported_model_ = false;
    is_initialized_ = true;
    LOGI("OnnxEngine: Initialized successfully");
    return true;
  } catch (const Ort::Exception& e) {
    LOGE("OnnxEngine init error: ", e.what());
    return false;
  }
}

std::vector<Detection> OnnxEngine::infer(const cv::Mat& image) {
  std::lock_guard<std::recursive_mutex> lock(engine_mtx_);
  if (!is_initialized_) {
    LOGE("OnnxEngine: Not initialized!");
    return {};
  }
  if (unsupported_model_) {
    LOGE("OnnxEngine: Model output layout previously marked unsupported; reinitialize with valid metadata");
    return {};
  }

  try {
    rkapp::preprocess::LetterboxInfo letterbox_info;
    cv::Mat processed = rkapp::preprocess::Preprocess::letterbox(image, input_size_, letterbox_info);
    if (processed.empty()) {
      LOGE("OnnxEngine: Preprocess failed (empty output). Input may be invalid.");
      return {};
    }
    cv::Mat rgb = rkapp::preprocess::Preprocess::convertColor(processed, cv::COLOR_BGR2RGB);
    cv::Mat normalized = rkapp::preprocess::Preprocess::normalize(rgb, 1.0f / 255.0f);
    cv::Mat blob = rkapp::preprocess::Preprocess::hwc2chw(normalized);

    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(*impl_->memory_info, blob.ptr<float>(),
                                                         blob.total(), input_shape.data(),
                                                         input_shape.size()));

    std::vector<const char*> in_names;
    std::vector<const char*> out_names;
    for (auto& n : impl_->input_names) in_names.push_back(n.c_str());
    for (auto& n : impl_->output_names) out_names.push_back(n.c_str());

    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                                       inputs.size(), out_names.data(), out_names.size());
    if (outputs.empty()) {
      LOGW("OnnxEngine inference returned no outputs");
      return {};
    }
    if (outputs.size() > 1) {
      LOGW("OnnxEngine: multiple outputs detected (", outputs.size(), "), using first tensor only");
    }

    bool unsupported = false;
    auto dets = onnx_internal::parseOutput(outputs[0], letterbox_info, image.size(), input_size_,
                                           decode_params_, impl_->decode_meta,
                                           impl_->output_quant, unsupported);
    if (unsupported) {
      unsupported_model_ = true;
      LOGE("OnnxEngine: Unsupported output layout; stopping further inference attempts");
      return {};
    }
    unsupported_model_ = false;

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
  if (!is_initialized_) {
    LOGW("OnnxEngine: Cannot warmup - not initialized!");
    return;
  }
  cv::Mat dummy(input_size_, input_size_, CV_8UC3, cv::Scalar(128, 128, 128));
  (void)infer(dummy);
}

void OnnxEngine::release() {
  std::lock_guard<std::recursive_mutex> lock(engine_mtx_);
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

void OnnxEngine::setDecodeParams(const DecodeParams& params) { decode_params_ = params; }

}  // namespace rkapp::infer
