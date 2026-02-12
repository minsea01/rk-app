#include "rkapp/infer/RknnEngine.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "internal/RknnEngineInternal.hpp"
#include "internal/RknnEngineState.hpp"
#include "rkapp/common/log.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

#if RKNN_PLATFORM
#include <rknn_api.h>
#if RKNN_USE_RGA
#include <im2d.h>
#include <rga.h>
#endif
#endif

namespace rkapp::infer {

RknnEngine::RknnEngine() = default;
RknnEngine::~RknnEngine() { release(); }

bool RknnEngine::init(const std::string& model_path, int img_size) {
  release();

  auto new_impl = std::make_shared<Impl>();
  ModelMeta model_meta = rknn_internal::loadModelMeta(model_path);
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
  if (!rknn_internal::readFile(model_path, blob, read_err)) {
    LOGE("RknnEngine: failed to read model file: ", model_path,
         " (exists=", std::filesystem::exists(model_path), ", reason=", read_err, ")");
    cleanup();
    return false;
  }
  if (blob.empty()) {
    LOGE("RknnEngine: model file empty: ", model_path);
    cleanup();
    return false;
  }

  int ret = rknn_init(&new_impl->ctx, blob.data(), blob.size(), 0, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine: rknn_init failed with code ", ret);
    cleanup();
    return false;
  }
  ctx_ready = true;

  uint32_t core_mask_snapshot = 0;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    core_mask_snapshot = core_mask_;
  }
  if (core_mask_snapshot != 0) {
    rknn_core_mask npu_core_mask = RKNN_NPU_CORE_AUTO;
    switch (core_mask_snapshot) {
      case 0x1:
        npu_core_mask = RKNN_NPU_CORE_0;
        break;
      case 0x2:
        npu_core_mask = RKNN_NPU_CORE_1;
        break;
      case 0x4:
        npu_core_mask = RKNN_NPU_CORE_2;
        break;
      case 0x3:
        npu_core_mask = RKNN_NPU_CORE_0_1;
        break;
      case 0x7:
        npu_core_mask = RKNN_NPU_CORE_0_1_2;
        break;
      default:
        LOGW("RknnEngine: Unknown core_mask 0x", std::hex, core_mask_snapshot, std::dec,
             ", defaulting to RKNN_NPU_CORE_0_1_2 (6 TOPS)");
        npu_core_mask = RKNN_NPU_CORE_0_1_2;
        break;
    }

    int mask_ret = rknn_set_core_mask(new_impl->ctx, npu_core_mask);
    if (mask_ret != RKNN_SUCC) {
      LOGW("RknnEngine: rknn_set_core_mask failed (code ", mask_ret,
           "). Running on default core. This may reduce throughput by up to 66%.");
    } else {
      const char* core_desc = "AUTO";
      switch (npu_core_mask) {
        case RKNN_NPU_CORE_0:
          core_desc = "Core0 (2 TOPS)";
          break;
        case RKNN_NPU_CORE_1:
          core_desc = "Core1 (2 TOPS)";
          break;
        case RKNN_NPU_CORE_2:
          core_desc = "Core2 (2 TOPS)";
          break;
        case RKNN_NPU_CORE_0_1:
          core_desc = "Core0+1 (4 TOPS)";
          break;
        case RKNN_NPU_CORE_0_1_2:
          core_desc = "Core0+1+2 (6 TOPS)";
          break;
        default:
          break;
      }
      LOGI("RknnEngine: NPU multi-core enabled: ", core_desc);
    }
  } else {
    LOGI("RknnEngine: NPU core_mask=0, using RKNN_NPU_CORE_AUTO mode");
  }

  ret = rknn_query(new_impl->ctx, RKNN_QUERY_IN_OUT_NUM, &new_impl->io_num,
                   sizeof(new_impl->io_num));
  if (ret != RKNN_SUCC) {
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
  ret = rknn_query(new_impl->ctx, RKNN_QUERY_INPUT_ATTR, &new_impl->in_attr,
                   sizeof(new_impl->in_attr));
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
  if (input_h > 0 && input_w > 0 && (input_h != img_size || input_w != img_size)) {
    LOGW("RknnEngine: Input size mismatch with model (model=", input_w, "x", input_h,
         ", cfg=", img_size, "x", img_size, ")");
  }

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

  new_impl->out_elems = 1;
  for (uint32_t i = 0; i < new_impl->out_attr.n_dims; i++) {
    new_impl->out_elems *= new_impl->out_attr.dims[i];
  }

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
    LOGI("RknnEngine: Detected channels_last layout [1, ", d1, ", ", d2,
         "], will transpose");
  } else {
    LOGE("RknnEngine: Output layout mismatch (dims=[1,", d1, ",", d2, "], expected C=",
         expected_c, ")");
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
    if (model_meta.reg_max > rknn_internal::kMaxSupportedRegMax) {
      LOGE("RknnEngine: reg_max=", model_meta.reg_max,
           " exceeds max (", rknn_internal::kMaxSupportedRegMax, ")");
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

  if (preprocessed_image.cols != input_size_snapshot || preprocessed_image.rows != input_size_snapshot) {
    LOGE("RknnEngine::inferPreprocessed: Input size mismatch. Expected ", input_size_snapshot,
         "x", input_size_snapshot, ", got ", preprocessed_image.cols, "x",
         preprocessed_image.rows);
    return {};
  }

  cv::Mat rgb =
      rkapp::preprocess::Preprocess::convertColor(preprocessed_image, cv::COLOR_BGR2RGB);
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
  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine: rknn_inputs_set failed: ", ret);
    return {};
  }

  ret = rknn_run(impl->ctx, nullptr);
  if (ret != RKNN_SUCC) {
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
  out.index = impl->out_attr.index;
  ret = rknn_outputs_get(impl->ctx, 1, &out, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine: rknn_outputs_get failed: ", ret);
    return {};
  }

  int num_classes = num_classes_snapshot;
  rknn_outputs_release(impl->ctx, 1, &out);
  lock.unlock();

  auto nms_result = rknn_internal::decodeOutputAndNms(
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
  (void)original_size;
  (void)letterbox_info;
  return {};
#endif
}

std::vector<Detection> RknnEngine::infer(const cv::Mat& image) {
#if RKNN_PLATFORM
  int input_size_snapshot = 0;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!is_initialized_) {
      LOGE("RknnEngine: Not initialized!");
      return {};
    }
    input_size_snapshot = input_size_;
  }

  rkapp::preprocess::LetterboxInfo letterbox_info;
  cv::Mat letter =
      rkapp::preprocess::Preprocess::letterbox(image, input_size_snapshot, letterbox_info);
  if (letter.empty()) {
    LOGE("RknnEngine: Preprocess failed (empty output). Input may be invalid.");
    return {};
  }

  return inferPreprocessed(letter, image.size(), letterbox_info);
#else
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!is_initialized_) {
      LOGE("RknnEngine: Not initialized!");
      return {};
    }
  }
  return {};
#endif
}

void RknnEngine::warmup() {
  int input_size_snapshot = 0;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!is_initialized_) {
      LOGW("RknnEngine: Cannot warmup - not initialized!");
      return;
    }
    input_size_snapshot = input_size_;
  }
  cv::Mat dummy(input_size_snapshot, input_size_snapshot, CV_8UC3, cv::Scalar(128, 128, 128));
  (void)infer(dummy);
}

void RknnEngine::release() {
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

}  // namespace rkapp::infer
