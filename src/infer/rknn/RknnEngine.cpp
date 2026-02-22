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
  // 重新初始化时先释放旧上下文，避免句柄泄漏或状态串扰。
  release();

  auto new_impl = std::make_shared<Impl>();
  // 读取模型元信息（head 类型、类别数、输出索引等），后续会严格校验。
  ModelMeta model_meta = rknn_internal::loadModelMeta(model_path);
  int inferred_num_classes = -1;
  bool inferred_has_objness = true;
 
#if RKNN_PLATFORM
  // cleanup 只负责当前 init 过程中的临时资源回收。
  bool ctx_ready = false;
  auto cleanup = [&]() {
    if (ctx_ready && new_impl && new_impl->ctx) {
      rknn_destroy(new_impl->ctx);
      new_impl->ctx = 0;
    }
  };

  std::vector<uint8_t> blob;
  std::string read_err;
  // 先把 rknn 文件读入内存，再调用 rknn_init。
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

  // core_mask 使用快照，避免读到并发修改中的状态。
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
    // 当掩码为 0 时交给 RKNN 自动调度。
    LOGI("RknnEngine: NPU core_mask=0, using RKNN_NPU_CORE_AUTO mode");
  }

  // 查询输入输出数量，是后续属性查询的前置条件。
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

  // 当前实现约束：输入需为 uint8 且布局是 NHWC/NCHW。
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

  // 从 RKNN 张量属性里提取输入尺寸，兼容 3D/4D 两种描述格式。
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
    // 仅告警不失败：允许配置尺寸与模型不一致，但结果可能异常或性能下降。
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

  // 多输出模型必须通过 metadata 指定要解码的输出分支，避免猜错输出头。
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

  // 计算输出总元素数，后续用于预分配缓存。
  new_impl->out_elems = 1;
  for (uint32_t i = 0; i < new_impl->out_attr.n_dims; i++) {
    new_impl->out_elems *= new_impl->out_attr.dims[i];
  }

  // 解码逻辑严格依赖 metadata，字段缺失直接报错，避免“静默错结果”。
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

  // 计算期望通道数 expected_c，用于判断输出张量布局是否匹配。
  const int kpt_ch = model_meta.num_keypoints > 0 ? model_meta.num_keypoints * 3 : 0;
  int expected_c = 0;
  if (model_meta.head == "dfl") {
    if (model_meta.reg_max <= 0 || model_meta.strides.empty()) {
      LOGE("RknnEngine: DFL decode requires reg_max and strides metadata");
      cleanup();
      return false;
    }
    expected_c = 4 * model_meta.reg_max + model_meta.num_classes + kpt_ch;
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
  // 输出常见为 [1, C, N] 或 [1, N, C]，这里自动识别并记录 out_n/out_c。
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
    // DFL 头需要构建 anchor layout，后续 decode 时会按该布局还原边框。
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

  // 走到这里说明初始化流程完成，将新状态一次性替换进对象。
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

  // 入参尺寸必须和模型输入一致；该函数不再做缩放补边。
  if (preprocessed_image.cols != input_size_snapshot || preprocessed_image.rows != input_size_snapshot) {
    LOGE("RknnEngine::inferPreprocessed: Input size mismatch. Expected ", input_size_snapshot,
         "x", input_size_snapshot, ", got ", preprocessed_image.cols, "x",
         preprocessed_image.rows);
    return {};
  }
  // Impl 字段在 init() 后不再变更，通过持有 shared_ptr 保证对象存活；
  // 此处读取无需再持有 state_mutex_（init 与 infer 不允许并发）。
  const int out_n_snapshot = impl->out_n;
  const int out_c_snapshot = impl->out_c;
  const int out_elems_snapshot = impl->out_elems;
  const int out_n_dims_snapshot = static_cast<int>(impl->out_attr.n_dims);
  const int out_dim1_snapshot = static_cast<int>(impl->out_attr.dims[1]);
  const int out_dim2_snapshot = static_cast<int>(impl->out_attr.dims[2]);
  const uint32_t out_index_snapshot = impl->out_attr.index;

  // RKNN 训练/转换常用 RGB 输入；上游传入 BGR，这里统一转换一次。
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
    // RKNN NCHW 输入时，需要把 HWC 内存重排成 CHW 连续内存。
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
    // NHWC 路径可直接复用 OpenCV 连续内存。
    in.size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
    in.buf = (void*)rgb.data;
  }

  // infer_mutex 串行化同一 RKNN context 的调用，避免并发访问 SDK 产生未定义行为。
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

  // thread_local 缓冲区：减少频繁堆分配，提高稳定帧率。
  const size_t logits_elems = static_cast<size_t>(impl->out_elems);
  thread_local std::vector<float> logits_local;
  logits_local.resize(logits_elems);

  rknn_output out{};
  out.want_float = 1;
  out.is_prealloc = 1;
  out.buf = logits_local.data();
  out.size = logits_local.size() * sizeof(float);
  out.index = out_index_snapshot;
  ret = rknn_outputs_get(impl->ctx, 1, &out, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine: rknn_outputs_get failed: ", ret);
    return {};
  }

  // outputs_release 后锁就可释放，后处理（decode+NMS）不需要持有 RKNN 互斥锁。
  int num_classes = num_classes_snapshot;
  rknn_outputs_release(impl->ctx, 1, &out);
  lock.unlock();

  // 统一的解码+NMS入口：根据 model_meta 自动选择 raw/dfl 解析逻辑。
  auto nms_result = rknn_internal::decodeOutputAndNms(
      logits_local.data(), out_n_snapshot, out_c_snapshot, out_elems_snapshot, out_n_dims_snapshot,
      out_dim1_snapshot, out_dim2_snapshot, num_classes, model_meta_snapshot,
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

  // 通用入口先做 letterbox，再复用 inferPreprocessed 的核心推理逻辑。
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
  // 灰色虚拟帧用于快速预热，不依赖真实输入。
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
    // 先标记 shutdown，再拿 infer_mutex，避免释放与正在推理并发冲突。
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

std::vector<std::string> RknnEngine::class_names() const {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  return class_names_;
}

}  // namespace rkapp::infer
