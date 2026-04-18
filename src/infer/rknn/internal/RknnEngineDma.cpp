#include "RknnEngineState.hpp"

#include <array>
#include <mutex>
#include <utility>
#include <vector>

#include <unistd.h>

#include "RknnEngineInternal.hpp"
#include "rkapp/common/DmaBuf.hpp"
#include "rkapp/common/log.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

#if RKNN_PLATFORM
#include <rknn_api.h>
#endif

namespace rkapp::infer {

// DMA-BUF 推理分支：优先零拷贝直通，失败时回退到复制路径。
namespace {

// fd 生命周期守卫，避免异常路径泄漏句柄。
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
    if (fd >= 0) {
      close(fd);
    }
    fd = new_fd;
  }

  int get() const { return fd; }
  bool valid() const { return fd >= 0; }
};

}  // namespace

std::vector<Detection> RknnEngine::inferDmaBuf(
    rkapp::common::DmaBuf& input,
    const cv::Size& original_size,
    const rkapp::preprocess::LetterboxInfo& letterbox_info) {

#if RKNN_PLATFORM
  // 先拍快照：后续在无锁阶段使用，减少状态锁持有时间。
  std::shared_ptr<Impl> impl;
  int input_size_snapshot = 0;
  ModelMeta model_meta_snapshot;
  DecodeParams decode_params_snapshot;
  int num_classes_snapshot = -1;
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!is_initialized_) {
      LOGE("RknnEngine::inferDmaBuf: Not initialized!");
      return {};
    }
    if (!impl_) {
      LOGE("RknnEngine::inferDmaBuf: Missing implementation state");
      return {};
    }
    impl = impl_;
    input_size_snapshot = input_size_;
    model_meta_snapshot = model_meta_;
    decode_params_snapshot = decode_params_;
    num_classes_snapshot = num_classes_;
  }

  auto fallback_to_copy = [&]() -> std::vector<Detection> {
    // 回退路径：DMA-BUF -> Mat -> 颜色转换 -> 常规推理。
    cv::Mat mat;
    if (!input.copyTo(mat)) {
      LOGE("RknnEngine::inferDmaBuf: Failed to copy DMA-BUF to Mat");
      return {};
    }
    cv::Mat bgr;
    try {
      switch (input.format()) {
        case rkapp::common::DmaBuf::PixelFormat::RGB888:
          cv::cvtColor(mat, bgr, cv::COLOR_RGB2BGR);
          break;
        case rkapp::common::DmaBuf::PixelFormat::BGR888:
          bgr = mat;
          break;
        case rkapp::common::DmaBuf::PixelFormat::RGBA8888:
          cv::cvtColor(mat, bgr, cv::COLOR_RGBA2BGR);
          break;
        case rkapp::common::DmaBuf::PixelFormat::BGRA8888:
          cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
          break;
        case rkapp::common::DmaBuf::PixelFormat::NV12:
          cv::cvtColor(mat, bgr, cv::COLOR_YUV2BGR_NV12);
          break;
        case rkapp::common::DmaBuf::PixelFormat::NV21:
          cv::cvtColor(mat, bgr, cv::COLOR_YUV2BGR_NV21);
          break;
        default:
          LOGE("RknnEngine::inferDmaBuf: Unsupported DMA-BUF format for fallback conversion");
          return {};
      }
    } catch (const cv::Exception& e) {
      LOGE("RknnEngine::inferDmaBuf: Fallback color conversion failed: ", e.what());
      return {};
    }
    if (bgr.empty()) {
      LOGE("RknnEngine::inferDmaBuf: Fallback conversion produced empty image");
      return {};
    }
    if (bgr.cols == input_size_snapshot && bgr.rows == input_size_snapshot) {
      return inferPreprocessed(bgr, original_size, letterbox_info);
    }
    LOGW("RknnEngine::inferDmaBuf: Fallback frame is not letterboxed (", bgr.cols, "x",
         bgr.rows, "), running full infer()");
    return infer(bgr);
  };

#if defined(RKAPP_RKNN_DMA_FD_INPUT)
  if (input.width() != input_size_snapshot || input.height() != input_size_snapshot) {
    LOGE("RknnEngine::inferDmaBuf: Input size mismatch. Expected ", input_size_snapshot, "x",
         input_size_snapshot, ", got ", input.width(), "x", input.height());
    return {};
  }

  const int out_n_snapshot = impl->out_n;
  const int out_c_snapshot = impl->out_c;
  const int out_elems_snapshot = impl->out_elems;
  const int out_n_dims_snapshot = impl->out_attr.n_dims;
  const int out_dim1_snapshot = impl->out_attr.dims[1];
  const int out_dim2_snapshot = impl->out_attr.dims[2];
  const uint32_t out_index_snapshot = impl->out_attr.index;

  if (impl->input_fmt != RKNN_TENSOR_NHWC || impl->input_type != RKNN_TENSOR_UINT8) {
    // 直接 DMA-FD 仅支持 NHWC UINT8。
    LOGW("RknnEngine::inferDmaBuf: Zero-copy requires NHWC UINT8, falling back to copy");
    return fallback_to_copy();
  }
  if (input.format() != rkapp::common::DmaBuf::PixelFormat::RGB888) {
    // 当前直通路径只接受 RGB888，其他格式走回退转换。
    LOGW("RknnEngine::inferDmaBuf: Direct DMA-FD path expects RGB888 input, falling back to copy");
    return fallback_to_copy();
  }

  ScopedFd dma_fd(input.exportFd());
  if (!dma_fd.valid()) {
    LOGW("RknnEngine::inferDmaBuf: Failed to export DMA-BUF fd, falling back to copy path");
    return fallback_to_copy();
  }

  std::unique_lock<std::mutex> lock(impl->infer_mutex);
  if (impl->shutting_down.load(std::memory_order_acquire)) {
    LOGW("RknnEngine::inferDmaBuf: Engine is shutting down");
    return {};
  }

  int ret = RKNN_SUCC;
#if defined(RKAPP_RKNN_IO_MEM)
  // RKNN IO_MEM 路径：由 SDK 负责 fd -> tensor 绑定。
  rknn_tensor_mem* input_mem = nullptr;
  rknn_mem_info mem_info{};
  mem_info.fd = dma_fd.get();
  mem_info.offset = 0;
  mem_info.size = input.size();
  input_mem = rknn_create_mem_from_fd(impl->ctx, &mem_info);
  if (!input_mem) {
    LOGW("RknnEngine::inferDmaBuf: rknn_create_mem_from_fd failed, falling back to copy path");
    dma_fd.reset();
    lock.unlock();
    return fallback_to_copy();
  }

  ret = rknn_set_io_mem(impl->ctx, input_mem, &impl->in_attr);
  if (ret != RKNN_SUCC) {
    rknn_destroy_mem(impl->ctx, input_mem);
    LOGW("RknnEngine::inferDmaBuf: rknn_set_io_mem failed (code ", ret,
         "), falling back to copy path");
    dma_fd.reset();
    lock.unlock();
    return fallback_to_copy();
  }
#else
  // 旧接口路径：通过 pass_through + buf 传递 fd。
  rknn_input in{};
  in.index = 0;
  in.type = impl->input_type;
  in.fmt = impl->input_fmt;
  in.size = static_cast<uint32_t>(input.size());
  in.pass_through = 1;
  in.buf = reinterpret_cast<void*>(static_cast<intptr_t>(dma_fd.get()));

  ret = rknn_inputs_set(impl->ctx, 1, &in);

  if (ret != RKNN_SUCC) {
    LOGW("RknnEngine::inferDmaBuf: rknn_inputs_set failed (code ", ret,
         "), falling back to copy path");
    dma_fd.reset();
    lock.unlock();
    return fallback_to_copy();
  }
#endif

  ret = rknn_run(impl->ctx, nullptr);
  if (ret != RKNN_SUCC) {
#if defined(RKAPP_RKNN_IO_MEM)
    if (input_mem) {
      rknn_destroy_mem(impl->ctx, input_mem);
    }
#endif
    LOGE("RknnEngine::inferDmaBuf: rknn_run failed: ", ret);
    return {};
  }

  const size_t logits_elems = static_cast<size_t>(out_elems_snapshot);
  // thread_local 减少重复分配。
  thread_local std::vector<float> logits_local;
  logits_local.resize(logits_elems);

  rknn_output out{};
  out.want_float = 1;
  out.is_prealloc = 1;
  out.buf = logits_local.data();
  out.size = logits_local.size() * sizeof(float);
  out.index = out_index_snapshot;
  ret = rknn_outputs_get(impl->ctx, 1, &out, nullptr);

#if defined(RKAPP_RKNN_IO_MEM)
  if (input_mem) {
    rknn_destroy_mem(impl->ctx, input_mem);
  }
#endif

  dma_fd.reset();

  if (ret != RKNN_SUCC) {
    LOGE("RknnEngine::inferDmaBuf: rknn_outputs_get failed: ", ret);
    return {};
  }

  int num_classes = num_classes_snapshot;
  rknn_outputs_release(impl->ctx, 1, &out);
  lock.unlock();

  auto nms_result = rknn_internal::decodeOutputAndNms(
      logits_local.data(), out_n_snapshot, out_c_snapshot, out_elems_snapshot, out_n_dims_snapshot,
      out_dim1_snapshot, out_dim2_snapshot, num_classes, model_meta_snapshot,
      decode_params_snapshot, original_size, letterbox_info, &impl->dfl_layout,
      "RknnEngine::inferDmaBuf");

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (num_classes > 0 && num_classes_ < 0) {
      num_classes_ = num_classes;
    }
  }
  return nms_result;
#else
  // 编译期未启用 DMA-FD，则仅提示并回退。
  static std::once_flag warn_once;
  std::call_once(warn_once, []() {
    LOGW("RknnEngine::inferDmaBuf: DMA-FD input disabled; enable ENABLE_RKNN_DMA_FD to use it");
  });
  return fallback_to_copy();
#endif

#else
  // 非 RKNN 平台：保留功能可用，退化到复制路径。
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (!is_initialized_) {
      LOGE("RknnEngine::inferDmaBuf: Not initialized!");
      return {};
    }
  }
  cv::Mat mat;
  if (!input.copyTo(mat)) {
    LOGE("RknnEngine::inferDmaBuf: Failed to copy DMA-BUF to Mat");
    return {};
  }
  (void)letterbox_info;
  return infer(mat);
#endif
}

}  // namespace rkapp::infer
