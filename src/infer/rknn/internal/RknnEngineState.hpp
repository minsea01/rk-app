#pragma once

#include <atomic>
#include <mutex>

#include "rkapp/infer/RknnDecodeUtils.hpp"
#include "rkapp/infer/RknnEngine.hpp"

#if RKNN_PLATFORM
#include <rknn_api.h>
#endif

namespace rkapp::infer {

struct RknnEngine::Impl {
  // 同一个 RKNN context 在推理阶段串行访问，规避并发调用 SDK 的未定义行为。
  std::mutex infer_mutex;
  std::atomic<bool> shutting_down{false};
#if RKNN_PLATFORM
  rknn_context ctx = 0;
  rknn_input_output_num io_num{};
  rknn_tensor_attr in_attr{};
  rknn_tensor_attr out_attr{};
  rknn_tensor_format input_fmt = RKNN_TENSOR_NHWC;
  rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
  // 输出布局字段在 init 完成后视为只读。
  int out_elems = 0;
  int out_c = 0;
  int out_n = 0;

  // DFL 头对应的 anchor 布局缓存（按输出 N 构建）。
  AnchorLayout dfl_layout;
#endif
};

}  // namespace rkapp::infer
