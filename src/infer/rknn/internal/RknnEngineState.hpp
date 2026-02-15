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
  std::mutex infer_mutex;
  std::atomic<bool> shutting_down{false};
#if RKNN_PLATFORM
  rknn_context ctx = 0;
  rknn_input_output_num io_num{};
  rknn_tensor_attr in_attr{};
  rknn_tensor_attr out_attr{};
  rknn_tensor_format input_fmt = RKNN_TENSOR_NHWC;
  rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
  // Output layout fields are initialized once in init() and treated as immutable afterwards.
  int out_elems = 0;
  int out_c = 0;
  int out_n = 0;

  AnchorLayout dfl_layout;
#endif
};

}  // namespace rkapp::infer
