#pragma once

#include <string>
#include <vector>

#include "rkapp/infer/ModelMeta.hpp"

namespace rkapp::infer {

enum class ModelBackend {
  AUTO,
  ONNX,
  RKNN,
};

struct ModelSpec {
  ModelBackend backend = ModelBackend::AUTO;
  std::string model_path;
  int input_size = 640;
  float conf_threshold = 0.5f;
  float iou_threshold = 0.45f;
  int max_detections = 100;
  float min_box_size = 0.0f;
  float max_box_size = 0.0f;
  float min_aspect_ratio = 0.0f;
  float max_aspect_ratio = 0.0f;
  bool use_npu_multicore = true;
  bool use_zero_copy = false;
  int buffer_pool_size = 4;
  std::vector<std::string> class_names;
  std::string decode_meta_path;
  ModelMeta decode_meta;
};

}  // namespace rkapp::infer
