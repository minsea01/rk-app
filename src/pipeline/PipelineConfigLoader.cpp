#include "rkapp/pipeline/DetectionPipeline.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <yaml-cpp/yaml.h>

#include "rkapp/common/StringUtils.hpp"
#include "rkapp/common/log.hpp"

namespace rkapp::pipeline {
namespace {

std::string trimCopy(std::string value) {
  auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(),
                           [&](unsigned char ch) { return !is_space(ch); }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [&](unsigned char ch) { return !is_space(ch); })
                  .base(),
              value.end());
  return value;
}

std::filesystem::path resolveConfigPath(const std::filesystem::path& base_dir,
                                        const std::string& raw_path) {
  if (raw_path.empty()) {
    return {};
  }
  const std::filesystem::path path(raw_path);
  if (path.is_absolute()) {
    return path.lexically_normal();
  }
  return std::filesystem::absolute(base_dir / path).lexically_normal();
}

capture::SourceType parseSourceType(const std::string& value) {
  const std::string lowered = rkapp::common::toLowerCopy(value);
  if (lowered == "folder" || lowered == "image") {
    return capture::SourceType::FOLDER;
  }
  if (lowered == "video") {
    return capture::SourceType::VIDEO;
  }
  if (lowered == "rtsp") {
    return capture::SourceType::RTSP;
  }
  if (lowered == "gige") {
    return capture::SourceType::GIGE;
  }
  if (lowered == "mpp") {
    return capture::SourceType::MPP;
  }
  if (lowered == "csi") {
    return capture::SourceType::CSI;
  }
  return capture::SourceType::VIDEO;
}

PipelineConfig::ModelBackend parseBackend(const std::string& value) {
  const std::string lowered = rkapp::common::toLowerCopy(value);
  if (lowered == "onnx") {
    return PipelineConfig::ModelBackend::ONNX;
  }
  if (lowered == "rknn") {
    return PipelineConfig::ModelBackend::RKNN;
  }
  return PipelineConfig::ModelBackend::AUTO;
}

std::vector<std::string> loadClassNamesFromFile(const std::filesystem::path& path) {
  std::ifstream handle(path);
  if (!handle.is_open()) {
    return {};
  }

  std::vector<std::string> names;
  std::string line;
  while (std::getline(handle, line)) {
    line = trimCopy(line);
    if (!line.empty()) {
      names.push_back(line);
    }
  }
  return names;
}

void appendWarning(std::vector<std::string>* warnings, std::string message) {
  if (warnings != nullptr) {
    warnings->push_back(std::move(message));
  }
}

void requireAbsent(const YAML::Node& node,
                   const std::string& key,
                   const std::string& replacement) {
  if (node && node[key]) {
    throw std::runtime_error("Deprecated config key '" + key +
                             "' is no longer supported; use '" + replacement + "'.");
  }
}

void rejectDeprecatedSchema(const YAML::Node& yaml) {
  requireAbsent(yaml, "input", "source");
  requireAbsent(yaml, "nms", "postprocess");

  if (yaml["engine"]) {
    requireAbsent(yaml["engine"], "imgsz", "engine.input_size");
  }

  if (yaml["output"]) {
    requireAbsent(yaml["output"], "ip", "output.tcp.host");
    requireAbsent(yaml["output"], "queue", "output.tcp.queue_size");
    if (yaml["output"]["tcp"]) {
      requireAbsent(yaml["output"]["tcp"], "ip", "output.tcp.host");
      requireAbsent(yaml["output"]["tcp"], "queue", "output.tcp.queue_size");
    }
  }
}

void parseClassesNode(const YAML::Node& classes,
                      const std::filesystem::path& base_dir,
                      PipelineConfig::ModelSpec& model) {
  if (!classes) {
    return;
  }
  if (classes.IsScalar()) {
    model.class_names =
        loadClassNamesFromFile(resolveConfigPath(base_dir, classes.as<std::string>("")));
    return;
  }
  if (!classes.IsMap()) {
    return;
  }
  if (classes["path"]) {
    model.class_names = loadClassNamesFromFile(
        resolveConfigPath(base_dir, classes["path"].as<std::string>("")));
  }
  if (classes["names"]) {
    const auto& names = classes["names"];
    if (names.IsSequence()) {
      model.class_names.clear();
      for (const auto& node : names) {
        model.class_names.push_back(node.as<std::string>());
      }
    } else if (names.IsScalar()) {
      model.class_names =
          loadClassNamesFromFile(resolveConfigPath(base_dir, names.as<std::string>("")));
    }
  }
}

void parsePreprocessNode(const YAML::Node& preprocess,
                         const std::filesystem::path& base_dir,
                         PipelineConfig::PreprocessSpec& spec) {
  if (!preprocess || !preprocess.IsMap()) {
    return;
  }
  if (preprocess["profile"]) {
    spec.profile = preprocess["profile"].as<std::string>(spec.profile);
  }
  if (preprocess["use_rga_preprocess"]) {
    spec.use_rga_preprocess =
        preprocess["use_rga_preprocess"].as<bool>(spec.use_rga_preprocess);
  }
  if (preprocess["undistort"]) {
    const auto& undistort = preprocess["undistort"];
    if (undistort["enable"]) {
      spec.enable_undistort = undistort["enable"].as<bool>(spec.enable_undistort);
    }
    if (undistort["calibration_file"]) {
      spec.calibration_file =
          resolveConfigPath(base_dir, undistort["calibration_file"].as<std::string>("")).string();
    }
  }
  if (preprocess["roi"]) {
    const auto& roi = preprocess["roi"];
    if (roi["enable"]) {
      spec.roi_enable = roi["enable"].as<bool>(spec.roi_enable);
    }
    if (roi["mode"]) {
      spec.roi_mode = roi["mode"].as<std::string>(spec.roi_mode);
    }
    if (roi["normalized_xywh"] && roi["normalized_xywh"].IsSequence() &&
        roi["normalized_xywh"].size() == 4) {
      for (size_t i = 0; i < 4; ++i) {
        spec.roi_normalized_xywh[i] =
            roi["normalized_xywh"][i].as<float>(spec.roi_normalized_xywh[i]);
      }
    }
    if (roi["pixel_xywh"] && roi["pixel_xywh"].IsSequence() &&
        roi["pixel_xywh"].size() == 4) {
      for (size_t i = 0; i < 4; ++i) {
        spec.roi_pixel_xywh[i] = roi["pixel_xywh"][i].as<int>(spec.roi_pixel_xywh[i]);
      }
    }
    if (roi["clamp"]) {
      spec.roi_clamp = roi["clamp"].as<bool>(spec.roi_clamp);
    }
    if (roi["min_size"]) {
      spec.roi_min_size = roi["min_size"].as<int>(spec.roi_min_size);
    }
  }
  if (preprocess["gamma"]) {
    const auto& gamma = preprocess["gamma"];
    if (gamma["enable"]) {
      spec.gamma_enable = gamma["enable"].as<bool>();
    }
    if (gamma["value"]) {
      spec.gamma_value = gamma["value"].as<float>(spec.gamma_value);
    }
  }
  if (preprocess["white_balance"]) {
    const auto& wb = preprocess["white_balance"];
    if (wb["enable"]) {
      spec.white_balance_enable = wb["enable"].as<bool>();
    }
    if (wb["clip_percent"]) {
      spec.white_balance_clip_percent =
          wb["clip_percent"].as<float>(spec.white_balance_clip_percent);
    }
  }
  if (preprocess["denoise"]) {
    const auto& denoise = preprocess["denoise"];
    if (denoise["enable"]) {
      spec.denoise_enable = denoise["enable"].as<bool>();
    }
    if (denoise["method"]) {
      spec.denoise_method = denoise["method"].as<std::string>(spec.denoise_method);
    }
    if (denoise["d"]) {
      spec.denoise_d = denoise["d"].as<int>(spec.denoise_d);
    }
    if (denoise["sigma_color"]) {
      spec.denoise_sigma_color = denoise["sigma_color"].as<float>(spec.denoise_sigma_color);
    }
    if (denoise["sigma_space"]) {
      spec.denoise_sigma_space = denoise["sigma_space"].as<float>(spec.denoise_sigma_space);
    }
  }
}

void resolveModelMetadata(PipelineConfig::ModelSpec& model, std::vector<std::string>* warnings) {
  if (model.model_path.empty()) {
    return;
  }
  if (model.backend == PipelineConfig::ModelBackend::AUTO) {
    const std::string lowered = rkapp::common::toLowerCopy(model.model_path);
    if (lowered.rfind(".onnx") != std::string::npos) {
      model.backend = PipelineConfig::ModelBackend::ONNX;
    } else if (lowered.rfind(".rknn") != std::string::npos) {
      model.backend = PipelineConfig::ModelBackend::RKNN;
    }
  }
  if (modelMetaHasAny(model.decode_meta)) {
    return;
  }

  const auto load_result = infer::loadModelMetaFromPath(model.model_path);
  model.decode_meta = load_result.meta;
  model.decode_meta_path = load_result.source_path;
  if (!modelMetaHasAny(model.decode_meta)) {
    appendWarning(warnings, "No model-local decode metadata found for " + model.model_path);
  }
}

}  // namespace

PipelineConfig normalizePipelineConfig(const PipelineConfig& raw_config,
                                       std::vector<std::string>* warnings) {
  PipelineConfig config = raw_config;
  resolveModelMetadata(config.model, warnings);
  return config;
}

PipelineConfigLoadResult loadPipelineConfigFile(const std::string& config_path) {
  PipelineConfigLoadResult result;
  const std::filesystem::path path = std::filesystem::absolute(config_path);
  const std::filesystem::path base_dir = path.parent_path();
  const YAML::Node yaml = YAML::LoadFile(path.string());

  rejectDeprecatedSchema(yaml);

  if (yaml["source"]) {
    const auto& source = yaml["source"];
    if (source["type"]) {
      result.config.source.type = parseSourceType(source["type"].as<std::string>("video"));
    }
    if (source["uri"]) {
      result.config.source.uri =
          resolveConfigPath(base_dir, source["uri"].as<std::string>("")).string();
    }
    if (source["use_mpp_decode"]) {
      result.config.source.use_mpp_decode =
          source["use_mpp_decode"].as<bool>(result.config.source.use_mpp_decode);
    }
  }

  if (yaml["engine"]) {
    const auto& engine = yaml["engine"];
    if (engine["type"]) {
      result.config.model.backend = parseBackend(engine["type"].as<std::string>(""));
    }
    if (engine["model"]) {
      result.config.model.model_path =
          resolveConfigPath(base_dir, engine["model"].as<std::string>("")).string();
    }
    if (engine["input_size"]) {
      const auto& size = engine["input_size"];
      if (size.IsSequence() && size.size() > 0) {
        result.config.model.input_size = size[0].as<int>(result.config.model.input_size);
      } else if (size.IsScalar()) {
        result.config.model.input_size = size.as<int>(result.config.model.input_size);
      }
    }
    if (engine["use_npu_multicore"]) {
      result.config.model.use_npu_multicore =
          engine["use_npu_multicore"].as<bool>(result.config.model.use_npu_multicore);
    }
    if (engine["use_zero_copy"]) {
      result.config.model.use_zero_copy =
          engine["use_zero_copy"].as<bool>(result.config.model.use_zero_copy);
    }
    if (engine["buffer_pool_size"]) {
      result.config.model.buffer_pool_size =
          engine["buffer_pool_size"].as<int>(result.config.model.buffer_pool_size);
    }
  }

  if (yaml["postprocess"]) {
    const auto& post = yaml["postprocess"];
    if (post["conf_threshold"]) {
      result.config.model.conf_threshold =
          post["conf_threshold"].as<float>(result.config.model.conf_threshold);
    }
    if (post["nms_threshold"]) {
      result.config.model.iou_threshold =
          post["nms_threshold"].as<float>(result.config.model.iou_threshold);
    }
    if (post["max_detections"]) {
      result.config.model.max_detections =
          post["max_detections"].as<int>(result.config.model.max_detections);
    }
    if (post["min_box_size"]) {
      result.config.model.min_box_size =
          post["min_box_size"].as<float>(result.config.model.min_box_size);
    }
    if (post["max_box_size"]) {
      result.config.model.max_box_size =
          post["max_box_size"].as<float>(result.config.model.max_box_size);
    }
    if (post["aspect_ratio_range"] && post["aspect_ratio_range"].IsSequence() &&
        post["aspect_ratio_range"].size() == 2) {
      result.config.model.min_aspect_ratio =
          post["aspect_ratio_range"][0].as<float>(result.config.model.min_aspect_ratio);
      result.config.model.max_aspect_ratio =
          post["aspect_ratio_range"][1].as<float>(result.config.model.max_aspect_ratio);
    }
  }

  parseClassesNode(yaml["classes"], base_dir, result.config.model);
  parsePreprocessNode(yaml["preprocess"], base_dir, result.config.preprocess);

  if (yaml["output"]) {
    const auto& output = yaml["output"];
    if (output["type"]) {
      result.config.output.type = output["type"].as<std::string>(result.config.output.type);
    }
    if (output["enable_profiling"]) {
      result.config.output.enable_profiling =
          output["enable_profiling"].as<bool>(result.config.output.enable_profiling);
    }
    if (output["tcp"]) {
      const auto& tcp = output["tcp"];
      if (tcp["host"]) {
        result.config.output.host = tcp["host"].as<std::string>(result.config.output.host);
      }
      if (tcp["port"]) {
        result.config.output.port = tcp["port"].as<int>(result.config.output.port);
      }
      if (tcp["queue_size"]) {
        result.config.output.queue_size =
            tcp["queue_size"].as<int>(result.config.output.queue_size);
      }
    }
  }

  if (yaml["failure"]) {
    const auto& failure = yaml["failure"];
    if (failure["frame_error_limit"]) {
      result.config.failure.frame_error_limit =
          failure["frame_error_limit"].as<int>(result.config.failure.frame_error_limit);
    }
    if (failure["source_recovery_grace_ms"]) {
      result.config.failure.source_recovery_grace_ms =
          failure["source_recovery_grace_ms"].as<int>(
              result.config.failure.source_recovery_grace_ms);
    }
    if (failure["max_reconnect_attempts"]) {
      result.config.failure.max_reconnect_attempts =
          failure["max_reconnect_attempts"].as<int>(
              result.config.failure.max_reconnect_attempts);
    }
    if (failure["initial_backoff_ms"]) {
      result.config.failure.initial_backoff_ms =
          failure["initial_backoff_ms"].as<int>(result.config.failure.initial_backoff_ms);
    }
    if (failure["max_backoff_ms"]) {
      result.config.failure.max_backoff_ms =
          failure["max_backoff_ms"].as<int>(result.config.failure.max_backoff_ms);
    }
  }

  result.config = normalizePipelineConfig(result.config, &result.warnings);
  return result;
}

}  // namespace rkapp::pipeline
