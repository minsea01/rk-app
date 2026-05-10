#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rkapp/common/StringUtils.hpp"
#include "rkapp/common/log.hpp"
#include "rkapp/output/TcpOutput.hpp"
#include "rkapp/pipeline/DetectionPipeline.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

namespace {

struct CliOptions {
  std::string config_path = "config/detect.yaml";
  std::string source_override;
  std::string model_override;
  std::string save_vis_dir;
  std::string json_output_file;
  std::string log_level_override;
  std::string undistort_calib_override;
  std::string preprocess_profile_override;
  std::optional<int> warmup_iterations_override;
  bool async_mode_override = false;
};

void printUsage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]\n"
            << "Options:\n"
            << "  --cfg <path>         Configuration file (default: config/detect.yaml)\n"
            << "  --source <uri>       Source URI (overrides config)\n"
            << "  --model <path>       Model path (overrides config)\n"
            << "  --save_vis <dir>     Save visualized results to directory\n"
            << "  --json <file>        Save JSON results to file\n"
            << "  --warmup <N>         Override runtime warmup iterations\n"
            << "  --async              Enable async pipeline mode\n"
            << "  --undistort-calib <path>  Override calibration file and enable undistort\n"
            << "  --pp-profile <name>  Preprocess profile: speed|balanced|quality\n"
            << "  --log-level <lvl>    Set log level (TRACE/DEBUG/INFO/WARN/ERROR)\n"
            << "  --help               Show this help message\n";
}

rklog::Level parseLogLevel(const std::string& level_name) {
  std::string upper;
  upper.reserve(level_name.size());
  for (char c : level_name) {
    upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  if (upper == "TRACE") return rklog::TRACE;
  if (upper == "DEBUG") return rklog::DEBUG;
  if (upper == "INFO") return rklog::INFO;
  if (upper == "WARN" || upper == "WARNING") return rklog::WARN;
  if (upper == "ERROR") return rklog::ERROR;
  LOGW("Unknown log level '", level_name, "', defaulting to INFO");
  return rklog::INFO;
}

std::string sourceTypeName(rkapp::capture::SourceType type) {
  switch (type) {
    case rkapp::capture::SourceType::FOLDER:
      return "folder";
    case rkapp::capture::SourceType::VIDEO:
      return "video";
    case rkapp::capture::SourceType::RTSP:
      return "rtsp";
    case rkapp::capture::SourceType::GIGE:
      return "gige";
    case rkapp::capture::SourceType::MPP:
      return "mpp";
    case rkapp::capture::SourceType::CSI:
      return "csi";
  }
  return "video";
}

std::string backendName(rkapp::pipeline::PipelineConfig::ModelBackend backend) {
  switch (backend) {
    case rkapp::pipeline::PipelineConfig::ModelBackend::AUTO:
      return "auto";
    case rkapp::pipeline::PipelineConfig::ModelBackend::ONNX:
      return "onnx";
    case rkapp::pipeline::PipelineConfig::ModelBackend::RKNN:
      return "rknn";
  }
  return "auto";
}

void drawDetections(cv::Mat& image, const std::vector<rkapp::infer::Detection>& detections) {
  for (const auto& det : detections) {
    cv::Rect box(static_cast<int>(det.x), static_cast<int>(det.y), static_cast<int>(det.w),
                 static_cast<int>(det.h));
    box &= cv::Rect(0, 0, image.cols, image.rows);
    if (box.width <= 0 || box.height <= 0) {
      continue;
    }

    cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

    const std::string label =
        det.class_name + " " + std::to_string(det.confidence).substr(0, 4);
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int label_x = std::max(0, box.x);
    const int label_y = std::max(text_size.height + baseline, box.y);
    cv::rectangle(image,
                  cv::Rect(label_x, label_y - text_size.height - baseline, text_size.width,
                           text_size.height + baseline),
                  cv::Scalar(0, 255, 0), -1);
    cv::putText(image, label, cv::Point(label_x, label_y - baseline),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }
}

bool modeIsNormalized(const std::string& mode) {
  return rkapp::common::toLowerCopy(mode) != "pixel";
}

std::vector<rkapp::infer::Detection> detectionsForImageRoi(
    const std::vector<rkapp::infer::Detection>& detections, const cv::Rect& roi) {
  std::vector<rkapp::infer::Detection> adjusted;
  adjusted.reserve(detections.size());
  const cv::Rect image_rect(0, 0, roi.width, roi.height);
  for (const auto& det : detections) {
    cv::Rect box(static_cast<int>(std::round(det.x - roi.x)),
                 static_cast<int>(std::round(det.y - roi.y)),
                 static_cast<int>(std::round(det.w)),
                 static_cast<int>(std::round(det.h)));
    box &= image_rect;
    if (box.width <= 0 || box.height <= 0) {
      continue;
    }
    auto shifted = det;
    shifted.x = static_cast<float>(box.x);
    shifted.y = static_cast<float>(box.y);
    shifted.w = static_cast<float>(box.width);
    shifted.h = static_cast<float>(box.height);
    for (auto& keypoint : shifted.keypoints) {
      keypoint.x -= static_cast<float>(roi.x);
      keypoint.y -= static_cast<float>(roi.y);
    }
    adjusted.push_back(std::move(shifted));
  }
  return adjusted;
}

std::optional<CliOptions> parseCliOptions(int argc, char* argv[]) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--cfg" && i + 1 < argc) {
      options.config_path = argv[++i];
    } else if (arg == "--source" && i + 1 < argc) {
      options.source_override = argv[++i];
    } else if (arg == "--model" && i + 1 < argc) {
      options.model_override = argv[++i];
    } else if (arg == "--save_vis" && i + 1 < argc) {
      options.save_vis_dir = argv[++i];
    } else if (arg == "--json" && i + 1 < argc) {
      options.json_output_file = argv[++i];
    } else if (arg == "--warmup" && i + 1 < argc) {
      try {
        options.warmup_iterations_override = std::stoi(argv[++i]);
      } catch (const std::exception&) {
        std::fprintf(stderr, "Invalid value for --warmup: %s\n", argv[i]);
        return std::nullopt;
      }
    } else if (arg == "--async") {
      options.async_mode_override = true;
    } else if (arg == "--log-level" && i + 1 < argc) {
      options.log_level_override = argv[++i];
    } else if (arg == "--undistort-calib" && i + 1 < argc) {
      options.undistort_calib_override = argv[++i];
    } else if (arg == "--pp-profile" && i + 1 < argc) {
      options.preprocess_profile_override = argv[++i];
    } else if (arg == "--help") {
      printUsage(argv[0]);
      return std::nullopt;
    } else {
      LOGE("Unknown argument: ", arg);
      printUsage(argv[0]);
      return std::nullopt;
    }
  }
  return options;
}

rkapp::pipeline::PipelineConfig loadConfig(const std::string& config_path) {
  rkapp::pipeline::PipelineConfig config;
  if (!std::filesystem::exists(config_path)) {
    LOGW("Configuration file not found: ", config_path, ", using defaults");
    return config;
  }

  try {
    auto loaded = rkapp::pipeline::loadPipelineConfigFile(config_path);
    for (const auto& warning : loaded.warnings) {
      LOGW("Config: ", warning);
    }
    LOGI("Loaded configuration from ", config_path);
    return loaded.config;
  } catch (const std::exception& e) {
    LOGE("Error loading config ", config_path, ": ", e.what());
    return config;
  }
}

void applyCliOverrides(rkapp::pipeline::PipelineConfig& config, const CliOptions& options) {
  if (!options.source_override.empty()) {
    config.source.uri = options.source_override;
  }
  if (!options.model_override.empty()) {
    config.model.model_path = options.model_override;
  }
  if (!options.undistort_calib_override.empty()) {
    config.preprocess.enable_undistort = true;
    config.preprocess.calibration_file = options.undistort_calib_override;
  }
  if (!options.preprocess_profile_override.empty()) {
    config.preprocess.profile = options.preprocess_profile_override;
  }
  if (options.warmup_iterations_override.has_value()) {
    config.runtime.warmup_iterations = *options.warmup_iterations_override;
  }
  if (options.async_mode_override) {
    config.runtime.async_mode = true;
  }
}

std::unique_ptr<rkapp::output::IOutput> createOutput(
    const rkapp::pipeline::PipelineConfig& config, const std::string& json_output_file) {
  const std::string normalized_output_type = rkapp::common::toLowerCopy(config.output.type);
  if (normalized_output_type != "tcp") {
    LOGE("Unsupported output.type='", config.output.type,
         "'. Only 'tcp' is implemented in this build.");
    return nullptr;
  }

  auto output = std::make_unique<rkapp::output::TcpOutput>();
  std::string output_config =
      config.output.host + ":" + std::to_string(config.output.port);
  if (config.output.queue_size > 0) {
    output_config += ",queue:" + std::to_string(config.output.queue_size);
  }
  if (!config.output.bind_ip.empty()) {
    output_config += ",bind_ip:" + config.output.bind_ip;
  }
  if (!config.output.bind_interface.empty()) {
    output_config += ",iface:" + config.output.bind_interface;
  }
  if (!json_output_file.empty()) {
    output_config += ",file:" + json_output_file;
  }
  if (!output->open(output_config)) {
    LOGW("Failed to open output '", output_config, "', continuing without output");
    return nullptr;
  }
  return output;
}

struct EncodedOutputImage {
  std::vector<uint8_t> bytes;
  int width = 0;
  int height = 0;
  bool roi_applied = false;
  cv::Rect roi;
};

void saveVisualization(const rkapp::pipeline::PipelineResult& result,
                       const std::filesystem::path& output_dir) {
  if (result.frame.empty()) {
    return;
  }
  cv::Mat vis_frame = result.frame.clone();
  drawDetections(vis_frame, result.detections);
  const std::filesystem::path vis_path =
      output_dir / ("frame_" + std::to_string(result.frame_id) + ".jpg");
  cv::imwrite(vis_path.string(), vis_frame);
}

EncodedOutputImage encodeOutputImage(
    const rkapp::pipeline::PipelineResult& result,
    const rkapp::pipeline::PipelineConfig::OutputSpec& output) {
  EncodedOutputImage encoded_image;
  if (!output.include_image || result.frame.empty()) {
    return encoded_image;
  }

  const int interval = std::max(1, output.image_interval);
  if (result.frame_id < 0 || (result.frame_id % interval) != 0) {
    return encoded_image;
  }

  cv::Mat image = result.frame;
  std::vector<rkapp::infer::Detection> display_detections = result.detections;
  cv::Rect image_roi(0, 0, result.frame.cols, result.frame.rows);
  if (output.image_roi_enable) {
    const cv::Rect2f normalized_roi(
        output.image_roi_normalized_xywh[0],
        output.image_roi_normalized_xywh[1],
        output.image_roi_normalized_xywh[2],
        output.image_roi_normalized_xywh[3]);
    const cv::Rect pixel_roi(
        output.image_roi_pixel_xywh[0],
        output.image_roi_pixel_xywh[1],
        output.image_roi_pixel_xywh[2],
        output.image_roi_pixel_xywh[3]);
    cv::Rect resolved_roi;
    if (rkapp::preprocess::Preprocess::resolveRoiRect(
            result.frame.size(), modeIsNormalized(output.image_roi_mode), normalized_roi,
            pixel_roi, output.image_roi_clamp, output.image_roi_min_size, resolved_roi)) {
      image_roi = resolved_roi;
      image = result.frame(image_roi);
      display_detections = detectionsForImageRoi(result.detections, image_roi);
      encoded_image.roi_applied = image_roi.x != 0 || image_roi.y != 0 ||
                                  image_roi.width != result.frame.cols ||
                                  image_roi.height != result.frame.rows;
    } else {
      LOGW("Invalid output image_roi; sending full frame");
    }
  }

  if (output.draw_detections && !display_detections.empty()) {
    if (image.channels() == 1) {
      cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    } else {
      image = image.clone();
    }
    drawDetections(image, display_detections);
  }

  const int quality = std::clamp(output.image_quality, 1, 100);
  if (!cv::imencode(".jpg", image, encoded_image.bytes, {cv::IMWRITE_JPEG_QUALITY, quality})) {
    LOGW("Failed to encode frame ", result.frame_id, " as JPEG for uplink");
    encoded_image.bytes.clear();
    return encoded_image;
  }
  encoded_image.width = image.cols;
  encoded_image.height = image.rows;
  encoded_image.roi = image_roi;
  return encoded_image;
}

void publishResult(const rkapp::pipeline::PipelineResult& result, rkapp::output::IOutput* output,
                   const rkapp::pipeline::PipelineConfig& config, const std::string& source_uri) {
  if (output == nullptr) {
    return;
  }

  rkapp::output::FrameResult frame_result;
  frame_result.frame_id = static_cast<int>(result.frame_id);
  frame_result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
  frame_result.width = result.frame.cols;
  frame_result.height = result.frame.rows;
  frame_result.detections = result.detections;
  frame_result.source_uri = source_uri;
  auto encoded_image = encodeOutputImage(result, config.output);
  frame_result.image_bytes = std::move(encoded_image.bytes);
  if (!frame_result.image_bytes.empty()) {
    frame_result.image_encoding = "jpeg";
    frame_result.image_contains_overlays = config.output.draw_detections;
    frame_result.image_width = encoded_image.width;
    frame_result.image_height = encoded_image.height;
    frame_result.image_roi_applied = encoded_image.roi_applied;
    frame_result.image_roi_x = encoded_image.roi.x;
    frame_result.image_roi_y = encoded_image.roi.y;
    frame_result.image_roi_w = encoded_image.roi.width;
    frame_result.image_roi_h = encoded_image.roi.height;
  }
  output->send(frame_result);
}

void logPerFrame(const rkapp::pipeline::PipelineResult& result) {
  const double elapsed_ms = static_cast<double>(result.timing.total_us) / 1000.0;
  LOGI("Frame ", result.frame_id, ": ", result.detections.size(), " detections (", elapsed_ms,
       " ms)");
}

void logSummary(const rkapp::pipeline::DetectionPipeline& pipeline) {
  const auto stats = pipeline.getStatistics();
  LOGI("=== Pipeline Summary ===");
  LOGI("Frames processed: ", stats.frames_processed);
  LOGI("Frames dropped: ", stats.frames_dropped);
  LOGI("Reconnect attempts: ", stats.reconnect_count);
  LOGI("Total detections: ", stats.total_detections);
  LOGI("Average FPS: ", stats.avg_fps);
  LOGI("Average latency: ", stats.avg_latency_ms, " ms");
  LOGI("RGA preprocess: ", (stats.rga_enabled ? "enabled" : "disabled"));
  LOGI("MPP decode: ", (stats.mpp_enabled ? "enabled" : "disabled"));
  LOGI("Zero-copy: ", (stats.zero_copy_enabled ? "enabled" : "disabled"));
}

}  // namespace

int main(int argc, char* argv[]) {
  auto cli_options = parseCliOptions(argc, argv);
  if (!cli_options.has_value()) {
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--help") {
        return 0;
      }
    }
    return 1;
  }

  CliOptions options = *cli_options;
  auto config = loadConfig(options.config_path);
  applyCliOverrides(config, options);

  const std::string effective_log_level = options.log_level_override.empty()
      ? config.logging.level
      : options.log_level_override;
  rklog::g_level.store(parseLogLevel(effective_log_level), std::memory_order_relaxed);

  if (!options.save_vis_dir.empty()) {
    std::filesystem::create_directories(options.save_vis_dir);
  }

  LOGI("=== Object Detection Pipeline ===");
  LOGI("Source: ", sourceTypeName(config.source.type), " (", config.source.uri, ")");
  LOGI("Engine: ", backendName(config.model.backend), " (", config.model.model_path, ")");
  LOGI("Input size: ", config.model.input_size);
  LOGI("Thresholds: conf=", config.model.conf_threshold, ", iou=", config.model.iou_threshold);
  LOGI("Undistort: ", (config.preprocess.enable_undistort ? "enabled" : "disabled"));
  LOGI("Preprocess profile: ", config.preprocess.profile);
  LOGI("Output: ", config.output.type, " -> ", config.output.host, ":", config.output.port);
  LOGI("Output bind_ip: ",
       (config.output.bind_ip.empty() ? std::string("(system route)") : config.output.bind_ip));
  LOGI("Output iface: ",
       (config.output.bind_interface.empty() ? std::string("(system route)")
                                             : config.output.bind_interface));
  LOGI("Output image uplink: ", (config.output.include_image ? "enabled" : "disabled"),
       " (quality=", config.output.image_quality,
       ", interval=", config.output.image_interval,
       ", overlays=", (config.output.draw_detections ? "on" : "off"), ")");
  LOGI("Warmup: ", config.runtime.warmup_iterations);
  LOGI("Async mode: ", (config.runtime.async_mode ? "enabled" : "disabled"));

  rkapp::pipeline::DetectionPipeline pipeline;
  if (!pipeline.init(config)) {
    LOGE("Failed to initialize DetectionPipeline");
    return 1;
  }
  if (rkapp::common::toLowerCopy(config.output.type) != "tcp") {
    return 1;
  }
  auto output = createOutput(config, options.json_output_file);
  const std::filesystem::path vis_dir(options.save_vis_dir);

  std::atomic<int64_t> handled_frames{0};
  auto handle_result = [&](const rkapp::pipeline::PipelineResult& result) {
    logPerFrame(result);
    if (!options.save_vis_dir.empty()) {
      saveVisualization(result, vis_dir);
    }
    publishResult(result, output.get(), config, config.source.uri);
    handled_frames.fetch_add(1, std::memory_order_relaxed);
  };

  if (!config.runtime.async_mode) {
    while (auto result = pipeline.next()) {
      handle_result(*result);
    }
  } else {
    pipeline.runAsync([&](rkapp::pipeline::PipelineResult&& result) { handle_result(result); });
    while (pipeline.isRunning()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  pipeline.stop();
  logSummary(pipeline);
  LOGI("detect_cli: handled ", handled_frames.load(std::memory_order_relaxed), " frames");
  return 0;
}
