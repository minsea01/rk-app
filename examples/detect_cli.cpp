#include <iostream>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <optional>
#include <yaml-cpp/yaml.h>
#include <vector>

#include "rkapp/capture/ISource.hpp"
#include "rkapp/infer/IInferEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "rkapp/output/IOutput.hpp"
#include "rkapp/common/StringUtils.hpp"
#include "rkapp/common/log.hpp"

// Concrete class headers
#include "rkapp/capture/FolderSource.hpp"
#include "rkapp/capture/VideoSource.hpp"
#ifdef RKAPP_WITH_GIGE
#include "rkapp/capture/GigeSource.hpp"
#endif
#ifdef RKAPP_WITH_CSI
#include "rkapp/capture/CsiSource.hpp"
#endif
#ifdef RKAPP_WITH_ONNX
#include "rkapp/infer/OnnxEngine.hpp"
#endif
#ifdef RKAPP_WITH_RKNN
#include "rkapp/infer/RknnEngine.hpp"
#endif
#include "rkapp/output/TcpOutput.hpp"
#include <thread>
#include <condition_variable>
#include <deque>

struct Config {
    std::string source_type = "folder";
    std::string source_uri;
    std::string engine_type = "onnx";
    std::string model_path;
    int imgsz = 640;
    float conf_thres = 0.25f;
    float iou_thres = 0.60f;
    int nms_topk = 1000;
    int warmup = 5;
    bool async = false;
    std::string log_level = "INFO";
    std::string output_type = "tcp";
    std::string output_ip = "127.0.0.1";
    int output_port = 9000;
    int output_queue = 0;
    std::string classes_path = "config/classes.txt";
    std::vector<std::string> inline_class_names;
    float min_box_size = 0.0f;
    float max_box_size = 0.0f;
    float min_aspect_ratio = 0.0f;
    float max_aspect_ratio = 0.0f;
    bool enable_undistort = false;
    std::string calibration_file;
    std::string preprocess_profile = "speed";
    bool roi_enable = false;
    std::string roi_mode = "normalized";
    std::array<float, 4> roi_normalized_xywh{0.0f, 0.0f, 1.0f, 1.0f};
    std::array<int, 4> roi_pixel_xywh{0, 0, 0, 0};
    bool roi_clamp = true;
    int roi_min_size = 8;
    std::optional<bool> gamma_enable;
    float gamma_value = 1.0f;
    std::optional<bool> white_balance_enable;
    float white_balance_clip_percent = 0.0f;
    std::optional<bool> denoise_enable;
    std::string denoise_method = "bilateral";
    int denoise_d = 5;
    float denoise_sigma_color = 35.0f;
    float denoise_sigma_space = 35.0f;
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\\n";
    std::cout << "Options:\\n";
    std::cout << "  --cfg <path>         Configuration file (default: config/detect.yaml)\\n";
    std::cout << "  --source <uri>       Source URI (overrides config)\\n";
    std::cout << "  --save_vis <dir>     Save visualized results to directory\\n";
    std::cout << "  --json <file>        Save JSON results to file\\n";
    std::cout << "  --warmup <N>         Warmup iterations before timing (default: 5)\\n";
    std::cout << "  --async              Enable async capture->infer pipeline\\n";
    std::cout << "  --undistort-calib <path>  Override calibration file and enable undistort\\n";
    std::cout << "  --pp-profile <name>  Preprocess profile: speed|balanced|quality\\n";
    std::cout << "  --log-level <lvl>    Set log level (TRACE/DEBUG/INFO/WARN/ERROR)\\n";
    std::cout << "  --help               Show this help message\\n";
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

bool shouldRetryRead(const std::string& source_type, rkapp::capture::SourceType actual_type) {
    if (source_type == "rtsp" || source_type == "gige" || source_type == "csi") {
        return true;
    }
    return actual_type == rkapp::capture::SourceType::RTSP ||
           actual_type == rkapp::capture::SourceType::GIGE ||
           actual_type == rkapp::capture::SourceType::CSI;
}

struct FeatureFlags {
    bool gamma = false;
    bool white_balance = false;
    bool denoise = false;
};

FeatureFlags resolveFeatureFlags(const Config& config) {
    FeatureFlags flags;
    const std::string profile = rkapp::common::toLowerCopy(config.preprocess_profile);
    if (profile == "balanced") {
        flags.gamma = true;
        flags.white_balance = true;
    } else if (profile == "quality") {
        flags.gamma = true;
        flags.white_balance = true;
        flags.denoise = true;
    }
    if (config.gamma_enable.has_value()) {
        flags.gamma = *config.gamma_enable;
    }
    if (config.white_balance_enable.has_value()) {
        flags.white_balance = *config.white_balance_enable;
    }
    if (config.denoise_enable.has_value()) {
        flags.denoise = *config.denoise_enable;
    }
    return flags;
}

void applyRoiOffsetAndClip(std::vector<rkapp::infer::Detection>& detections,
                           const cv::Rect& roi, cv::Size frame_size) {
    if (frame_size.width <= 0 || frame_size.height <= 0) {
        return;
    }
    const float max_x = static_cast<float>(frame_size.width - 1);
    const float max_y = static_cast<float>(frame_size.height - 1);
    for (auto& det : detections) {
        float x1 = det.x + static_cast<float>(roi.x);
        float y1 = det.y + static_cast<float>(roi.y);
        float x2 = x1 + det.w;
        float y2 = y1 + det.h;

        x1 = std::clamp(x1, 0.0f, max_x);
        y1 = std::clamp(y1, 0.0f, max_y);
        x2 = std::clamp(x2, 0.0f, max_x);
        y2 = std::clamp(y2, 0.0f, max_y);

        det.x = x1;
        det.y = y1;
        det.w = std::max(0.0f, x2 - x1);
        det.h = std::max(0.0f, y2 - y1);
    }
}

Config loadConfig(const std::string& config_path) {
    Config config;
    
    try {
        if (std::filesystem::exists(config_path)) {
            YAML::Node yaml = YAML::LoadFile(config_path);
            
            if (yaml["source"]) {
                config.source_type = yaml["source"]["type"].as<std::string>("folder");
                config.source_uri = yaml["source"]["uri"].as<std::string>("");
            }
            
            if (yaml["engine"]) {
                config.engine_type = yaml["engine"]["type"].as<std::string>("onnx");
                config.model_path = yaml["engine"]["model"].as<std::string>("");
                config.imgsz = yaml["engine"]["imgsz"].as<int>(640);
                if (yaml["engine"]["input_size"] && yaml["engine"]["input_size"].IsSequence()) {
                    const auto& s = yaml["engine"]["input_size"];
                    if (s.size() > 0) {
                        config.imgsz = s[0].as<int>(config.imgsz);
                    }
                }
            }

            if (yaml["nms"]) {
                config.conf_thres = yaml["nms"]["conf_thres"].as<float>(0.25f);
                config.iou_thres = yaml["nms"]["iou_thres"].as<float>(0.60f);
                if (yaml["nms"]["topk"]) config.nms_topk = yaml["nms"]["topk"].as<int>(1000);
                if (yaml["nms"]["min_box_size"]) config.min_box_size = yaml["nms"]["min_box_size"].as<float>(config.min_box_size);
                if (yaml["nms"]["max_box_size"]) config.max_box_size = yaml["nms"]["max_box_size"].as<float>(config.max_box_size);
                if (yaml["nms"]["aspect_ratio_range"] && yaml["nms"]["aspect_ratio_range"].IsSequence() && yaml["nms"]["aspect_ratio_range"].size() == 2) {
                    config.min_aspect_ratio = yaml["nms"]["aspect_ratio_range"][0].as<float>(config.min_aspect_ratio);
                    config.max_aspect_ratio = yaml["nms"]["aspect_ratio_range"][1].as<float>(config.max_aspect_ratio);
                }
            }

            if (yaml["postprocess"]) {
                const auto& post = yaml["postprocess"];
                if (post["min_box_size"]) config.min_box_size = post["min_box_size"].as<float>(config.min_box_size);
                if (post["max_box_size"]) config.max_box_size = post["max_box_size"].as<float>(config.max_box_size);
                if (post["aspect_ratio_range"] && post["aspect_ratio_range"].IsSequence() && post["aspect_ratio_range"].size() == 2) {
                    config.min_aspect_ratio = post["aspect_ratio_range"][0].as<float>(config.min_aspect_ratio);
                    config.max_aspect_ratio = post["aspect_ratio_range"][1].as<float>(config.max_aspect_ratio);
                }
                if (post["conf_threshold"]) config.conf_thres = post["conf_threshold"].as<float>(config.conf_thres);
                if (post["nms_threshold"]) config.iou_thres = post["nms_threshold"].as<float>(config.iou_thres);
            }

            if (yaml["preprocess"]) {
                const auto& preprocess = yaml["preprocess"];
                if (preprocess["profile"]) {
                    config.preprocess_profile =
                        preprocess["profile"].as<std::string>(config.preprocess_profile);
                }
                if (preprocess["undistort"]) {
                    const auto& undistort = preprocess["undistort"];
                    if (undistort["enable"]) {
                        config.enable_undistort = undistort["enable"].as<bool>(config.enable_undistort);
                    }
                    if (undistort["calibration_file"]) {
                        config.calibration_file =
                            undistort["calibration_file"].as<std::string>(config.calibration_file);
                    }
                }
                if (preprocess["roi"]) {
                    const auto& roi = preprocess["roi"];
                    if (roi["enable"]) {
                        config.roi_enable = roi["enable"].as<bool>(config.roi_enable);
                    }
                    if (roi["mode"]) {
                        config.roi_mode = roi["mode"].as<std::string>(config.roi_mode);
                    }
                    if (roi["normalized_xywh"] && roi["normalized_xywh"].IsSequence() &&
                        roi["normalized_xywh"].size() == 4) {
                        for (size_t i = 0; i < 4; ++i) {
                            config.roi_normalized_xywh[i] =
                                roi["normalized_xywh"][i].as<float>(config.roi_normalized_xywh[i]);
                        }
                    }
                    if (roi["pixel_xywh"] && roi["pixel_xywh"].IsSequence() &&
                        roi["pixel_xywh"].size() == 4) {
                        for (size_t i = 0; i < 4; ++i) {
                            config.roi_pixel_xywh[i] =
                                roi["pixel_xywh"][i].as<int>(config.roi_pixel_xywh[i]);
                        }
                    }
                    if (roi["clamp"]) {
                        config.roi_clamp = roi["clamp"].as<bool>(config.roi_clamp);
                    }
                    if (roi["min_size"]) {
                        config.roi_min_size = roi["min_size"].as<int>(config.roi_min_size);
                    }
                }
                if (preprocess["gamma"]) {
                    const auto& gamma = preprocess["gamma"];
                    if (gamma["enable"]) {
                        config.gamma_enable = gamma["enable"].as<bool>();
                    }
                    if (gamma["value"]) {
                        config.gamma_value = gamma["value"].as<float>(config.gamma_value);
                    }
                }
                if (preprocess["white_balance"]) {
                    const auto& wb = preprocess["white_balance"];
                    if (wb["enable"]) {
                        config.white_balance_enable = wb["enable"].as<bool>();
                    }
                    if (wb["clip_percent"]) {
                        config.white_balance_clip_percent =
                            wb["clip_percent"].as<float>(config.white_balance_clip_percent);
                    }
                }
                if (preprocess["denoise"]) {
                    const auto& denoise = preprocess["denoise"];
                    if (denoise["enable"]) {
                        config.denoise_enable = denoise["enable"].as<bool>();
                    }
                    if (denoise["method"]) {
                        config.denoise_method =
                            denoise["method"].as<std::string>(config.denoise_method);
                    }
                    if (denoise["d"]) {
                        config.denoise_d = denoise["d"].as<int>(config.denoise_d);
                    }
                    if (denoise["sigma_color"]) {
                        config.denoise_sigma_color =
                            denoise["sigma_color"].as<float>(config.denoise_sigma_color);
                    }
                    if (denoise["sigma_space"]) {
                        config.denoise_sigma_space =
                            denoise["sigma_space"].as<float>(config.denoise_sigma_space);
                    }
                }
            }

            // Alternate schema: input.* (used by detect_pedestrian.yaml)
            if (yaml["input"]) {
                const auto& input = yaml["input"];
                if (input["type"]) {
                    config.source_type = input["type"].as<std::string>(config.source_type);
                }
                auto try_assign = [&](const YAML::Node& node, const std::string& key) {
                    if (node && node[key]) config.source_uri = node[key].as<std::string>(config.source_uri);
                };
                if (config.source_type == "gige") {
                    try_assign(input["gige"], "pipeline");
                    if (config.source_uri.empty()) try_assign(input, "pipeline");
                } else if (config.source_type == "csi") {
                    try_assign(input["csi"], "pipeline");
                    if (config.source_uri.empty()) try_assign(input, "pipeline");
                } else if (config.source_type == "video" || config.source_type == "rtsp") {
                    try_assign(input["video"], "path");
                    if (config.source_uri.empty()) try_assign(input, "path");
                } else if (config.source_type == "image" || config.source_type == "folder") {
                    try_assign(input["image"], "path");
                    if (config.source_uri.empty()) try_assign(input, "path");
                }
            }

            if (yaml["perf"]) {
                config.warmup = yaml["perf"]["warmup"].as<int>(5);
                config.async = yaml["perf"]["async"].as<bool>(false);
            }
            
            if (yaml["output"]) {
                const auto& out = yaml["output"];
                config.output_type = out["type"].as<std::string>("tcp");
                if (out["ip"]) config.output_ip = out["ip"].as<std::string>(config.output_ip);
                if (out["port"]) config.output_port = out["port"].as<int>(config.output_port);
                if (out["queue"]) config.output_queue = out["queue"].as<int>(config.output_queue);

                if (out["tcp"]) {
                    const auto& tcp = out["tcp"];
                    if (tcp["host"]) config.output_ip = tcp["host"].as<std::string>(config.output_ip);
                    if (tcp["port"]) config.output_port = tcp["port"].as<int>(config.output_port);
                    if (tcp["queue"]) config.output_queue = tcp["queue"].as<int>(config.output_queue);
                }
            }

            if (yaml["classes"]) {
                const auto& cls = yaml["classes"];
                if (cls.IsScalar()) {
                    config.classes_path = cls.as<std::string>("config/classes.txt");
                    config.inline_class_names.clear();
                } else if (cls.IsSequence()) {
                    config.classes_path.clear();
                    config.inline_class_names.clear();
                    for (const auto& node : cls) {
                        config.inline_class_names.push_back(node.as<std::string>());
                    }
                } else if (cls.IsMap()) {
                    if (cls["path"]) {
                        config.classes_path = cls["path"].as<std::string>("config/classes.txt");
                    }
                    if (cls["names"] && cls["names"].IsSequence()) {
                        config.inline_class_names.clear();
                        for (const auto& node : cls["names"]) {
                            config.inline_class_names.push_back(node.as<std::string>());
                        }
                        if (!cls["path"]) {
                            config.classes_path.clear();
                        }
                    }
                }
            }

            if (yaml["logging"]) {
                const auto& logging = yaml["logging"];
                if (logging["level"]) {
                    config.log_level = logging["level"].as<std::string>(config.log_level);
                }
            } else if (yaml["log_level"]) {
                config.log_level = yaml["log_level"].as<std::string>(config.log_level);
            }

            LOGI("Loaded configuration from ", config_path);
        } else {
            LOGW("Configuration file not found: ", config_path, ", using defaults");
        }
    } catch (const YAML::Exception& e) {
        LOGE("Error loading config: ", e.what());
    }
    
    return config;
}

void drawDetections(cv::Mat& image, const std::vector<rkapp::infer::Detection>& detections) {
    for (const auto& det : detections) {
        // Draw bounding box
        cv::Rect box(static_cast<int>(det.x), static_cast<int>(det.y), 
                     static_cast<int>(det.w), static_cast<int>(det.h));
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        
        // Draw label
        std::string label = det.class_name + " " + std::to_string(det.confidence).substr(0, 4);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::Point label_pos(static_cast<int>(det.x), static_cast<int>(det.y) - 5);
        cv::rectangle(image, 
                     cv::Rect(label_pos.x, label_pos.y - text_size.height - baseline, 
                             text_size.width, text_size.height + baseline),
                     cv::Scalar(0, 255, 0), -1);
        cv::putText(image, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

int main(int argc, char* argv[]) {
    std::string config_path = "config/detect.yaml";
    std::string source_override;
    std::string save_vis_dir;
    std::string json_output_file;
    std::string log_level_override;
    std::string undistort_calib_override;
    std::string preprocess_profile_override;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--cfg" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            source_override = argv[++i];
        } else if (arg == "--save_vis" && i + 1 < argc) {
            save_vis_dir = argv[++i];
        } else if (arg == "--json" && i + 1 < argc) {
            json_output_file = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            // 延后到配置加载后再应用，这里仅消费一个数值参数防止被误判为未知参数
            ++i; // consume N
        } else if (arg == "--log-level" && i + 1 < argc) {
            log_level_override = argv[++i];
        } else if (arg == "--undistort-calib" && i + 1 < argc) {
            undistort_calib_override = argv[++i];
        } else if (arg == "--pp-profile" && i + 1 < argc) {
            preprocess_profile_override = argv[++i];
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            LOGE("Unknown argument: ", arg);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Load configuration
    Config config = loadConfig(config_path);
    
    // Override source/log level if provided via CLI
    if (!source_override.empty()) {
        config.source_uri = source_override;
    }
    if (!log_level_override.empty()) {
        config.log_level = log_level_override;
    }
    if (!undistort_calib_override.empty()) {
        config.enable_undistort = true;
        config.calibration_file = undistort_calib_override;
    }
    if (!preprocess_profile_override.empty()) {
        config.preprocess_profile = preprocess_profile_override;
    }

    // Re-scan argv to pick CLI-only options that override config
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) {
            config.warmup = std::stoi(argv[++i]);
        } else if (arg == "--async") {
            config.async = true;
        } else if (arg == "--log-level" && i + 1 < argc) {
            ++i; // already consumed earlier
        } else if (arg == "--undistort-calib" && i + 1 < argc) {
            ++i; // already consumed earlier
        } else if (arg == "--pp-profile" && i + 1 < argc) {
            ++i; // already consumed earlier
        }
    }
    
    rklog::g_level.store(parseLogLevel(config.log_level), std::memory_order_relaxed);

    // Create visualization output directory
    if (!save_vis_dir.empty()) {
        std::filesystem::create_directories(save_vis_dir);
    }
    
    LOGI("=== Object Detection Pipeline ===");
    LOGI("Source: ", config.source_type, " (", config.source_uri, ")");
    LOGI("Engine: ", config.engine_type, " (", config.model_path, ")");
    LOGI("Input size: ", config.imgsz);
    LOGI("Thresholds: conf=", config.conf_thres, ", iou=", config.iou_thres);
    LOGI("Undistort: ", (config.enable_undistort ? "enabled" : "disabled"));
    LOGI("Preprocess profile: ", config.preprocess_profile);

    const FeatureFlags feature_flags = resolveFeatureFlags(config);
    LOGI("ROI: ", (config.roi_enable ? "enabled" : "disabled"));
    LOGI("White balance: ", (feature_flags.white_balance ? "enabled" : "disabled"));
    LOGI("Gamma: ", (feature_flags.gamma ? "enabled" : "disabled"));
    LOGI("Denoise: ", (feature_flags.denoise ? "enabled" : "disabled"));

    rkapp::preprocess::CameraCalibration calibration;
    bool undistort_active = false;
    cv::Mat undistort_map1;
    cv::Mat undistort_map2;
    cv::Size undistort_size{0, 0};
    if (config.enable_undistort) {
        if (config.calibration_file.empty()) {
            LOGW("Undistort enabled but calibration file path is empty");
        } else if (rkapp::preprocess::Preprocess::loadCalibration(
                       config.calibration_file, calibration)) {
            undistort_active = true;
            LOGI("Loaded calibration file: ", config.calibration_file);
        } else {
            LOGW("Failed to load calibration file: ", config.calibration_file);
        }
    }

    struct PreprocessedFrame {
        cv::Mat infer_frame;
        cv::Mat coord_frame;
        cv::Rect roi_rect;
        bool roi_applied = false;
    };

    auto preprocess_frame = [&](const cv::Mat& src) -> PreprocessedFrame {
        PreprocessedFrame out;
        if (src.empty()) {
            return out;
        }

        cv::Mat bgr = rkapp::preprocess::Preprocess::ensureBgr8(
            src, rkapp::preprocess::AccelBackend::OPENCV,
            rkapp::preprocess::FourChannelOrder::UNKNOWN);
        if (bgr.empty()) {
            return out;
        }

        cv::Mat coord_frame = bgr;
        if (undistort_active) {
            if (undistort_size != bgr.size()) {
                if (rkapp::preprocess::Preprocess::buildUndistortMaps(
                        calibration, bgr.size(), undistort_map1, undistort_map2)) {
                    undistort_size = bgr.size();
                } else {
                    LOGW("Failed to build undistort maps for frame size ", bgr.cols, "x", bgr.rows,
                         ", disabling undistort");
                    undistort_active = false;
                    undistort_map1.release();
                    undistort_map2.release();
                    undistort_size = {0, 0};
                }
            }
            if (undistort_active && !undistort_map1.empty() && !undistort_map2.empty()) {
                cv::Mat undistorted =
                    rkapp::preprocess::Preprocess::undistort(bgr, undistort_map1, undistort_map2);
                if (!undistorted.empty()) {
                    coord_frame = std::move(undistorted);
                }
            }
        }

        out.coord_frame = coord_frame;
        out.roi_rect = cv::Rect(0, 0, coord_frame.cols, coord_frame.rows);

        cv::Mat working = coord_frame;
        if (config.roi_enable) {
            cv::Rect resolved_roi;
            const cv::Rect2f normalized_roi(
                config.roi_normalized_xywh[0], config.roi_normalized_xywh[1],
                config.roi_normalized_xywh[2], config.roi_normalized_xywh[3]);
            const cv::Rect pixel_roi(
                config.roi_pixel_xywh[0], config.roi_pixel_xywh[1],
                config.roi_pixel_xywh[2], config.roi_pixel_xywh[3]);
            if (rkapp::preprocess::Preprocess::resolveRoiRect(
                    coord_frame.size(), rkapp::common::toLowerCopy(config.roi_mode) != "pixel", normalized_roi,
                    pixel_roi, config.roi_clamp, config.roi_min_size, resolved_roi)) {
                out.roi_rect = resolved_roi;
                if (resolved_roi.x != 0 || resolved_roi.y != 0 ||
                    resolved_roi.width != coord_frame.cols || resolved_roi.height != coord_frame.rows) {
                    cv::Mat cropped = rkapp::preprocess::Preprocess::cropRoi(coord_frame, resolved_roi);
                    if (!cropped.empty()) {
                        working = std::move(cropped);
                        out.roi_applied = true;
                    } else {
                        LOGW("ROI crop failed, using full frame");
                        out.roi_rect = cv::Rect(0, 0, coord_frame.cols, coord_frame.rows);
                    }
                }
            } else {
                LOGW("Invalid ROI config, using full frame");
            }
        }

        if (feature_flags.denoise) {
            if (rkapp::common::toLowerCopy(config.denoise_method) != "bilateral") {
                LOGW("Unsupported denoise method '", config.denoise_method, "', using bilateral");
            }
            cv::Mat denoised = rkapp::preprocess::Preprocess::denoiseBilateral(
                working, config.denoise_d, config.denoise_sigma_color, config.denoise_sigma_space);
            if (!denoised.empty()) {
                working = std::move(denoised);
            }
        }
        if (feature_flags.white_balance) {
            cv::Mat balanced = rkapp::preprocess::Preprocess::whiteBalanceGrayWorld(
                working, config.white_balance_clip_percent);
            if (!balanced.empty()) {
                working = std::move(balanced);
            }
        }
        if (feature_flags.gamma) {
            cv::Mat gamma_corrected =
                rkapp::preprocess::Preprocess::applyGammaLut(working, config.gamma_value);
            if (!gamma_corrected.empty()) {
                working = std::move(gamma_corrected);
            }
        }

        out.infer_frame = working;
        return out;
    };
    
    // Create source
    std::unique_ptr<rkapp::capture::ISource> source;
    if (config.source_type == "folder") {
        source = std::make_unique<rkapp::capture::FolderSource>();
    } else if (config.source_type == "video" || config.source_type == "rtsp") {
        source = std::make_unique<rkapp::capture::VideoSource>();
    } else if (config.source_type == "gige") {
#ifdef RKAPP_WITH_GIGE
        source = std::make_unique<rkapp::capture::GigeSource>();
#else
        LOGE("GigE source requested but not built. Rebuild with -DENABLE_GIGE=ON and Aravis installed.");
        return 1;
#endif
    } else if (config.source_type == "csi") {
#ifdef RKAPP_WITH_CSI
        source = std::make_unique<rkapp::capture::CsiSource>();
#else
        LOGE("CSI source requested but not built. Rebuild with -DENABLE_CSI=ON and GStreamer installed.");
        return 1;
#endif
    } else {
        LOGE("Unsupported source type: ", config.source_type);
        return 1;
    }
    
    if (!source->open(config.source_uri)) {
        LOGE("Failed to open source: ", config.source_uri);
        return 1;
    }
    
    // Create inference engine
    std::unique_ptr<rkapp::infer::IInferEngine> engine;
    if (config.engine_type == "onnx") {
#ifdef RKAPP_WITH_ONNX
        engine = std::make_unique<rkapp::infer::OnnxEngine>();
#else
        LOGE("ONNX engine requested but not built. Rebuild with -DENABLE_ONNX=ON");
        return 1;
#endif
    } else if (config.engine_type == "rknn") {
#ifdef RKAPP_WITH_RKNN
        engine = std::make_unique<rkapp::infer::RknnEngine>();
#else
        LOGE("RKNN engine requested but not built. Rebuild with -DENABLE_RKNN=ON");
        return 1;
#endif
    } else {
        LOGE("Unsupported engine type: ", config.engine_type);
        return 1;
    }
    
    if (!engine->init(config.model_path, config.imgsz)) {
        LOGE("Failed to initialize inference engine");
        return 1;
    }

    // Propagate decode/NMS thresholds into the engine to avoid mismatched configs
    rkapp::infer::DecodeParams decode_params;
    decode_params.conf_thres = config.conf_thres;
    decode_params.iou_thres = config.iou_thres;
    decode_params.max_boxes = config.nms_topk;
    engine->setDecodeParams(decode_params);

    // Warmup
    LOGI("Warmup x", config.warmup, "...");
    for (int i = 0; i < config.warmup; ++i) {
        cv::Mat dummy(config.imgsz, config.imgsz, CV_8UC3, cv::Scalar(128,128,128));
        (void)engine->infer(dummy);
    }
    
    const std::string normalized_output_type = rkapp::common::toLowerCopy(config.output_type);
    if (normalized_output_type != "tcp") {
        LOGE("Unsupported output.type='", config.output_type,
             "'. Only 'tcp' is implemented in this build.");
        return 1;
    }

    // Create output
    std::unique_ptr<rkapp::output::IOutput> output;
    if (!json_output_file.empty() || normalized_output_type == "tcp") {
        output = std::make_unique<rkapp::output::TcpOutput>();
        
        std::string output_config = config.output_ip + ":" + std::to_string(config.output_port);
        if (config.output_queue > 0) {
            output_config += ",queue:" + std::to_string(config.output_queue);
        }
        if (!json_output_file.empty()) {
            output_config += ",file:" + json_output_file;
        }
        
        if (!output->open(output_config)) {
            LOGW("Failed to open output, continuing without output");
            output.reset();
        }
    }
    
    // Load class names
    std::vector<std::string> class_names;
    if (!config.inline_class_names.empty()) {
        class_names = config.inline_class_names;
    } else if (!config.classes_path.empty()) {
        class_names = rkapp::post::Postprocess::loadClassNames(config.classes_path);
    }

    LOGI("=== Starting Detection Loop ===");
    LOGI("Pipeline created successfully (STUB MODE)");
    LOGI("Ready to process frames from: ", config.source_uri);
    
    // Main processing loop
    cv::Mat frame;
    int frame_id = 0;

    if (!config.async) {
      const bool retry_on_fail = shouldRetryRead(config.source_type, source->getType());
      int read_failures = 0;
      for (;;) {
        if (!source->read(frame)) {
          if (!retry_on_fail) {
            break;
          }
          ++read_failures;
          if (read_failures % 10 == 1) {
            LOGW("Frame read failed (stream). Retrying...");
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          continue;
        }
        read_failures = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        PreprocessedFrame preprocessed = preprocess_frame(frame);
        if (preprocessed.infer_frame.empty() || preprocessed.coord_frame.empty()) {
          LOGW("Frame preprocessing failed, skipping frame ", frame_id);
          continue;
        }

        // Inference (engine handles its own letterbox consistently)
        std::vector<rkapp::infer::Detection> detections = engine->infer(preprocessed.infer_frame);
        if (preprocessed.roi_applied) {
            applyRoiOffsetAndClip(detections, preprocessed.roi_rect, preprocessed.coord_frame.size());
        }
        
        // Postprocess (class-name mapping only; NMS is handled inside the engine)
        rkapp::post::Postprocess::mapClassNames(detections, class_names);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        LOGI("Frame ", frame_id, ": ", detections.size(),
             " detections (", duration.count(), "ms)");
        
        // Save visualization
        if (!save_vis_dir.empty()) {
            cv::Mat vis_frame = preprocessed.coord_frame.clone();
            drawDetections(vis_frame, detections);
            std::string vis_path = save_vis_dir + "/frame_" + std::to_string(frame_id) + ".jpg";
            cv::imwrite(vis_path, vis_frame);
        }
        
        // Send output
        if (output) {
            rkapp::output::FrameResult result;
            result.frame_id = frame_id;
            result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            result.width = preprocessed.coord_frame.cols;
            result.height = preprocessed.coord_frame.rows;
            result.detections = detections;
            result.source_uri = config.source_uri;
            output->send(result);
        }
        
        frame_id++;
      }
    } else {
      // Async mode: simple 2-stage pipeline with bounded queue + pre-allocated Mat pool
      // Uses shared_ptr for thread-safe state sharing to avoid dangling reference issues
      struct Item { int id; cv::Mat img; };
      // QMAX = 3: balance between latency (smaller queue = less lag) and throughput
      // (larger queue = handles burst). 3 frames = ~100ms buffer at 30fps.
      constexpr size_t QMAX = 3;

      // Shared state wrapped in struct to ensure proper lifetime management
      // when captured by the worker thread lambda
      struct AsyncState {
        std::mutex mtx;
        std::condition_variable cv_not_full;
        std::condition_variable cv_not_empty;
        std::deque<Item> q;
        std::atomic<bool> done{false};
        std::atomic<int> next_id{0};
        // Pre-allocated Mat pool to avoid per-frame malloc
        std::mutex pool_mtx;
        std::vector<cv::Mat> mat_pool;
      };
      auto state = std::make_shared<AsyncState>();
      // Pre-allocate pool (will be sized on first frame)
      state->mat_pool.reserve(QMAX + 1);

      // Capture source as raw pointer (guaranteed valid for thread lifetime due to join below)
      // Capture state as shared_ptr to ensure thread-safe access
      rkapp::capture::ISource* source_ptr = source.get();
      const bool retry_on_fail = shouldRetryRead(config.source_type, source_ptr->getType());
      std::thread t_capture([state, source_ptr, QMAX, retry_on_fail]{
        cv::Mat f;
        int read_failures = 0;
        for (;;) {
          if (!source_ptr->read(f)) {
            if (!retry_on_fail) {
              break;
            }
            ++read_failures;
            if (read_failures % 10 == 1) {
              LOGW("Frame read failed (stream). Retrying...");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
          }
          read_failures = 0;
          // Get a Mat from pool or create new one (first few frames)
          cv::Mat pooled;
          {
            std::lock_guard<std::mutex> plk(state->pool_mtx);
            if (!state->mat_pool.empty()) {
              pooled = std::move(state->mat_pool.back());
              state->mat_pool.pop_back();
            }
          }
          // Reuse pooled Mat if same size, otherwise allocate
          if (pooled.rows == f.rows && pooled.cols == f.cols && pooled.type() == f.type()) {
            f.copyTo(pooled);  // Copy into pre-allocated memory
          } else {
            pooled = f.clone();  // First frame or size changed
          }

          std::unique_lock<std::mutex> lk(state->mtx);
          state->cv_not_full.wait(lk, [&state, QMAX]{ return state->q.size() < QMAX; });
          int id = state->next_id.fetch_add(1, std::memory_order_relaxed);
          state->q.push_back(Item{id, std::move(pooled)});
          lk.unlock();
          state->cv_not_empty.notify_one();
        }
        {
          std::lock_guard<std::mutex> lk(state->mtx);
          state->done.store(true, std::memory_order_release);
        }
        state->cv_not_empty.notify_all();
      });

      while (true) {
        Item it; bool has=false;
        {
          std::unique_lock<std::mutex> lk(state->mtx);
          state->cv_not_empty.wait(lk, [&state]{ return !state->q.empty() || state->done.load(std::memory_order_acquire); });
          if (!state->q.empty()) {
            it = std::move(state->q.front());
            state->q.pop_front();
            has = true;
            state->cv_not_full.notify_one();
          } else if (state->done.load(std::memory_order_acquire)) {
            break;
          }
        }
        if (!has) break;

        PreprocessedFrame preprocessed = preprocess_frame(it.img);
        if (preprocessed.infer_frame.empty() || preprocessed.coord_frame.empty()) {
          LOGW("Frame preprocessing failed, skipping frame ", it.id);
          {
            std::lock_guard<std::mutex> plk(state->pool_mtx);
            if (state->mat_pool.size() < QMAX + 1) {
              state->mat_pool.push_back(std::move(it.img));
            }
          }
          continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<rkapp::infer::Detection> detections = engine->infer(preprocessed.infer_frame);
        if (preprocessed.roi_applied) {
            applyRoiOffsetAndClip(detections, preprocessed.roi_rect, preprocessed.coord_frame.size());
        }
        rkapp::post::Postprocess::mapClassNames(detections, class_names);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        LOGI("Frame ", it.id, ": ", detections.size(),
             " detections (", duration.count(), "ms)");

        if (!save_vis_dir.empty()) {
            cv::Mat vis_frame = preprocessed.coord_frame.clone();
            drawDetections(vis_frame, detections);
            std::string vis_path = save_vis_dir + "/frame_" + std::to_string(it.id) + ".jpg";
            cv::imwrite(vis_path, vis_frame);
        }

        if (output) {
            rkapp::output::FrameResult result;
            result.frame_id = it.id;
            result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            result.width = preprocessed.coord_frame.cols;
            result.height = preprocessed.coord_frame.rows;
            result.detections = detections;
            result.source_uri = config.source_uri;
            output->send(result);
        }

        // Return Mat to pool for reuse (avoids per-frame malloc)
        {
          std::lock_guard<std::mutex> plk(state->pool_mtx);
          if (state->mat_pool.size() < QMAX + 1) {
            state->mat_pool.push_back(std::move(it.img));
          }
          // else: pool full, let Mat deallocate naturally
        }
      }
      t_capture.join();
    }
    
    return 0;
}
