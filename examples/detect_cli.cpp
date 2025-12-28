#include <iostream>
#include <chrono>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <vector>

#include "rkapp/capture/ISource.hpp"
#include "rkapp/infer/IInferEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "rkapp/output/IOutput.hpp"
#include "log.hpp"

// Concrete class headers
#include "rkapp/capture/FolderSource.hpp"
#include "rkapp/capture/VideoSource.hpp"
#ifdef RKAPP_WITH_GIGE
#include "rkapp/capture/GigeSource.hpp"
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

    // Re-scan argv to pick CLI-only options that override config
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) {
            config.warmup = std::stoi(argv[++i]);
        } else if (arg == "--async") {
            config.async = true;
        } else if (arg == "--log-level" && i + 1 < argc) {
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
    
    // Create output
    std::unique_ptr<rkapp::output::IOutput> output;
    if (!json_output_file.empty() || config.output_type == "tcp") {
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
      while (source->read(frame)) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Inference (engine handles its own letterbox consistently)
        std::vector<rkapp::infer::Detection> detections = engine->infer(frame);
        
        // Postprocess (class-name mapping only; NMS is handled inside the engine)
        rkapp::post::Postprocess::mapClassNames(detections, class_names);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        LOGI("Frame ", frame_id, ": ", detections.size(),
             " detections (", duration.count(), "ms)");
        
        // Save visualization
        if (!save_vis_dir.empty()) {
            cv::Mat vis_frame = frame.clone();
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
            result.width = frame.cols;
            result.height = frame.rows;
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
      std::thread t_capture([state, source_ptr, QMAX]{
        cv::Mat f;
        while (source_ptr->read(f)) {
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

        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<rkapp::infer::Detection> detections = engine->infer(it.img);
        rkapp::post::Postprocess::mapClassNames(detections, class_names);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        LOGI("Frame ", it.id, ": ", detections.size(),
             " detections (", duration.count(), "ms)");

        if (!save_vis_dir.empty()) {
            cv::Mat vis_frame = it.img.clone();
            drawDetections(vis_frame, detections);
            std::string vis_path = save_vis_dir + "/frame_" + std::to_string(it.id) + ".jpg";
            cv::imwrite(vis_path, vis_frame);
        }

        if (output) {
            rkapp::output::FrameResult result;
            result.frame_id = it.id;
            result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            result.width = it.img.cols;
            result.height = it.img.rows;
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
