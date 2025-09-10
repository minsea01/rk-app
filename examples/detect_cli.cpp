#include <iostream>
#include <chrono>
#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "rkapp/capture/ISource.hpp"
#include "rkapp/infer/IInferEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "rkapp/output/IOutput.hpp"

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
    std::string output_type = "tcp";
    std::string output_ip = "127.0.0.1";
    int output_port = 9000;
    std::string classes_path = "config/classes.txt";
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
    std::cout << "  --help               Show this help message\\n";
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
            }
            
            if (yaml["nms"]) {
                config.conf_thres = yaml["nms"]["conf_thres"].as<float>(0.25f);
                config.iou_thres = yaml["nms"]["iou_thres"].as<float>(0.60f);
                if (yaml["nms"]["topk"]) config.nms_topk = yaml["nms"]["topk"].as<int>(1000);
            }

            if (yaml["perf"]) {
                config.warmup = yaml["perf"]["warmup"].as<int>(5);
                config.async = yaml["perf"]["async"].as<bool>(false);
            }
            
            if (yaml["output"]) {
                config.output_type = yaml["output"]["type"].as<std::string>("tcp");
                config.output_ip = yaml["output"]["ip"].as<std::string>("127.0.0.1");
                config.output_port = yaml["output"]["port"].as<int>(9000);
            }
            
            config.classes_path = yaml["classes"].as<std::string>("config/classes.txt");
            
            std::cout << "Loaded configuration from " << config_path << std::endl;
        } else {
            std::cout << "Configuration file not found: " << config_path << ", using defaults" << std::endl;
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
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
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Load configuration
    Config config = loadConfig(config_path);
    
    // Override source if provided
    if (!source_override.empty()) {
        config.source_uri = source_override;
    }

    // Re-scan argv to pick CLI-only options that override config
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) {
            config.warmup = std::stoi(argv[++i]);
        } else if (arg == "--async") {
            config.async = true;
        }
    }
    
    // Create visualization output directory
    if (!save_vis_dir.empty()) {
        std::filesystem::create_directories(save_vis_dir);
    }
    
    std::cout << "=== Object Detection Pipeline ===" << std::endl;
    std::cout << "Source: " << config.source_type << " (" << config.source_uri << ")" << std::endl;
    std::cout << "Engine: " << config.engine_type << " (" << config.model_path << ")" << std::endl;
    std::cout << "Input size: " << config.imgsz << std::endl;
    std::cout << "Thresholds: conf=" << config.conf_thres << ", iou=" << config.iou_thres << std::endl;
    
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
        std::cerr << "GigE source requested but not built. Rebuild with -DENABLE_GIGE=ON and Aravis installed." << std::endl;
        return 1;
#endif
    } else {
        std::cerr << "Unsupported source type: " << config.source_type << std::endl;
        return 1;
    }
    
    if (!source->open(config.source_uri)) {
        std::cerr << "Failed to open source: " << config.source_uri << std::endl;
        return 1;
    }
    
    // Create inference engine
    std::unique_ptr<rkapp::infer::IInferEngine> engine;
    if (config.engine_type == "onnx") {
#ifdef RKAPP_WITH_ONNX
        engine = std::make_unique<rkapp::infer::OnnxEngine>();
#else
        std::cerr << "ONNX engine requested but not built. Rebuild with -DENABLE_ONNX=ON" << std::endl;
        return 1;
#endif
    } else if (config.engine_type == "rknn") {
#ifdef RKAPP_WITH_RKNN
        engine = std::make_unique<rkapp::infer::RknnEngine>();
#else
        std::cerr << "RKNN engine requested but not built. Rebuild with -DENABLE_RKNN=ON" << std::endl;
        return 1;
#endif
    } else {
        std::cerr << "Unsupported engine type: " << config.engine_type << std::endl;
        return 1;
    }
    
    if (!engine->init(config.model_path, config.imgsz)) {
        std::cerr << "Failed to initialize inference engine" << std::endl;
        return 1;
    }
    
    // Warmup
    std::cout << "Warmup x" << config.warmup << "..." << std::endl;
    for (int i = 0; i < config.warmup; ++i) {
        cv::Mat dummy(config.imgsz, config.imgsz, CV_8UC3, cv::Scalar(128,128,128));
        (void)engine->infer(dummy);
    }
    
    // Create output
    std::unique_ptr<rkapp::output::IOutput> output;
    if (!json_output_file.empty() || config.output_type == "tcp") {
        output = std::make_unique<rkapp::output::TcpOutput>();
        
        std::string output_config = config.output_ip + ":" + std::to_string(config.output_port);
        if (!json_output_file.empty()) {
            output_config += ",file:" + json_output_file;
        }
        
        if (!output->open(output_config)) {
            std::cout << "Warning: Failed to open output, continuing without output" << std::endl;
            output.reset();
        }
    }
    
    // Load class names
    std::vector<std::string> class_names = rkapp::post::Postprocess::loadClassNames(config.classes_path);
    
    std::cout << "\\n=== Starting Detection Loop ===" << std::endl;
    std::cout << "Pipeline created successfully (STUB MODE)" << std::endl;
    std::cout << "Ready to process frames from: " << config.source_uri << std::endl;
    
    // Main processing loop
    cv::Mat frame;
    int frame_id = 0;

    if (!config.async) {
      while (source->read(frame)) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Inference (engine handles its own letterbox consistently)
        std::vector<rkapp::infer::Detection> detections = engine->infer(frame);
        
        // Postprocess
        rkapp::post::NMSConfig nms_config{config.conf_thres, config.iou_thres, 1000, config.nms_topk};
        detections = rkapp::post::Postprocess::nms(detections, nms_config);
        rkapp::post::Postprocess::mapClassNames(detections, class_names);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Frame " << frame_id << ": " << detections.size() 
                  << " detections (" << duration.count() << "ms)" << std::endl;
        
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
      // Async mode: simple 2-stage pipeline with bounded queue
      struct Item { int id; cv::Mat img; };
      const size_t QMAX = 3;
      std::mutex mtx; std::condition_variable cv_not_full, cv_not_empty; std::deque<Item> q;
      bool done = false; std::atomic<int> next_id{0};

      std::thread t_capture([&]{
        cv::Mat f;
        while (source->read(f)) {
          std::unique_lock<std::mutex> lk(mtx);
          cv_not_full.wait(lk, [&]{ return q.size() < QMAX; });
          q.push_back(Item{next_id++, f.clone()});
          lk.unlock(); cv_not_empty.notify_one();
        }
        std::lock_guard<std::mutex> lk(mtx); done = true; cv_not_empty.notify_all();
      });

      while (true) {
        Item it; bool has=false;
        {
          std::unique_lock<std::mutex> lk(mtx);
          cv_not_empty.wait(lk, [&]{ return !q.empty() || done; });
          if (!q.empty()) { it = std::move(q.front()); q.pop_front(); has=true; cv_not_full.notify_one(); }
          else if (done) break;
        }
        if (!has) break;

        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<rkapp::infer::Detection> detections = engine->infer(it.img);
        rkapp::post::NMSConfig nms_config{config.conf_thres, config.iou_thres, 1000, config.nms_topk};
        detections = rkapp::post::Postprocess::nms(detections, nms_config);
        rkapp::post::Postprocess::mapClassNames(detections, class_names);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Frame " << it.id << ": " << detections.size() << " detections (" << duration.count() << "ms)" << std::endl;

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
      }
      t_capture.join();
    }
    
    return 0;
}
