/**
 * C++端到端延迟基准测试
 *
 * 对比Python版本的性能
 *
 * 编译（板端）：
 *   见 scripts/build_cpp_board.sh
 *
 * 运行：
 *   ./build/board/bench_e2e_cpp --model artifacts/models/yolo11n_416.rknn --image assets/test.jpg
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

#ifdef RKAPP_WITH_RKNN
#include "rkapp/infer/RknnEngine.hpp"
#endif

struct BenchmarkResult {
    std::vector<double> capture_times;
    std::vector<double> preprocess_times;
    std::vector<double> inference_times;
    std::vector<double> postprocess_times;
    std::vector<double> total_times;
    std::vector<int> detection_counts;
};

double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double mean(const std::vector<int>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double stddev(const std::vector<double>& v) {
    double m = mean(v);
    double sq_sum = 0.0;
    for (auto val : v) {
        sq_sum += (val - m) * (val - m);
    }
    return std::sqrt(sq_sum / v.size());
}

int main(int argc, char** argv) {
    std::string model_path = "artifacts/models/yolo11n_416.rknn";
    std::string image_path = "assets/test.jpg";
    int iterations = 50;
    float conf_thres = 0.5f;

    // 解析参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            conf_thres = std::stof(argv[++i]);
        }
    }

    std::cout << "========================================\n";
    std::cout << "C++端到端延迟基准测试\n";
    std::cout << "========================================\n";
    std::cout << "模型: " << model_path << "\n";
    std::cout << "图片: " << image_path << "\n";
    std::cout << "迭代: " << iterations << "\n";
    std::cout << "置信度: " << conf_thres << "\n";
    std::cout << "\n";

#ifdef RKAPP_WITH_RKNN
    // 初始化引擎
    std::cout << "加载RKNN模型...\n";
    auto engine = std::make_unique<rkapp::infer::RknnEngine>();
    if (!engine->init(model_path, 416)) {
        std::cerr << "❌ 模型加载失败\n";
        return 1;
    }
    std::cout << "✓ 模型加载成功\n\n";

    // 设置解码参数
    rkapp::infer::DecodeParams params;
    params.conf_thres = conf_thres;
    params.iou_thres = 0.45f;
    engine->setDecodeParams(params);

    // 加载图片
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "❌ 图片加载失败: " << image_path << "\n";
        return 1;
    }
    std::cout << "✓ 图片加载成功: " << image.cols << "×" << image.rows << "\n\n";

    // 预热
    std::cout << "预热 (3次)...\n";
    for (int i = 0; i < 3; ++i) {
        engine->infer(image);
    }
    std::cout << "✓ 预热完成\n\n";

    // 基准测试
    std::cout << "开始基准测试...\n\n";
    BenchmarkResult result;

    for (int i = 0; i < iterations; ++i) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. Capture (模拟)
        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat frame = image.clone();
        auto t1 = std::chrono::high_resolution_clock::now();
        result.capture_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );

        // 2-4. 推理 (包含预处理、推理、后处理)
        t0 = std::chrono::high_resolution_clock::now();
        auto detections = engine->infer(frame);
        t1 = std::chrono::high_resolution_clock::now();
        double infer_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.inference_times.push_back(infer_total);

        auto t_end = std::chrono::high_resolution_clock::now();
        double total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        result.total_times.push_back(total);
        result.detection_counts.push_back(detections.size());

        if ((i + 1) % 10 == 0) {
            std::cout << "迭代 " << (i + 1) << "/" << iterations
                      << ": 总延迟 " << total << "ms"
                      << ", 检测 " << detections.size() << "个目标\n";
        }
    }

    // 统计结果
    std::cout << "\n========================================\n";
    std::cout << "基准测试结果\n";
    std::cout << "========================================\n\n";

    double capture_mean = mean(result.capture_times);
    double infer_mean = mean(result.inference_times);
    double total_mean = mean(result.total_times);
    double detect_mean = mean(result.detection_counts);

    std::cout << "Capture:    " << capture_mean << " ms\n";
    std::cout << "推理(全部): " << infer_mean << " ms\n";
    std::cout << "总延迟:     " << total_mean << " ms\n";
    std::cout << "平均检测:   " << detect_mean << " 个目标\n";
    std::cout << "\n";

    std::cout << "端到端FPS:  " << (1000.0 / total_mean) << "\n";
    std::cout << "\n";

    std::cout << "========================================\n";
    std::cout << "任务书指标检查\n";
    std::cout << "========================================\n";
    std::cout << "要求: 1080P处理延迟 ≤ 45ms\n";
    std::cout << "实测: " << total_mean << " ms\n";
    if (total_mean <= 45.0) {
        std::cout << "状态: ✅ 合规\n";
    } else {
        std::cout << "状态: ⚠️  超出 " << (total_mean - 45.0) << "ms\n";
    }
    std::cout << "========================================\n\n";

    return 0;
#else
    std::cerr << "❌ 项目未启用RKNN支持\n";
    std::cerr << "   请使用 -DENABLE_RKNN=ON 重新编译\n";
    return 1;
#endif
}
