#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>

#include "rkapp/capture/ISource.hpp"
#include "rkapp/infer/IInferEngine.hpp"
#include "rkapp/preprocess/Preprocess.hpp"
#include "rkapp/common/DmaBuf.hpp"

namespace rkapp::pipeline {

/**
 * @brief 检测流水线配置
 */
struct PipelineConfig {
    using ModelBackend = infer::ModelBackend;

    struct SourceSpec {
        std::string uri;                // 视频/图片路径、RTSP URL 或 CSI URI
        capture::SourceType type = capture::SourceType::VIDEO;
        bool use_mpp_decode = true;     // 启用 MPP 硬解码
    };
    using ModelSpec = infer::ModelSpec;

    struct PreprocessSpec {
        bool use_rga_preprocess = true;  // 启用 RGA 预处理
        bool enable_undistort = false;   // 启用去畸变（cv::remap）
        std::string calibration_file;    // OpenCV 标定文件路径（YAML/XML）
        std::string profile = "speed";   // speed|balanced|quality
        bool roi_enable = false;
        std::string roi_mode = "normalized";  // normalized|pixel
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

    struct OutputSpec {
        std::string type = "tcp";
        std::string host = "127.0.0.1";
        int port = 9000;
        int queue_size = 0;
        bool enable_profiling = false;
    };

    struct RuntimeSpec {
        int warmup_iterations = 5;
        bool async_mode = false;
    };

    struct LoggingSpec {
        std::string level = "INFO";
    };

    struct FailurePolicy {
        int frame_error_limit = 8;
        int source_recovery_grace_ms = 15000;
        int max_reconnect_attempts = -1;
        int initial_backoff_ms = 500;
        int max_backoff_ms = 5000;
    };

    SourceSpec source;
    ModelSpec model;
    PreprocessSpec preprocess;
    OutputSpec output;
    RuntimeSpec runtime;
    LoggingSpec logging;
    FailurePolicy failure;
};

struct PipelineConfigLoadResult {
    PipelineConfig config;
    std::vector<std::string> warnings;
};

PipelineConfig normalizePipelineConfig(
    const PipelineConfig& raw_config,
    std::vector<std::string>* warnings = nullptr);
PipelineConfigLoadResult loadPipelineConfigFile(const std::string& config_path);

/**
 * @brief 单帧检测结果（含可选耗时）
 */
struct PipelineResult {
    std::vector<infer::Detection> detections;
    int64_t frame_id = -1;
    cv::Mat frame;  // 可选：原始图像（按调用方需要保留）

    // 分阶段耗时（微秒，仅在 enable_profiling=true 时有意义）
    struct Timing {
        int64_t capture_us = 0;
        int64_t preprocess_us = 0;
        int64_t inference_us = 0;
        int64_t postprocess_us = 0;
        int64_t total_us = 0;
    } timing;
};

/**
 * @brief 异步模式回调签名
 */
using ResultCallback = std::function<void(PipelineResult&&)>;

/**
 * @brief RK3588 高性能检测流水线
 *
 * 将采集、预处理、推理、后处理串成统一接口，并可启用硬件加速：
 * - NPU 多核推理
 * - RGA 预处理加速
 * - MPP 视频硬解码
 * - DMA-BUF 零拷贝
 *
 * 使用示例：
 * @code
 *   PipelineConfig cfg;
 *   cfg.source.uri = "video.mp4";
 *   cfg.model.model_path = "yolo11n.rknn";
 *
 *   DetectionPipeline pipeline;
 *   pipeline.init(cfg);
 *
 *   // Synchronous mode
 *   while (auto result = pipeline.next()) {
 *       for (const auto& det : result->detections) {
 *           // Process detection
 *       }
 *   }
 *
 *   // Or async mode with callback
 *   pipeline.runAsync([](PipelineResult&& result) {
 *       // Handle result in callback
 *   });
 * @endcode
 */
class DetectionPipeline {
public:
    DetectionPipeline();
    ~DetectionPipeline();

    // Non-copyable
    DetectionPipeline(const DetectionPipeline&) = delete;
    DetectionPipeline& operator=(const DetectionPipeline&) = delete;

    /**
     * @brief 初始化流水线
     *
     * @param config 流水线配置
     * @return 初始化是否成功
     */
    bool init(const PipelineConfig& config);

    /**
     * @brief 同步处理下一帧
     *
     * @return 结果；无更多数据时返回 nullopt
     */
    std::optional<PipelineResult> next();

    /**
     * @brief 处理携带像素格式/stride/存储元数据的输入帧
     *
     * @param frame 输入帧（可为 BGR/RGB/NV12/Bayer 等原始格式）
     * @return 单帧结果
     */
    PipelineResult process(const capture::CaptureFrame& frame);

    /**
     * @brief 处理单张图像（旁路调用）
     *
     * @param image 输入 BGR 图像
     * @return 单帧结果
     */
    PipelineResult process(const cv::Mat& image);

    /**
     * @brief 异步运行流水线并回调结果
     *
     * 后台线程持续拉取帧并执行处理，每帧结果通过 callback 返回。
     */
    void runAsync(ResultCallback callback);

    /**
     * @brief 停止异步处理并回收资源
     */
    void stop();

    /**
     * @brief 查询流水线是否处于运行状态
     */
    bool isRunning() const;

    /**
     * @brief 获取当前统计 FPS
     */
    double getFps() const;

    /**
     * @brief 获取流水线统计信息
     */
    struct Statistics {
        int64_t frames_processed = 0;
        int64_t frames_dropped = 0;
        int64_t reconnect_count = 0;
        int64_t total_detections = 0;
        double avg_fps = 0.0;
        double avg_latency_ms = 0.0;

        // 硬件启用状态/利用率（可用时）
        double npu_utilization = 0.0;  // 0-100%
        bool rga_enabled = false;
        bool mpp_enabled = false;
        bool zero_copy_enabled = false;
    };

    Statistics getStatistics() const;

    /**
     * @brief 重置统计计数
     */
    void resetStatistics();

private:
    std::optional<PipelineResult> nextInternal(bool respect_running_flag);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 根据配置创建输入源
 */
capture::SourcePtr createSource(const PipelineConfig& config);
capture::SourcePtr createSource(const PipelineConfig::SourceSpec& source);

/**
 * @brief 根据配置创建推理引擎
 */
std::unique_ptr<infer::IInferEngine> createEngine(const PipelineConfig& config);
std::unique_ptr<infer::IInferEngine> createEngine(const PipelineConfig::ModelSpec& model);

} // namespace rkapp::pipeline
