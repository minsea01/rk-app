#pragma once

#include <atomic>
#include <functional>
#include <future>
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
 * @brief Pipeline configuration
 */
struct PipelineConfig {
    // Input source configuration
    std::string source_uri;          // Video/image path, RTSP URL, or CSI URI string
    capture::SourceType source_type = capture::SourceType::VIDEO;

    // Model configuration
    std::string model_path;          // RKNN model path
    int input_size = 640;            // Model input size

    // Inference configuration
    float conf_threshold = 0.5f;     // Confidence threshold
    float iou_threshold = 0.45f;     // NMS IOU threshold
    int max_detections = 100;        // Maximum detections per frame

    // Hardware acceleration options
    bool use_npu_multicore = true;   // Enable NPU multi-core (6 TOPS)
    bool use_rga_preprocess = true;  // Enable RGA hardware preprocessing
    bool use_mpp_decode = true;      // Enable MPP hardware video decode
    bool use_zero_copy = true;       // Enable DMA-BUF zero-copy

    // Camera preprocessing options
    bool enable_undistort = false;   // Enable lens undistortion (cv::remap)
    std::string calibration_file;    // OpenCV calibration YAML/XML file path
    std::string preprocess_profile = "speed";  // speed|balanced|quality
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

    // Performance tuning
    int buffer_pool_size = 4;        // Number of pre-allocated DMA buffers
    bool enable_profiling = false;   // Enable timing measurements
};

/**
 * @brief Detection result with timing info
 */
struct PipelineResult {
    std::vector<infer::Detection> detections;
    int64_t frame_id = -1;
    cv::Mat frame;  // Optional: original frame (if requested)

    // Timing breakdown (in microseconds, only when profiling enabled)
    struct Timing {
        int64_t capture_us = 0;
        int64_t preprocess_us = 0;
        int64_t inference_us = 0;
        int64_t postprocess_us = 0;
        int64_t total_us = 0;
    } timing;
};

/**
 * @brief Callback for async detection results
 */
using ResultCallback = std::function<void(PipelineResult&&)>;

/**
 * @brief High-performance detection pipeline for RK3588
 *
 * Integrates all hardware acceleration features:
 * - NPU multi-core inference (6 TOPS)
 * - RGA hardware preprocessing (~0.3ms vs ~3ms OpenCV)
 * - MPP hardware video decoding (~50% CPU reduction)
 * - DMA-BUF zero-copy (~3-5x memory bandwidth reduction)
 *
 * Usage:
 * @code
 *   PipelineConfig cfg;
 *   cfg.source_uri = "video.mp4";
 *   cfg.model_path = "yolo11n.rknn";
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
     * @brief Initialize the pipeline
     *
     * @param config Pipeline configuration
     * @return true if initialization succeeded
     */
    bool init(const PipelineConfig& config);

    /**
     * @brief Process next frame synchronously
     *
     * @return Detection result, or nullopt if no more frames
     */
    std::optional<PipelineResult> next();

    /**
     * @brief Process single frame (for image input)
     *
     * @param image Input image (BGR format)
     * @return Detection result
     */
    PipelineResult process(const cv::Mat& image);

    /**
     * @brief Run pipeline asynchronously with callback
     *
     * Processes frames in a background thread and calls
     * the callback for each result.
     *
     * @param callback Function called with each result
     */
    void runAsync(ResultCallback callback);

    /**
     * @brief Stop async processing
     */
    void stop();

    /**
     * @brief Check if pipeline is running
     */
    bool isRunning() const;

    /**
     * @brief Get current FPS
     */
    double getFps() const;

    /**
     * @brief Get pipeline statistics
     */
    struct Statistics {
        int64_t frames_processed = 0;
        int64_t total_detections = 0;
        double avg_fps = 0.0;
        double avg_latency_ms = 0.0;

        // Hardware utilization (when available)
        double npu_utilization = 0.0;  // 0-100%
        bool rga_enabled = false;
        bool mpp_enabled = false;
        bool zero_copy_enabled = false;
    };

    Statistics getStatistics() const;

    /**
     * @brief Reset statistics
     */
    void resetStatistics();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Factory for creating sources based on config
 */
capture::SourcePtr createSource(const PipelineConfig& config);

/**
 * @brief Factory for creating inference engine based on config
 */
std::unique_ptr<infer::IInferEngine> createEngine(const PipelineConfig& config);

} // namespace rkapp::pipeline
