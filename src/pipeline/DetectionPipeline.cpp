#include "rkapp/pipeline/DetectionPipeline.hpp"
#include "rkapp/capture/VideoSource.hpp"
#include "rkapp/capture/FolderSource.hpp"
#include "rkapp/capture/MppSource.hpp"
#include "rkapp/capture/GigeSource.hpp"
#include "rkapp/capture/CsiSource.hpp"
#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/infer/OnnxEngine.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "rkapp/common/StringUtils.hpp"
#include "rkapp/common/log.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

namespace rkapp::pipeline {

// Use steady_clock for stable latency/jitter measurements (not affected by wall-clock changes).
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

namespace {

int64_t microsecondsSince(const TimePoint& start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start).count();
}

bool isMppDecodeEnabledInStats(const PipelineConfig& config) {
#if RKAPP_WITH_MPP
    if (config.source_type == capture::SourceType::MPP) {
        return true;
    }
    if ((config.source_type == capture::SourceType::VIDEO ||
         config.source_type == capture::SourceType::RTSP) &&
        config.use_mpp_decode) {
        return true;
    }
#else
    (void)config;
#endif
    return false;
}

struct PreprocessFeatureFlags {
    bool gamma = false;
    bool white_balance = false;
    bool denoise = false;
};

PreprocessFeatureFlags resolveFeatureFlags(const PipelineConfig& config) {
    PreprocessFeatureFlags flags;
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

bool roiModeIsNormalized(const std::string& mode) {
    return rkapp::common::toLowerCopy(mode) != "pixel";
}

void applyRoiOffsetAndClip(std::vector<infer::Detection>& detections,
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

} // namespace

// ============================================================================
// Pipeline Implementation
// ============================================================================

struct DetectionPipeline::Impl {
    PipelineConfig config;

    // Pipeline components
    capture::SourcePtr source;
    std::unique_ptr<infer::IInferEngine> engine;
    infer::RknnEngine* rknn_engine = nullptr;
    std::unique_ptr<common::DmaBufPool> buffer_pool;

    // State
    std::atomic<bool> running{false};
    std::atomic<bool> initialized{false};
    int64_t frame_counter{0};

    // Statistics
    mutable std::mutex stats_mutex;
    Statistics stats;
    TimePoint start_time;
    int64_t total_latency_us{0};

    // Async processing
    std::thread worker_thread;
    ResultCallback result_callback;

    // FPS calculation
    std::atomic<double> current_fps{0.0};
    TimePoint last_fps_update;
    int frames_since_update{0};

    // Optional undistort state
    preprocess::CameraCalibration calibration;
    bool calibration_loaded{false};
    cv::Mat undistort_map1;
    cv::Mat undistort_map2;
    cv::Size undistort_size{0, 0};
    PreprocessFeatureFlags preprocess_flags;

    void updateFps() {
        frames_since_update++;
        auto now = Clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_fps_update).count();

        if (elapsed_ms >= 1000) {
            current_fps = frames_since_update * 1000.0 / elapsed_ms;
            frames_since_update = 0;
            last_fps_update = now;
        }
    }

    void updateStats(const PipelineResult& result) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.frames_processed++;
        stats.total_detections += result.detections.size();
        total_latency_us += result.timing.total_us;
        stats.avg_latency_ms = (total_latency_us / stats.frames_processed) / 1000.0;

        auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
            Clock::now() - start_time).count();
        if (elapsed_s > 0) {
            stats.avg_fps = static_cast<double>(stats.frames_processed) / elapsed_s;
        }
    }
};

// ============================================================================
// Public API
// ============================================================================

DetectionPipeline::DetectionPipeline()
    : impl_(std::make_unique<Impl>()) {}

DetectionPipeline::~DetectionPipeline() {
    stop();
}

bool DetectionPipeline::init(const PipelineConfig& config) {
    // Release any previously acquired resources before reinitializing
    if (impl_->engine) {
        impl_->engine->release();
        impl_->engine.reset();
    }
    if (impl_->source) {
        impl_->source->release();
        impl_->source.reset();
    }

    impl_->config = config;
    impl_->rknn_engine = nullptr;
    impl_->calibration = {};
    impl_->calibration_loaded = false;
    impl_->undistort_map1.release();
    impl_->undistort_map2.release();
    impl_->undistort_size = {0, 0};
    impl_->preprocess_flags = resolveFeatureFlags(config);

    if (config.enable_undistort) {
        if (config.calibration_file.empty()) {
            LOGW("DetectionPipeline: Undistort requested but calibration_file is empty");
        } else if (preprocess::Preprocess::loadCalibration(
                       config.calibration_file, impl_->calibration)) {
            impl_->calibration_loaded = true;
            LOGI("DetectionPipeline: Loaded calibration from ", config.calibration_file);
        } else {
            LOGW("DetectionPipeline: Failed to load calibration file: ",
                 config.calibration_file, "; undistort disabled");
        }
    }

    // Create source
    impl_->source = createSource(config);
    if (!impl_->source) {
        LOGE("DetectionPipeline: Failed to create source");
        return false;
    }

    // Open source
    if (!impl_->source->open(config.source_uri)) {
        LOGE("DetectionPipeline: Failed to open source: ", config.source_uri);
        return false;
    }

    // Create inference engine
    impl_->engine = createEngine(config);
    if (!impl_->engine) {
        LOGE("DetectionPipeline: Failed to create inference engine");
        return false;
    }

#if RKAPP_WITH_RKNN
    impl_->rknn_engine = dynamic_cast<infer::RknnEngine*>(impl_->engine.get());
    if (impl_->rknn_engine && config.use_npu_multicore) {
        impl_->rknn_engine->setCoreMask(0x7);  // All 3 NPU cores (6 TOPS)
    }
#else
    impl_->rknn_engine = nullptr;
#endif

    // Initialize engine
    if (!impl_->engine->init(config.model_path, config.input_size)) {
        LOGE("DetectionPipeline: Failed to initialize model: ", config.model_path);
        return false;
    }

    // Set decode parameters
    infer::DecodeParams decode_params;
    decode_params.conf_thres = config.conf_threshold;
    decode_params.iou_thres = config.iou_threshold;
    decode_params.max_boxes = config.max_detections;
    impl_->engine->setDecodeParams(decode_params);

    // Reset zero-copy state for clean re-init
    impl_->buffer_pool.reset();
    impl_->stats.zero_copy_enabled = false;

    // Create DMA buffer pool for zero-copy (if enabled and supported)
    if (config.use_zero_copy && impl_->rknn_engine && common::DmaBuf::isSupported()) {
        impl_->buffer_pool = std::make_unique<common::DmaBufPool>(
            config.buffer_pool_size,
            config.input_size,
            config.input_size,
            common::DmaBuf::PixelFormat::RGB888);
        if (impl_->buffer_pool->available() > 0) {
            impl_->stats.zero_copy_enabled = true;
            LOGI("DetectionPipeline: DMA-BUF zero-copy enabled with ",
                 impl_->buffer_pool->available(), " buffers");
        } else {
            LOGW("DetectionPipeline: DMA-BUF pool empty; zero-copy disabled");
            impl_->buffer_pool.reset();
            impl_->stats.zero_copy_enabled = false;
        }
    }

    // Warmup
    impl_->engine->warmup();

    // Initialize stats
    impl_->stats.rga_enabled = config.use_rga_preprocess;
    impl_->stats.mpp_enabled = isMppDecodeEnabledInStats(config);
    impl_->start_time = Clock::now();
    impl_->last_fps_update = impl_->start_time;

    impl_->initialized = true;
    LOGI("DetectionPipeline: Initialized successfully");
    LOGI("  - Input: ", config.source_uri);
    LOGI("  - Model: ", config.model_path, " (", config.input_size, "x", config.input_size, ")");
    LOGI("  - NPU multi-core: ", (config.use_npu_multicore ? "enabled" : "disabled"));
    LOGI("  - RGA preprocess: ", (impl_->stats.rga_enabled ? "enabled" : "disabled"));
    LOGI("  - MPP decode: ", (impl_->stats.mpp_enabled ? "enabled" : "disabled"));
    LOGI("  - Zero-copy: ", (impl_->stats.zero_copy_enabled ? "enabled" : "disabled"));
    LOGI("  - Undistort: ", (impl_->calibration_loaded ? "enabled" : "disabled"));
    LOGI("  - Preprocess profile: ", config.preprocess_profile);
    LOGI("  - ROI: ", (config.roi_enable ? "enabled" : "disabled"));
    LOGI("  - White balance: ", (impl_->preprocess_flags.white_balance ? "enabled" : "disabled"));
    LOGI("  - Gamma: ", (impl_->preprocess_flags.gamma ? "enabled" : "disabled"));
    LOGI("  - Denoise: ", (impl_->preprocess_flags.denoise ? "enabled" : "disabled"));

    return true;
}

std::optional<PipelineResult> DetectionPipeline::next() {
    if (!impl_->initialized || !impl_->source) {
        return std::nullopt;
    }

    capture::CaptureFrame frame;
    auto capture_start = Clock::now();

    if (!impl_->source->readFrame(frame)) {
        return std::nullopt;
    }

    if (frame.mat.empty()) {
        return std::nullopt;
    }

    auto result = process(frame.mat);
    const int64_t elapsed_from_capture_start = microsecondsSince(capture_start);
    result.timing.capture_us = std::max<int64_t>(
        0, elapsed_from_capture_start - result.timing.total_us);
    result.timing.total_us = elapsed_from_capture_start;
    result.frame_id = impl_->frame_counter++;

    impl_->updateFps();
    impl_->updateStats(result);

    return result;
}

PipelineResult DetectionPipeline::process(const cv::Mat& image) {
    PipelineResult result;
    auto total_start = Clock::now();

    if (!impl_->initialized || image.empty()) {
        return result;
    }

    preprocess::AccelBackend backend = impl_->config.use_rga_preprocess
        ? preprocess::AccelBackend::AUTO
        : preprocess::AccelBackend::OPENCV;

    auto preprocess_start = Clock::now();

    cv::Mat bgr_input = preprocess::Preprocess::ensureBgr8( image, backend);
    if (bgr_input.empty()) {
        LOGW("DetectionPipeline: Failed to normalize frame to BGR8");
        return result;
    }

    cv::Mat working = bgr_input;
    if (impl_->calibration_loaded) {
        if (impl_->undistort_size != bgr_input.size()) {
            if (preprocess::Preprocess::buildUndistortMaps(
                    impl_->calibration, bgr_input.size(), impl_->undistort_map1,
                    impl_->undistort_map2)) {
                impl_->undistort_size = bgr_input.size();
            } else {
                LOGW("DetectionPipeline: Failed to build undistort maps; bypassing undistort");
                impl_->undistort_map1.release();
                impl_->undistort_map2.release();
                impl_->undistort_size = {0, 0};
                impl_->calibration_loaded = false;
            }
        }
        if (!impl_->undistort_map1.empty() && !impl_->undistort_map2.empty()) {
            cv::Mat undistorted = preprocess::Preprocess::undistort(
                bgr_input, impl_->undistort_map1, impl_->undistort_map2);
            if (!undistorted.empty()) {
                working = std::move(undistorted);
            }
        }
    }

    const cv::Size coord_space_size = working.size();
    cv::Rect roi_rect(0, 0, working.cols, working.rows);
    bool roi_applied = false;
    if (impl_->config.roi_enable) {
        cv::Rect resolved_roi;
        const cv::Rect2f normalized_roi(
            impl_->config.roi_normalized_xywh[0], impl_->config.roi_normalized_xywh[1],
            impl_->config.roi_normalized_xywh[2], impl_->config.roi_normalized_xywh[3]);
        const cv::Rect pixel_roi(
            impl_->config.roi_pixel_xywh[0], impl_->config.roi_pixel_xywh[1],
            impl_->config.roi_pixel_xywh[2], impl_->config.roi_pixel_xywh[3]);
        if (preprocess::Preprocess::resolveRoiRect(
                working.size(), roiModeIsNormalized(impl_->config.roi_mode), normalized_roi,
                pixel_roi, impl_->config.roi_clamp, impl_->config.roi_min_size,
                resolved_roi)) {
            roi_rect = resolved_roi;
            if (roi_rect.x != 0 || roi_rect.y != 0 ||
                roi_rect.width != working.cols || roi_rect.height != working.rows) {
                cv::Mat cropped = preprocess::Preprocess::cropRoi(working, roi_rect);
                if (!cropped.empty()) {
                    working = std::move(cropped);
                    roi_applied = true;
                } else {
                    LOGW("DetectionPipeline: ROI crop failed, using full frame");
                    roi_rect = cv::Rect(0, 0, working.cols, working.rows);
                }
            }
        } else {
            LOGW("DetectionPipeline: Invalid ROI config, using full frame");
        }
    }

    if (impl_->preprocess_flags.denoise) {
        if (rkapp::common::toLowerCopy(impl_->config.denoise_method) != "bilateral") {
            LOGW("DetectionPipeline: Unsupported denoise method '",
                 impl_->config.denoise_method, "', using bilateral");
        }
        cv::Mat denoised = preprocess::Preprocess::denoiseBilateral(
            working, impl_->config.denoise_d, impl_->config.denoise_sigma_color,
            impl_->config.denoise_sigma_space);
        if (!denoised.empty()) {
            working = std::move(denoised);
        }
    }

    if (impl_->preprocess_flags.white_balance) {
        cv::Mat balanced = preprocess::Preprocess::whiteBalanceGrayWorld(
            working, impl_->config.white_balance_clip_percent);
        if (!balanced.empty()) {
            working = std::move(balanced);
        }
    }

    if (impl_->preprocess_flags.gamma) {
        const float gamma_value = impl_->config.gamma_value;
        if (gamma_value > 0.0f) {
            cv::Mat gamma_corrected =
                preprocess::Preprocess::applyGammaLut(working, gamma_value);
            if (!gamma_corrected.empty()) {
                working = std::move(gamma_corrected);
            }
        } else if (gamma_value <= 0.0f) {
            LOGW("DetectionPipeline: Invalid gamma value ", gamma_value, ", skipping gamma correction");
        }
    }

#if RKAPP_WITH_RKNN
    if (impl_->rknn_engine) {
        // Preprocess
        preprocess::LetterboxInfo letterbox_info;

        cv::Mat preprocessed = preprocess::Preprocess::letterbox(
            working, impl_->config.input_size, letterbox_info, backend);
        if (preprocessed.empty()) {
            LOGW("DetectionPipeline: Letterbox preprocessing failed");
            return result;
        }

        if (impl_->config.enable_profiling) {
            result.timing.preprocess_us = microsecondsSince(preprocess_start);
        }

        // Inference
        auto inference_start = Clock::now();

        // Lambda for running detection inference on the primary engine
        auto runDetectionInfer = [&]() -> std::vector<infer::Detection> {
            if (impl_->buffer_pool && impl_->stats.zero_copy_enabled) {
                common::DmaBuf* dma_buf = impl_->buffer_pool->acquire();
                if (dma_buf) {
                    cv::Mat rgb = preprocess::Preprocess::convertColor(
                        preprocessed, cv::COLOR_BGR2RGB, backend);
                    std::vector<infer::Detection> dets;
                    if (dma_buf->copyFrom(rgb)) {
                        dets = impl_->rknn_engine->inferDmaBuf(
                            *dma_buf, working.size(), letterbox_info);
                    } else {
                        dets = impl_->rknn_engine->inferPreprocessed(
                            preprocessed, working.size(), letterbox_info);
                    }
                    impl_->buffer_pool->release(dma_buf);
                    return dets;
                }
            }
            return impl_->rknn_engine->inferPreprocessed(
                preprocessed, working.size(), letterbox_info);
        };

        result.detections = runDetectionInfer();

        if (roi_applied) {
            applyRoiOffsetAndClip(result.detections, roi_rect, coord_space_size);
        }

        if (impl_->config.enable_profiling) {
            result.timing.inference_us = microsecondsSince(inference_start);
        }

        // Total timing
        result.timing.total_us = microsecondsSince(total_start);
        return result;
    }
#endif

    if (impl_->config.enable_profiling) {
        result.timing.preprocess_us = microsecondsSince(preprocess_start);
    }
    auto inference_start = Clock::now();
    result.detections = impl_->engine->infer(working);
    if (roi_applied) {
        applyRoiOffsetAndClip(result.detections, roi_rect, coord_space_size);
    }
    if (impl_->config.enable_profiling) {
        result.timing.inference_us = microsecondsSince(inference_start);
    }
    result.timing.total_us = microsecondsSince(total_start);
    return result;
}

void DetectionPipeline::runAsync(ResultCallback callback) {
    if (!impl_->initialized) {
        LOGE("DetectionPipeline::runAsync: Pipeline not initialized");
        return;
    }

    if (impl_->running) {
        LOGW("DetectionPipeline::runAsync: Already running");
        return;
    }

    impl_->result_callback = std::move(callback);
    impl_->running = true;

    impl_->worker_thread = std::thread([this]() {
        LOGI("DetectionPipeline: Async worker started");

        // Retry logic for transient failures (e.g., RTSP/GIGE timeouts)
        constexpr int MAX_CONSECUTIVE_FAILURES = 3;
        constexpr int RETRY_DELAY_MS = 100;
        int consecutive_failures = 0;

        while (impl_->running) {
            auto result = this->next();

            if (!result) {
                // Read failure - check if transient or fatal
                consecutive_failures++;

                if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES) {
                    LOGE("DetectionPipeline: Max consecutive read failures reached (",
                         MAX_CONSECUTIVE_FAILURES, "), stopping");
                    break;
                }

                LOGW("DetectionPipeline: Read failed (", consecutive_failures, "/",
                     MAX_CONSECUTIVE_FAILURES, "), retrying...");
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                continue;
            }

            // Reset failure counter on success
            consecutive_failures = 0;

            // Invoke callback with exception handling
            if (impl_->result_callback) {
                try {
                    impl_->result_callback(std::move(*result));
                } catch (const std::exception& e) {
                    LOGE("DetectionPipeline: Callback threw exception: ", e.what());
                    // Continue processing despite callback error
                } catch (...) {
                    LOGE("DetectionPipeline: Callback threw unknown exception");
                    // Continue processing despite callback error
                }
            }
        }

        impl_->running = false;
        LOGI("DetectionPipeline: Async worker stopped");
    });
}

void DetectionPipeline::stop() {
    impl_->running = false;

    if (impl_->worker_thread.joinable()) {
        impl_->worker_thread.join();
    }

    if (impl_->engine) {
        impl_->engine->release();
        impl_->engine.reset();
    }

    if (impl_->source) {
        impl_->source->release();
        impl_->source.reset();
    }
}

bool DetectionPipeline::isRunning() const {
    return impl_->running;
}

double DetectionPipeline::getFps() const {
    return impl_->current_fps;
}

DetectionPipeline::Statistics DetectionPipeline::getStatistics() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    return impl_->stats;
}

void DetectionPipeline::resetStatistics() {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    impl_->stats = Statistics{};
    impl_->stats.rga_enabled = impl_->config.use_rga_preprocess;
    impl_->stats.mpp_enabled = isMppDecodeEnabledInStats(impl_->config);
    impl_->stats.zero_copy_enabled = (impl_->buffer_pool != nullptr);
    impl_->total_latency_us = 0;
    impl_->start_time = Clock::now();
    impl_->last_fps_update = impl_->start_time;
    impl_->frames_since_update = 0;
}

// ============================================================================
// Factory Functions
// ============================================================================

capture::SourcePtr createSource(const PipelineConfig& config) {
    switch (config.source_type) {
        case capture::SourceType::FOLDER:
            return std::make_unique<capture::FolderSource>();

        case capture::SourceType::VIDEO:
        case capture::SourceType::RTSP:
#if RKAPP_WITH_MPP
            if (config.use_mpp_decode) {
                LOGI("DetectionPipeline: Using MPP hardware video decode");
                return std::make_unique<capture::MppSource>();
            }
#endif
            return std::make_unique<capture::VideoSource>();

        case capture::SourceType::GIGE:
#if RKAPP_WITH_GIGE
            return std::make_unique<capture::GigeSource>();
#else
            LOGW("DetectionPipeline: GIGE not available, falling back to VideoSource");
            return std::make_unique<capture::VideoSource>();
#endif

        case capture::SourceType::CSI:
#if RKAPP_WITH_CSI
            return std::make_unique<capture::CsiSource>();
#else
            LOGW("DetectionPipeline: CSI not available, falling back to VideoSource");
            return std::make_unique<capture::VideoSource>();
#endif

        case capture::SourceType::MPP:
#if RKAPP_WITH_MPP
            return std::make_unique<capture::MppSource>();
#else
            LOGW("DetectionPipeline: MPP not available, falling back to VideoSource");
            return std::make_unique<capture::VideoSource>();
#endif

        default:
            LOGW("DetectionPipeline: Unknown source type, using VideoSource");
            return std::make_unique<capture::VideoSource>();
    }
}

std::unique_ptr<infer::IInferEngine> createEngine(const PipelineConfig& config) {
    const std::string model_path_lower = rkapp::common::toLowerCopy(config.model_path);

    // Always use RKNN engine for .rknn models
    if (model_path_lower.find(".rknn") != std::string::npos) {
#if RKAPP_WITH_RKNN
        return std::make_unique<infer::RknnEngine>();
#else
        LOGE("DetectionPipeline: RKNN model specified but RKNN not enabled at build time");
        return nullptr;
#endif
    }

    if (model_path_lower.find(".onnx") != std::string::npos) {
#if RKAPP_WITH_ONNX
        return std::make_unique<infer::OnnxEngine>();
#else
        LOGE("DetectionPipeline: ONNX model specified but ONNX not enabled at build time");
        return nullptr;
#endif
    }

    LOGE("DetectionPipeline: Unsupported model type (expected .rknn or .onnx): ", config.model_path);
    return nullptr;
}

} // namespace rkapp::pipeline
