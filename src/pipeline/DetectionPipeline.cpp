#include "rkapp/pipeline/DetectionPipeline.hpp"
#include "rkapp/capture/VideoSource.hpp"
#include "rkapp/capture/FolderSource.hpp"
#include "rkapp/capture/MppSource.hpp"
#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/post/Postprocess.hpp"
#include "log.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

namespace rkapp::pipeline {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

namespace {

int64_t microsecondsSince(const TimePoint& start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start).count();
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
    impl_->config = config;

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
    if (config.use_zero_copy && common::DmaBuf::isSupported()) {
        impl_->buffer_pool = std::make_unique<common::DmaBufPool>(
            config.buffer_pool_size,
            config.input_size,
            config.input_size,
            common::DmaBuf::PixelFormat::RGB888);
        impl_->stats.zero_copy_enabled = true;
        LOGI("DetectionPipeline: DMA-BUF zero-copy enabled with ",
             config.buffer_pool_size, " buffers");
    }

    // Warmup
    impl_->engine->warmup();

    // Initialize stats
    impl_->stats.rga_enabled = config.use_rga_preprocess;
    impl_->stats.mpp_enabled = (config.source_type == capture::SourceType::MPP ||
                                config.source_type == capture::SourceType::VIDEO) &&
                               config.use_mpp_decode;
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

    return true;
}

std::optional<PipelineResult> DetectionPipeline::next() {
    if (!impl_->initialized || !impl_->source) {
        return std::nullopt;
    }

    cv::Mat frame;
    auto capture_start = Clock::now();

    if (!impl_->source->read(frame)) {
        return std::nullopt;
    }

    if (frame.empty()) {
        return std::nullopt;
    }

    auto result = process(frame);
    result.timing.capture_us = microsecondsSince(capture_start) - result.timing.total_us;
    result.timing.total_us += result.timing.capture_us;
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

    // Preprocess
    auto preprocess_start = Clock::now();
    preprocess::LetterboxInfo letterbox_info;
    preprocess::AccelBackend backend = impl_->config.use_rga_preprocess
        ? preprocess::AccelBackend::AUTO
        : preprocess::AccelBackend::OPENCV;

    cv::Mat preprocessed = preprocess::Preprocess::letterbox(
        image, impl_->config.input_size, letterbox_info, backend);

    if (impl_->config.enable_profiling) {
        result.timing.preprocess_us = microsecondsSince(preprocess_start);
    }

    // Inference
    auto inference_start = Clock::now();

    // Try zero-copy path if enabled and buffer pool available
    if (impl_->buffer_pool && impl_->stats.zero_copy_enabled) {
        common::DmaBuf* dma_buf = impl_->buffer_pool->acquire();
        if (dma_buf) {
            // Convert to RGB and copy to DMA buffer
            cv::Mat rgb;
            cv::cvtColor(preprocessed, rgb, cv::COLOR_BGR2RGB);
            if (dma_buf->copyFrom(rgb)) {
                // Cast to RknnEngine for inferDmaBuf
                auto* rknn_engine = dynamic_cast<infer::RknnEngine*>(impl_->engine.get());
                if (rknn_engine) {
                    result.detections = rknn_engine->inferDmaBuf(*dma_buf, image.size(), letterbox_info);
                } else {
                    // Non-RKNN engines do their own preprocessing
                    result.detections = impl_->engine->infer(image);
                }
            } else {
                // Fallback on copy failure: use preprocessed path for RKNN, otherwise original image
                auto* rknn_engine = dynamic_cast<infer::RknnEngine*>(impl_->engine.get());
                if (rknn_engine) {
                    result.detections = rknn_engine->inferPreprocessed(preprocessed, image.size(), letterbox_info);
                } else {
                    result.detections = impl_->engine->infer(image);
                }
            }
            impl_->buffer_pool->release(dma_buf);
        } else {
            // No buffer available, use regular path
            auto* rknn_engine = dynamic_cast<infer::RknnEngine*>(impl_->engine.get());
            if (rknn_engine) {
                result.detections = rknn_engine->inferPreprocessed(preprocessed, image.size(), letterbox_info);
            } else {
                result.detections = impl_->engine->infer(image);
            }
        }
    } else {
        // Standard inference path: pass preprocessed image to avoid double letterbox
        // Cast to RknnEngine to use inferPreprocessed
        auto* rknn_engine = dynamic_cast<infer::RknnEngine*>(impl_->engine.get());
        if (rknn_engine) {
            result.detections = rknn_engine->inferPreprocessed(preprocessed, image.size(), letterbox_info);
        } else {
            // Fallback: use original image (engine will do its own preprocessing)
            result.detections = impl_->engine->infer(image);
        }
    }

    if (impl_->config.enable_profiling) {
        result.timing.inference_us = microsecondsSince(inference_start);
    }

    // Total timing
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

    if (impl_->source) {
        impl_->source->release();
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
    impl_->stats.mpp_enabled = impl_->config.use_mpp_decode;
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
    // Always use RKNN engine for .rknn models
    if (config.model_path.find(".rknn") != std::string::npos) {
#if RKNN_PLATFORM || RKAPP_WITH_RKNN
        auto engine = std::make_unique<infer::RknnEngine>();
        if (config.use_npu_multicore) {
            engine->setCoreMask(0x7);  // All 3 NPU cores (6 TOPS)
        }
        return engine;
#else
        LOGE("DetectionPipeline: RKNN model specified but RKNN not enabled at build time");
        return nullptr;
#endif
    }

    // For ONNX models, could use OnnxEngine (not shown here)
    LOGE("DetectionPipeline: Only RKNN models are supported in this pipeline");
    return nullptr;
}

} // namespace rkapp::pipeline
