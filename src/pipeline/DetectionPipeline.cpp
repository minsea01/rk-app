#include "rkapp/pipeline/DetectionPipeline.hpp"
#include "rkapp/capture/VideoSource.hpp"
#include "rkapp/capture/FolderSource.hpp"
#include "rkapp/capture/MppSource.hpp"
#include "rkapp/capture/GigeSource.hpp"
#include "rkapp/capture/CsiSource.hpp"
#include "rkapp/capture/FrameOps.hpp"
#include "rkapp/infer/RknnEngine.hpp"
#include "rkapp/infer/OnnxEngine.hpp"
#include "rkapp/pipeline/BoxTracker.hpp"
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

// 管线主实现：组织采集、预处理、推理和后处理全流程。
// 使用 steady_clock 统计耗时，避免系统时间跳变影响。
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

namespace {

int64_t microsecondsSince(const TimePoint& start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start).count();
}

bool isMppDecodeEnabledInStats(const PipelineConfig& config) {
#if RKAPP_WITH_MPP
    if (config.source.type == capture::SourceType::MPP) {
        return true;
    }
    if ((config.source.type == capture::SourceType::VIDEO ||
         config.source.type == capture::SourceType::RTSP) &&
        config.source.use_mpp_decode) {
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
    const std::string profile = rkapp::common::toLowerCopy(config.preprocess.profile);
    if (profile == "balanced") {
        flags.gamma = true;
        flags.white_balance = true;
    } else if (profile == "quality") {
        flags.gamma = true;
        flags.white_balance = true;
        flags.denoise = true;
    }

    if (config.preprocess.gamma_enable.has_value()) {
        flags.gamma = *config.preprocess.gamma_enable;
    }
    if (config.preprocess.white_balance_enable.has_value()) {
        flags.white_balance = *config.preprocess.white_balance_enable;
    }
    if (config.preprocess.denoise_enable.has_value()) {
        flags.denoise = *config.preprocess.denoise_enable;
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
// Pipeline 内部实现
// ============================================================================

struct DetectionPipeline::Impl {
    PipelineConfig config;

    // 管线核心组件
    capture::SourcePtr source;
    std::unique_ptr<infer::IInferEngine> engine;
    infer::RknnEngine* rknn_engine = nullptr;
    std::unique_ptr<common::DmaBufPool> buffer_pool;

    // 运行状态
    std::atomic<bool> running{false};
    std::atomic<bool> initialized{false};
    std::atomic<int64_t> frame_counter{0};

    // 统计数据
    mutable std::mutex stats_mutex;
    Statistics stats;
    TimePoint start_time;
    int64_t total_latency_us{0};

    // 异步处理
    std::thread worker_thread;
    ResultCallback result_callback;

    // FPS 统计
    std::atomic<double> current_fps{0.0};
    TimePoint last_fps_update;
    int frames_since_update{0};

    // 可选去畸变状态
    preprocess::CameraCalibration calibration;
    bool calibration_loaded{false};
    cv::Mat undistort_map1;
    cv::Mat undistort_map2;
    cv::Size undistort_size{0, 0};
    PreprocessFeatureFlags preprocess_flags;
    int consecutive_frame_errors = 0;
    BoxTracker tracker;

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

    void recordDroppedFrame() {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.frames_dropped++;
    }

    void recordReconnectAttempt() {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.reconnect_count++;
    }

    void configureTracker() {
        BoxTracker::Config tracker_config;
        tracker_config.enable = config.tracking.enable;
        tracker_config.match_iou = config.tracking.match_iou;
        tracker_config.ema_alpha = config.tracking.ema_alpha;
        tracker_config.confirm_hits = config.tracking.confirm_hits;
        tracker_config.max_misses = config.tracking.max_misses;
        tracker_config.keep_missing_tracks = config.tracking.keep_missing_tracks;
        tracker_config.missing_conf_decay = config.tracking.missing_conf_decay;
        tracker.configure(tracker_config);
    }

    void maybeStabilizeDetections(PipelineResult& result) {
        if (!config.tracking.enable) {
            return;
        }
        result.detections = tracker.update(result.detections);
    }
};

namespace {

} // namespace

// ============================================================================
// 对外 API
// ============================================================================

DetectionPipeline::DetectionPipeline()
    : impl_(std::make_unique<Impl>()) {}

DetectionPipeline::~DetectionPipeline() {
    stop();
}

bool DetectionPipeline::init(const PipelineConfig& config) {
    impl_->initialized = false;  // 重置状态，确保失败时不残留旧状态

    // 重新初始化前先释放旧资源，避免状态污染。
    if (impl_->engine) {
        impl_->engine->release();
        impl_->engine.reset();
    }
    if (impl_->source) {
        impl_->source->release();
        impl_->source.reset();
    }

    std::vector<std::string> normalize_warnings;
    impl_->config = normalizePipelineConfig(config, &normalize_warnings);
    for (const auto& warning : normalize_warnings) {
        LOGW("DetectionPipeline: ", warning);
    }
    impl_->rknn_engine = nullptr;
    impl_->calibration = {};
    impl_->calibration_loaded = false;
    impl_->undistort_map1.release();
    impl_->undistort_map2.release();
    impl_->undistort_size = {0, 0};
    impl_->preprocess_flags = resolveFeatureFlags(impl_->config);
    impl_->configureTracker();
    impl_->frame_counter.store(0, std::memory_order_relaxed);

    if (impl_->config.preprocess.enable_undistort) {
        if (impl_->config.preprocess.calibration_file.empty()) {
            LOGW("DetectionPipeline: Undistort requested but calibration_file is empty");
        } else if (preprocess::Preprocess::loadCalibration(
                       impl_->config.preprocess.calibration_file, impl_->calibration)) {
            impl_->calibration_loaded = true;
            LOGI("DetectionPipeline: Loaded calibration from ",
                 impl_->config.preprocess.calibration_file);
        } else {
            LOGW("DetectionPipeline: Failed to load calibration file: ",
                 impl_->config.preprocess.calibration_file, "; undistort disabled");
        }
    }

    // 创建输入源。
    impl_->source = createSource(impl_->config.source);
    if (!impl_->source) {
        LOGE("DetectionPipeline: Failed to create source");
        return false;
    }

    // 打开输入源。
    if (!impl_->source->open(impl_->config.source.uri)) {
        LOGE("DetectionPipeline: Failed to open source: ", impl_->config.source.uri);
        return false;
    }

    // 创建推理引擎。
    impl_->engine = createEngine(impl_->config.model);
    if (!impl_->engine) {
        LOGE("DetectionPipeline: Failed to create inference engine");
        return false;
    }

#if RKAPP_WITH_RKNN
    impl_->rknn_engine = dynamic_cast<infer::RknnEngine*>(impl_->engine.get());
    if (impl_->rknn_engine && impl_->config.model.use_npu_multicore) {
        impl_->rknn_engine->setCoreMask(0x7);  // All 3 NPU cores (6 TOPS)
    }
#else
    impl_->rknn_engine = nullptr;
#endif

    // 初始化推理引擎。
    if (!impl_->engine->init(impl_->config.model)) {
        LOGE("DetectionPipeline: Failed to initialize model: ",
             impl_->config.model.model_path);
        return false;
    }

    // 下发解码参数。
    infer::DecodeParams decode_params;
    decode_params.conf_thres = impl_->config.model.conf_threshold;
    decode_params.iou_thres = impl_->config.model.iou_threshold;
    decode_params.max_boxes = impl_->config.model.max_detections;
    impl_->engine->setDecodeParams(decode_params);

    // 重置零拷贝状态，保证重初始化后状态一致。
    impl_->buffer_pool.reset();
    impl_->stats.zero_copy_enabled = false;

    // 仅 source 提供兼容 DMA 输入时才视为“零拷贝”。
    if (impl_->config.model.use_zero_copy && impl_->rknn_engine && common::DmaBuf::isSupported()) {
        LOGW("DetectionPipeline: Direct zero-copy is reserved for source-provided DMA frames; "
             "cv::Mat-to-DMA staging is not treated as zero-copy and remains disabled");
    }

    // 预热模型，降低首帧延迟。
    const int warmup_iterations = std::max(0, impl_->config.runtime.warmup_iterations);
    for (int i = 0; i < warmup_iterations; ++i) {
        impl_->engine->warmup();
    }

    // 初始化统计项。
    impl_->stats.rga_enabled = impl_->config.preprocess.use_rga_preprocess;
    impl_->stats.mpp_enabled = isMppDecodeEnabledInStats(impl_->config);
    impl_->start_time = Clock::now();
    impl_->last_fps_update = impl_->start_time;
    impl_->current_fps.store(0.0, std::memory_order_relaxed);

    impl_->initialized = true;
    LOGI("DetectionPipeline: Initialized successfully");
    LOGI("  - Input: ", impl_->config.source.uri);
    LOGI("  - Model: ", impl_->config.model.model_path, " (", impl_->config.model.input_size,
         "x", impl_->config.model.input_size, ")");
    LOGI("  - NPU multi-core: ",
         (impl_->config.model.use_npu_multicore ? "enabled" : "disabled"));
    LOGI("  - RGA preprocess: ", (impl_->stats.rga_enabled ? "enabled" : "disabled"));
    LOGI("  - MPP decode: ", (impl_->stats.mpp_enabled ? "enabled" : "disabled"));
    LOGI("  - Zero-copy: ", (impl_->stats.zero_copy_enabled ? "enabled" : "disabled"));
    LOGI("  - Runtime warmup: ", impl_->config.runtime.warmup_iterations);
    LOGI("  - Runtime async: ", (impl_->config.runtime.async_mode ? "enabled" : "disabled"));
    LOGI("  - Tracking: ", (impl_->config.tracking.enable ? "enabled" : "disabled"));
    LOGI("  - Undistort: ", (impl_->calibration_loaded ? "enabled" : "disabled"));
    LOGI("  - Preprocess profile: ", impl_->config.preprocess.profile);
    LOGI("  - ROI: ", (impl_->config.preprocess.roi_enable ? "enabled" : "disabled"));
    LOGI("  - White balance: ", (impl_->preprocess_flags.white_balance ? "enabled" : "disabled"));
    LOGI("  - Gamma: ", (impl_->preprocess_flags.gamma ? "enabled" : "disabled"));
    LOGI("  - Denoise: ", (impl_->preprocess_flags.denoise ? "enabled" : "disabled"));

    return true;
}

std::optional<PipelineResult> DetectionPipeline::next() {
    return nextInternal(false);
}

std::optional<PipelineResult> DetectionPipeline::nextInternal(bool respect_running_flag) {
    if (!impl_->initialized || !impl_->source) {
        return std::nullopt;
    }

    const int configured_backoff = std::max(0, impl_->config.failure.initial_backoff_ms);
    const int configured_backoff_max =
        std::max(configured_backoff, impl_->config.failure.max_backoff_ms);
    std::chrono::milliseconds backoff(configured_backoff);
    const std::chrono::milliseconds backoff_max(configured_backoff_max);
    const std::chrono::milliseconds grace_window(
        std::max(0, impl_->config.failure.source_recovery_grace_ms));
    std::optional<TimePoint> recovery_start;
    int reconnect_attempts = 0;

    while (!respect_running_flag || impl_->running.load(std::memory_order_acquire)) {
        capture::CaptureFrame frame;
        auto capture_start = Clock::now();
        const auto read_status = impl_->source->readFrameEx(frame);

        if (read_status == capture::ReadStatus::FrameReady) {
            if (recovery_start.has_value()) {
                const auto recovered_after = std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now() - *recovery_start);
                LOGI("DetectionPipeline: Source recovered after ", recovered_after.count(),
                     " ms and ", reconnect_attempts, " retry attempts");
            }
            recovery_start.reset();
            reconnect_attempts = 0;
            backoff = std::chrono::milliseconds(configured_backoff);

            if (frame.mat.empty()) {
                impl_->recordDroppedFrame();
                impl_->consecutive_frame_errors++;
                if (impl_->config.failure.frame_error_limit >= 0 &&
                    impl_->consecutive_frame_errors > impl_->config.failure.frame_error_limit) {
                    LOGE("DetectionPipeline: Too many consecutive frame errors (empty frames)");
                    return std::nullopt;
                }
                LOGW("DetectionPipeline: Dropping empty frame");
                continue;
            }

            auto result = process(frame);
            if (result.timing.total_us <= 0 && result.frame.empty() && result.detections.empty()) {
                impl_->recordDroppedFrame();
                impl_->consecutive_frame_errors++;
                if (impl_->config.failure.frame_error_limit >= 0 &&
                    impl_->consecutive_frame_errors > impl_->config.failure.frame_error_limit) {
                    LOGE("DetectionPipeline: Too many consecutive frame processing failures");
                    return std::nullopt;
                }
                LOGW("DetectionPipeline: Frame processing failed, dropping frame");
                continue;
            }

            impl_->consecutive_frame_errors = 0;
            const int64_t elapsed_from_capture_start = microsecondsSince(capture_start);
            result.timing.capture_us = std::max<int64_t>(
                0, elapsed_from_capture_start - result.timing.total_us);
            result.timing.total_us = elapsed_from_capture_start;
            result.frame_id = impl_->frame_counter.fetch_add(1, std::memory_order_relaxed);

            impl_->updateFps();
            impl_->updateStats(result);
            return result;
        }

        if (read_status == capture::ReadStatus::EndOfStream) {
            LOGI("DetectionPipeline: Source exhausted");
            return std::nullopt;
        }

        if (read_status == capture::ReadStatus::FatalError) {
            LOGE("DetectionPipeline: Source reported fatal read error");
            return std::nullopt;
        }

        const auto now = Clock::now();
        if (!recovery_start.has_value()) {
            recovery_start = now;
        }

        const int next_attempt = reconnect_attempts + 1;
        if (impl_->config.failure.max_reconnect_attempts >= 0 &&
            next_attempt > impl_->config.failure.max_reconnect_attempts) {
            LOGE("DetectionPipeline: Reconnect attempts exceeded limit (",
                 impl_->config.failure.max_reconnect_attempts, ")");
            return std::nullopt;
        }
        if (grace_window.count() > 0 && now - *recovery_start >= grace_window) {
            LOGE("DetectionPipeline: Source did not recover within ", grace_window.count(),
                 " ms");
            return std::nullopt;
        }

        reconnect_attempts = next_attempt;
        impl_->recordReconnectAttempt();
        LOGW("DetectionPipeline: Recoverable source read failure, retry ", reconnect_attempts,
             " in ", backoff.count(), " ms");

        if (respect_running_flag && !impl_->running.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        if (backoff.count() > 0) {
            std::this_thread::sleep_for(backoff);
        }
        backoff = std::min(backoff_max, std::max(backoff, std::chrono::milliseconds(1)) * 2);
    }

    return std::nullopt;
}

PipelineResult DetectionPipeline::process(const cv::Mat& image) {
    capture::CaptureFrame frame;
    frame.setMatFrame(image, capture::PixelFormat::BGR888);
    return process(frame);
}

PipelineResult DetectionPipeline::process(const capture::CaptureFrame& frame) {
    PipelineResult result;
    auto total_start = Clock::now();

    if (!impl_->initialized || frame.mat.empty()) {
        return result;
    }

    preprocess::AccelBackend backend = impl_->config.preprocess.use_rga_preprocess
        ? preprocess::AccelBackend::AUTO
        : preprocess::AccelBackend::OPENCV;

    auto preprocess_start = Clock::now();

#if RKAPP_WITH_RKNN
    const bool gray_fast_path =
        impl_->rknn_engine && frame.pixel_format == capture::PixelFormat::GRAY8 &&
        frame.mat.type() == CV_8UC1 && !impl_->calibration_loaded &&
        !impl_->preprocess_flags.denoise && !impl_->preprocess_flags.white_balance &&
        !impl_->preprocess_flags.gamma;
    if (gray_fast_path) {
        cv::Mat working = frame.mat;
        bool working_aliases_source = true;
        const cv::Size coord_space_size = working.size();
        cv::Rect roi_rect(0, 0, working.cols, working.rows);
        bool roi_applied = false;

        if (impl_->config.preprocess.roi_enable) {
            cv::Rect resolved_roi;
            const cv::Rect2f normalized_roi(
                impl_->config.preprocess.roi_normalized_xywh[0],
                impl_->config.preprocess.roi_normalized_xywh[1],
                impl_->config.preprocess.roi_normalized_xywh[2],
                impl_->config.preprocess.roi_normalized_xywh[3]);
            const cv::Rect pixel_roi(
                impl_->config.preprocess.roi_pixel_xywh[0],
                impl_->config.preprocess.roi_pixel_xywh[1],
                impl_->config.preprocess.roi_pixel_xywh[2],
                impl_->config.preprocess.roi_pixel_xywh[3]);
            if (preprocess::Preprocess::resolveRoiRect(
                    working.size(), roiModeIsNormalized(impl_->config.preprocess.roi_mode),
                    normalized_roi, pixel_roi, impl_->config.preprocess.roi_clamp,
                    impl_->config.preprocess.roi_min_size, resolved_roi)) {
                roi_rect = resolved_roi;
                if (roi_rect.x != 0 || roi_rect.y != 0 ||
                    roi_rect.width != working.cols || roi_rect.height != working.rows) {
                    cv::Mat cropped = preprocess::Preprocess::cropRoi(working, roi_rect);
                    if (!cropped.empty()) {
                        working = std::move(cropped);
                        working_aliases_source = false;
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

        preprocess::LetterboxInfo letterbox_info;
        cv::Mat preprocessed_gray = preprocess::Preprocess::letterbox(
            working, impl_->config.model.input_size, letterbox_info,
            preprocess::AccelBackend::OPENCV);
        if (preprocessed_gray.empty()) {
            LOGW("DetectionPipeline: Gray letterbox preprocessing failed");
            return result;
        }

        cv::Mat preprocessed_bgr;
        cv::cvtColor(preprocessed_gray, preprocessed_bgr, cv::COLOR_GRAY2BGR);

        if (impl_->config.output.enable_profiling) {
            result.timing.preprocess_us = microsecondsSince(preprocess_start);
        }

        auto inference_start = Clock::now();
        result.detections = impl_->rknn_engine->inferPreprocessed(
            preprocessed_bgr, working.size(), letterbox_info);

        if (roi_applied) {
            applyRoiOffsetAndClip(result.detections, roi_rect, coord_space_size);
        }

        auto postprocess_start = Clock::now();
        post::Postprocess::mapClassNames(result.detections, impl_->config.model.class_names);
        if (impl_->config.model.min_box_size > 0.0f || impl_->config.model.max_box_size > 0.0f ||
            impl_->config.model.min_aspect_ratio > 0.0f ||
            impl_->config.model.max_aspect_ratio > 0.0f) {
            post::NMSConfig filter_cfg;
            filter_cfg.conf_thres = 0.0f;
            filter_cfg.iou_thres = 1.0f;
            filter_cfg.topk = 0;
            filter_cfg.max_det = impl_->config.model.max_detections;
            filter_cfg.min_box_size = impl_->config.model.min_box_size;
            filter_cfg.max_box_size = impl_->config.model.max_box_size;
            filter_cfg.min_aspect_ratio = impl_->config.model.min_aspect_ratio;
            filter_cfg.max_aspect_ratio = impl_->config.model.max_aspect_ratio;
            result.detections = post::Postprocess::nms(result.detections, filter_cfg);
        }
        impl_->maybeStabilizeDetections(result);
        result.frame = capture::detachFrameIfAliased(working, frame, working_aliases_source);

        if (impl_->config.output.enable_profiling) {
            result.timing.inference_us = microsecondsSince(inference_start);
            result.timing.postprocess_us = microsecondsSince(postprocess_start);
        }

        result.timing.total_us = microsecondsSince(total_start);
        return result;
    }
#endif

    capture::BgrFrameView bgr_view = capture::convertToBgr(frame, backend);
    if (bgr_view.empty()) {
        LOGW("DetectionPipeline: Failed to normalize frame to BGR8");
        return result;
    }

    cv::Mat working = bgr_view.image;
    bool working_aliases_source = bgr_view.aliases_source;
    if (impl_->calibration_loaded) {
        if (impl_->undistort_size != bgr_view.image.size()) {
            if (preprocess::Preprocess::buildUndistortMaps(
                    impl_->calibration, bgr_view.image.size(), impl_->undistort_map1,
                    impl_->undistort_map2)) {
                impl_->undistort_size = bgr_view.image.size();
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
                bgr_view.image, impl_->undistort_map1, impl_->undistort_map2);
            if (!undistorted.empty()) {
                working = std::move(undistorted);
                working_aliases_source = false;
            }
        }
    }

    cv::Mat coord_frame = working;
    bool coord_frame_aliases_source = working_aliases_source;
    const cv::Size coord_space_size = working.size();
    cv::Rect roi_rect(0, 0, working.cols, working.rows);
    bool roi_applied = false;
    if (impl_->config.preprocess.roi_enable) {
        cv::Rect resolved_roi;
        const cv::Rect2f normalized_roi(
            impl_->config.preprocess.roi_normalized_xywh[0],
            impl_->config.preprocess.roi_normalized_xywh[1],
            impl_->config.preprocess.roi_normalized_xywh[2],
            impl_->config.preprocess.roi_normalized_xywh[3]);
        const cv::Rect pixel_roi(
            impl_->config.preprocess.roi_pixel_xywh[0],
            impl_->config.preprocess.roi_pixel_xywh[1],
            impl_->config.preprocess.roi_pixel_xywh[2],
            impl_->config.preprocess.roi_pixel_xywh[3]);
        if (preprocess::Preprocess::resolveRoiRect(
                working.size(), roiModeIsNormalized(impl_->config.preprocess.roi_mode),
                normalized_roi, pixel_roi, impl_->config.preprocess.roi_clamp,
                impl_->config.preprocess.roi_min_size,
                resolved_roi)) {
            roi_rect = resolved_roi;
            if (roi_rect.x != 0 || roi_rect.y != 0 ||
                roi_rect.width != working.cols || roi_rect.height != working.rows) {
                cv::Mat cropped = preprocess::Preprocess::cropRoi(working, roi_rect);
                if (!cropped.empty()) {
                    working = std::move(cropped);
                    working_aliases_source = false;
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
        if (rkapp::common::toLowerCopy(impl_->config.preprocess.denoise_method) != "bilateral") {
            LOGW("DetectionPipeline: Unsupported denoise method '",
                 impl_->config.preprocess.denoise_method, "', using bilateral");
        }
        cv::Mat denoised = preprocess::Preprocess::denoiseBilateral(
            working, impl_->config.preprocess.denoise_d,
            impl_->config.preprocess.denoise_sigma_color,
            impl_->config.preprocess.denoise_sigma_space);
        if (!denoised.empty()) {
            working = std::move(denoised);
            working_aliases_source = false;
        }
    }

    if (impl_->preprocess_flags.white_balance) {
        cv::Mat balanced = preprocess::Preprocess::whiteBalanceGrayWorld(
            working, impl_->config.preprocess.white_balance_clip_percent);
        if (!balanced.empty()) {
            working = std::move(balanced);
            working_aliases_source = false;
        }
    }

    if (impl_->preprocess_flags.gamma) {
        const float gamma_value = impl_->config.preprocess.gamma_value;
        if (gamma_value > 0.0f) {
            cv::Mat gamma_corrected =
                preprocess::Preprocess::applyGammaLut(working, gamma_value);
            if (!gamma_corrected.empty()) {
                working = std::move(gamma_corrected);
                working_aliases_source = false;
            }
        } else if (gamma_value <= 0.0f) {
            LOGW("DetectionPipeline: Invalid gamma value ", gamma_value, ", skipping gamma correction");
        }
    }
    coord_frame = working;
    coord_frame_aliases_source = working_aliases_source;

#if RKAPP_WITH_RKNN
    if (impl_->rknn_engine) {
        // 预处理（letterbox）。
        preprocess::LetterboxInfo letterbox_info;

        cv::Mat preprocessed = preprocess::Preprocess::letterbox(
            working, impl_->config.model.input_size, letterbox_info, backend);
        if (preprocessed.empty()) {
            LOGW("DetectionPipeline: Letterbox preprocessing failed");
            return result;
        }

        if (impl_->config.output.enable_profiling) {
            result.timing.preprocess_us = microsecondsSince(preprocess_start);
        }

        // 推理。
        auto inference_start = Clock::now();
        result.detections = impl_->rknn_engine->inferPreprocessed(
            preprocessed, working.size(), letterbox_info);

        if (roi_applied) {
            applyRoiOffsetAndClip(result.detections, roi_rect, coord_space_size);
        }

        auto postprocess_start = Clock::now();
        post::Postprocess::mapClassNames(result.detections, impl_->config.model.class_names);
        if (impl_->config.model.min_box_size > 0.0f || impl_->config.model.max_box_size > 0.0f ||
            impl_->config.model.min_aspect_ratio > 0.0f ||
            impl_->config.model.max_aspect_ratio > 0.0f) {
            post::NMSConfig filter_cfg;
            filter_cfg.conf_thres = 0.0f;
            filter_cfg.iou_thres = 1.0f;
            filter_cfg.topk = 0;
            filter_cfg.max_det = impl_->config.model.max_detections;
            filter_cfg.min_box_size = impl_->config.model.min_box_size;
            filter_cfg.max_box_size = impl_->config.model.max_box_size;
            filter_cfg.min_aspect_ratio = impl_->config.model.min_aspect_ratio;
            filter_cfg.max_aspect_ratio = impl_->config.model.max_aspect_ratio;
            result.detections = post::Postprocess::nms(result.detections, filter_cfg);
        }
        impl_->maybeStabilizeDetections(result);
        result.frame = capture::detachFrameIfAliased(coord_frame, frame,
                                                     coord_frame_aliases_source);

        if (impl_->config.output.enable_profiling) {
            result.timing.inference_us = microsecondsSince(inference_start);
            result.timing.postprocess_us = microsecondsSince(postprocess_start);
        }

        // 总耗时统计。
        result.timing.total_us = microsecondsSince(total_start);
        return result;
    }
#endif

    if (impl_->config.output.enable_profiling) {
        result.timing.preprocess_us = microsecondsSince(preprocess_start);
    }
    auto inference_start = Clock::now();
    result.detections = impl_->engine->infer(working);
    if (roi_applied) {
        applyRoiOffsetAndClip(result.detections, roi_rect, coord_space_size);
    }
    auto postprocess_start = Clock::now();
    post::Postprocess::mapClassNames(result.detections, impl_->config.model.class_names);
    if (impl_->config.model.min_box_size > 0.0f || impl_->config.model.max_box_size > 0.0f ||
        impl_->config.model.min_aspect_ratio > 0.0f ||
        impl_->config.model.max_aspect_ratio > 0.0f) {
        post::NMSConfig filter_cfg;
        filter_cfg.conf_thres = 0.0f;
        filter_cfg.iou_thres = 1.0f;
        filter_cfg.topk = 0;
        filter_cfg.max_det = impl_->config.model.max_detections;
        filter_cfg.min_box_size = impl_->config.model.min_box_size;
        filter_cfg.max_box_size = impl_->config.model.max_box_size;
        filter_cfg.min_aspect_ratio = impl_->config.model.min_aspect_ratio;
        filter_cfg.max_aspect_ratio = impl_->config.model.max_aspect_ratio;
        result.detections = post::Postprocess::nms(result.detections, filter_cfg);
    }
    impl_->maybeStabilizeDetections(result);
    result.frame = capture::detachFrameIfAliased(coord_frame, frame,
                                                 coord_frame_aliases_source);
    if (impl_->config.output.enable_profiling) {
        result.timing.inference_us = microsecondsSince(inference_start);
        result.timing.postprocess_us = microsecondsSince(postprocess_start);
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

        while (impl_->running) {
            auto result = this->nextInternal(true);

            if (!result) {
                LOGI("DetectionPipeline: Source exhausted or stopped; stopping async worker");
                break;
            }

            // 回调加异常保护，避免单次异常中断整个线程。
            if (impl_->result_callback) {
                try {
                    impl_->result_callback(std::move(*result));
                } catch (const std::exception& e) {
                    LOGE("DetectionPipeline: Callback threw exception: ", e.what());
                    // 回调异常不影响后续处理。
                } catch (...) {
                    LOGE("DetectionPipeline: Callback threw unknown exception");
                    // 回调异常不影响后续处理。
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

    impl_->buffer_pool.reset();
    impl_->initialized = false;
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
    impl_->stats.rga_enabled = impl_->config.preprocess.use_rga_preprocess;
    impl_->stats.mpp_enabled = isMppDecodeEnabledInStats(impl_->config);
    impl_->stats.zero_copy_enabled = false;
    impl_->total_latency_us = 0;
    impl_->consecutive_frame_errors = 0;
    impl_->tracker.reset();
    impl_->start_time = Clock::now();
    impl_->last_fps_update = impl_->start_time;
    impl_->frames_since_update = 0;
    impl_->current_fps.store(0.0, std::memory_order_relaxed);
}

// ============================================================================
// 工厂函数
// ============================================================================

capture::SourcePtr createSource(const PipelineConfig& config) {
    return createSource(config.source);
}

capture::SourcePtr createSource(const PipelineConfig::SourceSpec& source) {
    switch (source.type) {
        case capture::SourceType::FOLDER:
            return std::make_unique<capture::FolderSource>();

        case capture::SourceType::VIDEO:
        case capture::SourceType::RTSP:
#if RKAPP_WITH_MPP
            if (source.use_mpp_decode) {
                LOGI("DetectionPipeline: Using MPP hardware video decode");
                return std::make_unique<capture::MppSource>();
            }
#endif
            return std::make_unique<capture::VideoSource>();

        case capture::SourceType::GIGE:
#if RKAPP_WITH_GIGE
            return std::make_unique<capture::GigeSource>();
#else
            LOGE("DetectionPipeline: GIGE source requested but build does not enable GIGE");
            return nullptr;
#endif

        case capture::SourceType::CSI:
#if RKAPP_WITH_CSI
            return std::make_unique<capture::CsiSource>();
#else
            LOGE("DetectionPipeline: CSI source requested but build does not enable CSI");
            return nullptr;
#endif

        case capture::SourceType::MPP:
#if RKAPP_WITH_MPP
            return std::make_unique<capture::MppSource>();
#else
            LOGE("DetectionPipeline: MPP source requested but build does not enable MPP");
            return nullptr;
#endif

        default:
            LOGW("DetectionPipeline: Unknown source type, using VideoSource");
            return std::make_unique<capture::VideoSource>();
    }
}

std::unique_ptr<infer::IInferEngine> createEngine(const PipelineConfig& config) {
    return createEngine(config.model);
}

std::unique_ptr<infer::IInferEngine> createEngine(const PipelineConfig::ModelSpec& model) {
    PipelineConfig::ModelBackend backend = model.backend;
    const std::string model_path_lower = rkapp::common::toLowerCopy(model.model_path);
    if (backend == PipelineConfig::ModelBackend::AUTO) {
        if (model_path_lower.find(".rknn") != std::string::npos) {
            backend = PipelineConfig::ModelBackend::RKNN;
        } else if (model_path_lower.find(".onnx") != std::string::npos) {
            backend = PipelineConfig::ModelBackend::ONNX;
        }
    }

    // .rknn 模型优先走 RKNN 引擎。
    if (backend == PipelineConfig::ModelBackend::RKNN) {
#if RKAPP_WITH_RKNN
        return std::make_unique<infer::RknnEngine>();
#else
        LOGE("DetectionPipeline: RKNN model specified but RKNN not enabled at build time");
        return nullptr;
#endif
    }

    if (backend == PipelineConfig::ModelBackend::ONNX) {
#if RKAPP_WITH_ONNX
        return std::make_unique<infer::OnnxEngine>();
#else
        LOGE("DetectionPipeline: ONNX model specified but ONNX not enabled at build time");
        return nullptr;
#endif
    }

    LOGE("DetectionPipeline: Unsupported model type (expected .rknn or .onnx): ",
         model.model_path);
    return nullptr;
}

} // namespace rkapp::pipeline
