#include "rkapp/preprocess/Preprocess.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>

// RGA hardware acceleration headers (RK3588)
#if RKNN_USE_RGA
#include <im2d.h>
#include <rga.h>
#include <mutex>
#endif

// Logging support
#if __has_include("rkapp/common/log.hpp")
#include "rkapp/common/log.hpp"
#define RKAPP_HAS_LOG 1
#elif __has_include("log.hpp")
#include "log.hpp"
#define RKAPP_HAS_LOG 1
#else
#include <iostream>
#define LOGI(...) do { (void)0; } while(0)
#define LOGW(...) do { std::cerr << "[WARN] Preprocess: " << __VA_ARGS__ << std::endl; } while(0)
#define LOGE(...) do { std::cerr << "[ERROR] Preprocess: " << __VA_ARGS__ << std::endl; } while(0)
#define LOGD(...) do { (void)0; } while(0)
#endif

namespace rkapp::preprocess {

// ============================================================================
// RGA Hardware Acceleration Implementation (RK3588)
// ============================================================================

#if RKNN_USE_RGA

// Static initialization
bool Preprocess::rga_initialized_ = false;
static std::once_flag rga_init_flag;

bool Preprocess::initRga() {
    std::call_once(rga_init_flag, []() {
        // Query RGA hardware capability
        const char* version = querystring(RGA_VERSION);
        if (version != nullptr && strlen(version) > 0) {
            rga_initialized_ = true;
            LOGI("RGA hardware initialized: ", version);
        } else {
            rga_initialized_ = false;
            LOGW("RGA hardware not available, falling back to OpenCV");
        }
    });
    return rga_initialized_;
}

bool Preprocess::isRgaAvailable() {
    return initRga();
}

/**
 * @brief RGA-accelerated letterbox implementation
 *
 * Performance: ~0.3ms for 1080p -> 640x640 (vs ~3ms OpenCV)
 *
 * Pipeline:
 * 1. imresize() - Hardware bilinear scaling
 * 2. imcopy() - Blit to center of padded destination
 */
cv::Mat Preprocess::letterboxRga(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info) {
    if (!initRga()) {
        return letterboxCpu(src, target_size, info);
    }

    constexpr int kMinDim = 1;
    constexpr float kMinRatio = 1e-6f;

    // Validate input
    if (src.empty() || src.cols < kMinDim || src.rows < kMinDim ||
        target_size.width < kMinDim || target_size.height < kMinDim) {
        info = {0.f, 0.f, 0.f, 0, 0};
        return {};
    }

    // Only support BGR 3-channel images
    if (src.type() != CV_8UC3) {
        LOGW("RGA letterbox only supports CV_8UC3, falling back to CPU");
        return letterboxCpu(src, target_size, info);
    }

    int src_w = src.cols;
    int src_h = src.rows;
    int dst_w = target_size.width;
    int dst_h = target_size.height;

    // Calculate scale factor
    float scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
    if (scale < kMinRatio) {
        info = {0.f, 0.f, 0.f, 0, 0};
        return {};
    }

    int new_w = static_cast<int>(std::round(src_w * scale));
    int new_h = static_cast<int>(std::round(src_h * scale));

    // Calculate padding
    float dx = (dst_w - new_w) / 2.0f;
    float dy = (dst_h - new_h) / 2.0f;

    info.scale = scale;
    info.dx = dx;
    info.dy = dy;
    info.new_width = new_w;
    info.new_height = new_h;

    // Create destination buffer (padding filled after blit)
    cv::Mat dst(dst_h, dst_w, CV_8UC3);

    // Ensure source is contiguous
    cv::Mat src_cont = src.isContinuous() ? src : src.clone();

    // Create RGA buffer for source
    rga_buffer_t src_buf = wrapbuffer_virtualaddr(
        (void*)src_cont.data,
        src_w, src_h,
        RK_FORMAT_BGR_888
    );

    // Create intermediate buffer for resized result
    cv::Mat resized(new_h, new_w, CV_8UC3);
    rga_buffer_t resize_buf = wrapbuffer_virtualaddr(
        (void*)resized.data,
        new_w, new_h,
        RK_FORMAT_BGR_888
    );

    // Step 1: RGA hardware resize
    IM_STATUS ret = imresize(src_buf, resize_buf);
    if (ret != IM_STATUS_SUCCESS) {
        LOGW("RGA imresize failed (", imStrError(ret), "), fallback to CPU");
        return letterboxCpu(src, target_size, info);
    }

    // Step 2: RGA blit resized image to center of padded destination
    int top = static_cast<int>(std::round(dy - 0.1f));
    int left = static_cast<int>(std::round(dx - 0.1f));
    top = std::max(top, 0);
    left = std::max(left, 0);

    if (left + new_w > dst_w || top + new_h > dst_h) {
        info = {0.f, 0.f, 0.f, 0, 0};
        return {};
    }

    // Create RGA buffer for destination with offset rect
    rga_buffer_t dst_buf = wrapbuffer_virtualaddr(
        (void*)dst.data,
        dst_w, dst_h,
        RK_FORMAT_BGR_888
    );

    // Define destination rect for the blit
    im_rect dst_rect = {left, top, new_w, new_h};
    im_rect src_rect = {0, 0, new_w, new_h};

    // RGA blit (copy with offset) - avoids CPU memcpy
    ret = improcess(resize_buf, dst_buf, {}, src_rect, dst_rect, {}, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS) {
        // Fallback to CPU copy if RGA blit fails
        LOGW("RGA improcess blit failed (", imStrError(ret), "), using CPU copy");
        cv::Rect roi(left, top, new_w, new_h);
        resized.copyTo(dst(roi));
    }

    const cv::Scalar pad_color(114, 114, 114);
    if (top > 0) {
        dst(cv::Rect(0, 0, dst_w, top)).setTo(pad_color);
    }
    const int bottom = top + new_h;
    if (bottom < dst_h) {
        dst(cv::Rect(0, bottom, dst_w, dst_h - bottom)).setTo(pad_color);
    }
    if (left > 0) {
        dst(cv::Rect(0, top, left, new_h)).setTo(pad_color);
    }
    const int right = left + new_w;
    if (right < dst_w) {
        dst(cv::Rect(right, top, dst_w - right, new_h)).setTo(pad_color);
    }

    return dst;
}

/**
 * @brief RGA-accelerated color conversion
 *
 * Supported: BGR<->RGB, YUV420SP->RGB/BGR
 * Performance: ~0.1ms vs ~0.5ms OpenCV
 */
cv::Mat Preprocess::convertColorRga(const cv::Mat& src, int code) {
    if (!initRga()) {
        return convertColorCpu(src, code);
    }

    // Map OpenCV code to RGA formats
    int src_format, dst_format;
    int color_mode = IM_COLOR_SPACE_DEFAULT;

    switch (code) {
        case cv::COLOR_BGR2RGB:
            src_format = RK_FORMAT_BGR_888;
            dst_format = RK_FORMAT_RGB_888;
            break;
        case cv::COLOR_RGB2BGR:
            src_format = RK_FORMAT_RGB_888;
            dst_format = RK_FORMAT_BGR_888;
            break;
        case cv::COLOR_YUV2RGB_NV12:
            src_format = RK_FORMAT_YCbCr_420_SP;
            dst_format = RK_FORMAT_RGB_888;
            color_mode = IM_YUV_TO_RGB_BT601_LIMIT;
            break;
        case cv::COLOR_YUV2BGR_NV12:
            src_format = RK_FORMAT_YCbCr_420_SP;
            dst_format = RK_FORMAT_BGR_888;
            color_mode = IM_YUV_TO_RGB_BT601_LIMIT;
            break;
        default:
            // Unsupported, fallback to OpenCV
            return convertColorCpu(src, code);
    }

    cv::Mat dst(src.rows, src.cols, CV_8UC3);
    cv::Mat src_cont = src.isContinuous() ? src : src.clone();

    rga_buffer_t src_buf = wrapbuffer_virtualaddr(
        (void*)src_cont.data, src.cols, src.rows, src_format);
    rga_buffer_t dst_buf = wrapbuffer_virtualaddr(
        (void*)dst.data, dst.cols, dst.rows, dst_format);

    IM_STATUS ret = imcvtcolor(src_buf, dst_buf, src_format, dst_format, color_mode);
    if (ret != IM_STATUS_SUCCESS) {
        LOGW("RGA imcvtcolor failed (", imStrError(ret), "), fallback to CPU");
        return convertColorCpu(src, code);
    }

    return dst;
}

#endif  // RKNN_USE_RGA

// ============================================================================
// OpenCV CPU Implementations
// ============================================================================

cv::Mat Preprocess::letterboxCpu(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info) {
    constexpr int kMinDim = 1;
    constexpr float kMinRatio = 1e-6f;
    constexpr float kPadEps = 0.1f;

    if (src.empty() || src.cols < kMinDim || src.rows < kMinDim ||
        target_size.width < kMinDim || target_size.height < kMinDim) {
        info = {0.f, 0.f, 0.f, 0, 0};
        return {};
    }

    int src_w = src.cols;
    int src_h = src.rows;
    int dst_w = target_size.width;
    int dst_h = target_size.height;

    float scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
    if (scale < kMinRatio) {
        info = {0.f, 0.f, 0.f, 0, 0};
        return {};
    }

    int new_w = static_cast<int>(std::round(src_w * scale));
    int new_h = static_cast<int>(std::round(src_h * scale));

    float dx = (dst_w - new_w) / 2.0f;
    float dy = (dst_h - new_h) / 2.0f;

    info.scale = scale;
    info.dx = dx;
    info.dy = dy;
    info.new_width = new_w;
    info.new_height = new_h;

    cv::Mat resized;
    if (scale != 1.0f) {
        cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = src;  // 浅拷贝，避免不必要的内存分配
    }

    cv::Mat dst(dst_h, dst_w, src.type(), cv::Scalar(114, 114, 114));

    int top = static_cast<int>(std::round(dy - kPadEps));
    int left = static_cast<int>(std::round(dx - kPadEps));
    top = std::max(top, 0);
    left = std::max(left, 0);

    cv::Rect roi(left, top, new_w, new_h);
    if (roi.x >= 0 && roi.y >= 0 &&
        roi.x + roi.width <= dst.cols &&
        roi.y + roi.height <= dst.rows) {
        resized.copyTo(dst(roi));
    } else {
        info = {0.f, 0.f, 0.f, 0, 0};
        return {};
    }

    return dst;
}

cv::Mat Preprocess::convertColorCpu(const cv::Mat& src, int code) {
    cv::Mat dst;
    cv::cvtColor(src, dst, code);
    return dst;
}

// ============================================================================
// Public Interface - Automatic Backend Selection
// ============================================================================

cv::Mat Preprocess::letterbox(const cv::Mat& src, int target_size, LetterboxInfo& info,
                               AccelBackend backend) {
    return letterbox(src, cv::Size(target_size, target_size), info, backend);
}

cv::Mat Preprocess::letterbox(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info,
                               AccelBackend backend) {
#if RKNN_USE_RGA
    switch (backend) {
        case AccelBackend::RGA:
            return letterboxRga(src, target_size, info);
        case AccelBackend::OPENCV:
            return letterboxCpu(src, target_size, info);
        case AccelBackend::AUTO:
        default:
            if (isRgaAvailable()) {
                return letterboxRga(src, target_size, info);
            }
            return letterboxCpu(src, target_size, info);
    }
#else
    (void)backend;
    return letterboxCpu(src, target_size, info);
#endif
}

cv::Mat Preprocess::convertColor(const cv::Mat& src, int code, AccelBackend backend) {
#if RKNN_USE_RGA
    switch (backend) {
        case AccelBackend::RGA:
            return convertColorRga(src, code);
        case AccelBackend::OPENCV:
            return convertColorCpu(src, code);
        case AccelBackend::AUTO:
        default:
            if (isRgaAvailable()) {
                // Only use RGA for supported conversions
                switch (code) {
                    case cv::COLOR_BGR2RGB:
                    case cv::COLOR_RGB2BGR:
                    case cv::COLOR_YUV2RGB_NV12:
                    case cv::COLOR_YUV2BGR_NV12:
                        return convertColorRga(src, code);
                    default:
                        break;
                }
            }
            return convertColorCpu(src, code);
    }
#else
    (void)backend;
    return convertColorCpu(src, code);
#endif
}

bool Preprocess::loadCalibration(const std::string& calibration_path,
                                 CameraCalibration& calibration) {
    calibration = {};
    if (calibration_path.empty()) {
        LOGW("Preprocess::loadCalibration: Empty calibration path");
        return false;
    }

    cv::FileStorage fs(calibration_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        LOGW("Preprocess::loadCalibration: Failed to open calibration file: ", calibration_path);
        return false;
    }

    auto readFirstNode = [&](const std::vector<std::string>& keys, cv::Mat& dst) -> bool {
        for (const auto& key : keys) {
            if (!fs[key].empty()) {
                fs[key] >> dst;
                if (!dst.empty()) {
                    return true;
                }
            }
        }
        return false;
    };

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    const bool has_camera = readFirstNode(
        {"camera_matrix", "cameraMatrix", "K", "intrinsic_matrix"}, camera_matrix);
    const bool has_dist = readFirstNode(
        {"dist_coeffs", "distCoeffs", "distortion_coefficients", "D"}, dist_coeffs);

    if (!has_camera || !has_dist) {
        LOGW("Preprocess::loadCalibration: Missing camera_matrix/dist_coeffs in ",
             calibration_path);
        return false;
    }

    camera_matrix.convertTo(camera_matrix, CV_64F);
    dist_coeffs.convertTo(dist_coeffs, CV_64F);

    if (camera_matrix.rows == 1 && camera_matrix.cols == 9) {
        camera_matrix = camera_matrix.reshape(1, 3);
    }
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3) {
        LOGW("Preprocess::loadCalibration: camera_matrix must be 3x3");
        return false;
    }

    if (dist_coeffs.rows > 1 && dist_coeffs.cols > 1) {
        dist_coeffs = dist_coeffs.reshape(1, 1);
    }
    if (dist_coeffs.rows != 1) {
        dist_coeffs = dist_coeffs.reshape(1, 1);
    }
    if (dist_coeffs.cols < 4) {
        LOGW("Preprocess::loadCalibration: dist_coeffs length must be >= 4");
        return false;
    }

    calibration.camera_matrix = camera_matrix;
    calibration.dist_coeffs = dist_coeffs;
    return calibration.isValid();
}

bool Preprocess::buildUndistortMaps(const CameraCalibration& calibration, cv::Size image_size,
                                    cv::Mat& map1, cv::Mat& map2) {
    map1.release();
    map2.release();

    if (!calibration.isValid()) {
        LOGW("Preprocess::buildUndistortMaps: Invalid camera calibration");
        return false;
    }
    if (image_size.width <= 0 || image_size.height <= 0) {
        LOGW("Preprocess::buildUndistortMaps: Invalid image size");
        return false;
    }

    try {
        cv::Mat new_camera = cv::getOptimalNewCameraMatrix(
            calibration.camera_matrix, calibration.dist_coeffs, image_size, 0.0, image_size);
        cv::initUndistortRectifyMap(
            calibration.camera_matrix, calibration.dist_coeffs, cv::Mat(), new_camera,
            image_size, CV_16SC2, map1, map2);
    } catch (const cv::Exception& e) {
        LOGW("Preprocess::buildUndistortMaps: OpenCV error: ", e.what());
        map1.release();
        map2.release();
        return false;
    }

    return !map1.empty() && !map2.empty();
}

cv::Mat Preprocess::undistort(const cv::Mat& src, const cv::Mat& map1, const cv::Mat& map2) {
    if (src.empty() || map1.empty() || map2.empty()) {
        return {};
    }

    cv::Mat dst;
    try {
        cv::remap(src, dst, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    } catch (const cv::Exception& e) {
        LOGW("Preprocess::undistort: OpenCV error: ", e.what());
        return {};
    }
    return dst;
}

cv::Mat Preprocess::ensureBgr8(const cv::Mat& src, AccelBackend backend,
                               FourChannelOrder four_channel_order) {
    if (src.empty()) {
        return {};
    }

    cv::Mat input = src;
    if (src.depth() != CV_8U) {
        src.convertTo(input, CV_8U);
    }

    if (input.type() == CV_8UC3) {
        return input;
    }
    if (input.type() == CV_8UC1) {
        return convertColor(input, cv::COLOR_GRAY2BGR, backend);
    }
    if (input.type() == CV_8UC4) {
        if (four_channel_order == FourChannelOrder::BGRA) {
            return convertColor(input, cv::COLOR_BGRA2BGR, backend);
        }
        if (four_channel_order == FourChannelOrder::RGBA) {
            return convertColor(input, cv::COLOR_RGBA2BGR, backend);
        }
        LOGW("Preprocess::ensureBgr8: Ambiguous 4-channel input. Specify BGRA or RGBA order.");
        return {};
    }

    LOGW("Preprocess::ensureBgr8: Unsupported input type: ", input.type());
    return {};
}

bool Preprocess::resolveRoiRect(cv::Size image_size, bool normalized_mode,
                                const cv::Rect2f& normalized_xywh,
                                const cv::Rect& pixel_xywh, bool clamp,
                                int min_size, cv::Rect& roi_out) {
    roi_out = cv::Rect();
    if (image_size.width <= 0 || image_size.height <= 0) {
        return false;
    }

    const int width = image_size.width;
    const int height = image_size.height;
    const int min_dim = std::max(1, min_size);

    if (width < min_dim || height < min_dim) {
        return false;
    }

    int x = 0;
    int y = 0;
    int roi_w = width;
    int roi_h = height;

    if (normalized_mode) {
        x = static_cast<int>(std::round(normalized_xywh.x * static_cast<float>(width)));
        y = static_cast<int>(std::round(normalized_xywh.y * static_cast<float>(height)));
        roi_w =
            static_cast<int>(std::round(normalized_xywh.width * static_cast<float>(width)));
        roi_h =
            static_cast<int>(std::round(normalized_xywh.height * static_cast<float>(height)));
    } else {
        x = pixel_xywh.x;
        y = pixel_xywh.y;
        roi_w = pixel_xywh.width;
        roi_h = pixel_xywh.height;
    }

    if (clamp) {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        roi_w = std::max(roi_w, min_dim);
        roi_h = std::max(roi_h, min_dim);

        if (x + roi_w > width) {
            roi_w = width - x;
        }
        if (y + roi_h > height) {
            roi_h = height - y;
        }
    }

    if (roi_w < min_dim || roi_h < min_dim) {
        return false;
    }
    if (x < 0 || y < 0 || x + roi_w > width || y + roi_h > height) {
        return false;
    }

    roi_out = cv::Rect(x, y, roi_w, roi_h);
    return roi_out.area() > 0;
}

cv::Mat Preprocess::cropRoi(const cv::Mat& src, const cv::Rect& roi) {
    if (src.empty()) {
        return {};
    }
    if (roi.width <= 0 || roi.height <= 0 || roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width > src.cols || roi.y + roi.height > src.rows) {
        return {};
    }
    if (roi.x == 0 && roi.y == 0 && roi.width == src.cols && roi.height == src.rows) {
        return src;
    }
    return src(roi).clone();
}

cv::Mat Preprocess::applyGammaLut(const cv::Mat& src, float gamma) {
    if (src.empty() || gamma <= 0.0f) {
        return {};
    }

    cv::Mat input = src;
    if (src.depth() != CV_8U) {
        src.convertTo(input, CV_8U);
    }

    if (std::fabs(gamma - 1.0f) < 1e-6f) {
        return input;
    }

    struct GammaLutCache {
        float gamma = std::numeric_limits<float>::quiet_NaN();
        cv::Mat lut;
    };
    thread_local GammaLutCache cache;

    if (cache.lut.empty() || std::fabs(cache.gamma - gamma) > 1e-6f) {
        cache.lut = cv::Mat(1, 256, CV_8U);
        for (int i = 0; i < 256; ++i) {
            const double normalized = static_cast<double>(i) / 255.0;
            const double corrected = std::pow(normalized, static_cast<double>(gamma));
            cache.lut.at<uint8_t>(0, i) =
                cv::saturate_cast<uint8_t>(std::round(corrected * 255.0));
        }
        cache.gamma = gamma;
    }

    cv::Mat dst;
    cv::LUT(input, cache.lut, dst);
    return dst;
}

namespace {

float percentileOfChannel(const cv::Mat& channel, float percentile) {
    if (channel.empty()) {
        return 0.0f;
    }

    cv::Mat flat = channel.reshape(1, 1);
    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_ASCENDING);

    const int count = sorted.cols;
    if (count <= 0) {
        return 0.0f;
    }

    const float clamped_percentile = std::clamp(percentile, 0.0f, 100.0f);
    const int idx = static_cast<int>(
        std::round((clamped_percentile / 100.0f) * static_cast<float>(count - 1)));
    return sorted.at<float>(0, idx);
}

}  // namespace

cv::Mat Preprocess::whiteBalanceGrayWorld(const cv::Mat& src, float clip_percent) {
    if (src.empty()) {
        return {};
    }

    cv::Mat input = src;
    if (src.depth() != CV_8U) {
        src.convertTo(input, CV_8U);
    }

    if (input.channels() != 3) {
        return input;
    }

    cv::Mat working;
    input.convertTo(working, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(working, channels);
    if (channels.size() != 3) {
        return input;
    }

    const float clipped = std::clamp(clip_percent, 0.0f, 49.0f);
    if (clipped > 0.0f) {
        const float each_side = clipped * 0.5f;
        for (auto& channel : channels) {
            const float lo = percentileOfChannel(channel, each_side);
            const float hi = percentileOfChannel(channel, 100.0f - each_side);
            cv::max(channel, lo, channel);
            cv::min(channel, hi, channel);
        }
    }

    const double mean_b = cv::mean(channels[0])[0];
    const double mean_g = cv::mean(channels[1])[0];
    const double mean_r = cv::mean(channels[2])[0];
    const double gray = (mean_b + mean_g + mean_r) / 3.0;
    if (gray <= std::numeric_limits<double>::epsilon()) {
        return input;
    }

    const double gain_b = std::clamp(gray / (mean_b + 1e-6), 0.2, 5.0);
    const double gain_g = std::clamp(gray / (mean_g + 1e-6), 0.2, 5.0);
    const double gain_r = std::clamp(gray / (mean_r + 1e-6), 0.2, 5.0);

    channels[0] *= gain_b;
    channels[1] *= gain_g;
    channels[2] *= gain_r;

    cv::Mat merged;
    cv::merge(channels, merged);
    cv::Mat dst;
    merged.convertTo(dst, CV_8U);
    return dst;
}

cv::Mat Preprocess::denoiseBilateral(const cv::Mat& src, int d, double sigma_color,
                                     double sigma_space) {
    if (src.empty()) {
        return {};
    }

    cv::Mat input = src;
    if (src.depth() != CV_8U) {
        src.convertTo(input, CV_8U);
    }
    if (input.channels() != 3) {
        return input;
    }

    cv::Mat dst;
    try {
        cv::bilateralFilter(input, dst, d, std::max(0.1, sigma_color),
                            std::max(0.1, sigma_space));
    } catch (const cv::Exception& e) {
        LOGW("Preprocess::denoiseBilateral: OpenCV error: ", e.what());
        return input;
    }
    return dst;
}

// ============================================================================
// Normalization & Format Conversion
// ============================================================================

cv::Mat Preprocess::normalize(const cv::Mat& src, float scale) {
    cv::Mat dst;
    src.convertTo(dst, CV_32F, scale);
    return dst;
}

cv::Mat Preprocess::hwc2chw(const cv::Mat& src) {
    // Validate input type - must be CV_32F (float)
    if (src.type() != CV_32FC3 && src.type() != CV_32FC1) {
        LOGE("Preprocess::hwc2chw: Expected CV_32F input, got type: ", src.type());
        return cv::Mat();  // Return empty on error
    }

    if (src.channels() == 1) {
        return src.clone().reshape(1, 1);
    }

    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    int h = src.rows;
    int w = src.cols;
    int c = src.channels();

    cv::Mat dst(1, c * h * w, CV_32F);

    for (int i = 0; i < c; ++i) {
        std::memcpy(dst.ptr<float>() + i * h * w,
                   channels[i].ptr<float>(),
                   h * w * sizeof(float));
    }

    return dst;
}

cv::Mat Preprocess::blob(const cv::Mat& src) {
    cv::Mat rgb = convertColor(src, cv::COLOR_BGR2RGB);
    cv::Mat norm = normalize(rgb);
    return hwc2chw(norm);
}

} // namespace rkapp::preprocess
