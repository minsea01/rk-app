 #pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace rkapp::preprocess {

struct LetterboxInfo {
    float scale;
    float dx, dy;
    int new_width, new_height;
};

struct CameraCalibration {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;

    bool isValid() const {
        return !camera_matrix.empty() && camera_matrix.rows == 3 &&
               camera_matrix.cols == 3 && !dist_coeffs.empty();
    }
};

/**
 * @brief Hardware acceleration backend selection
 */
enum class AccelBackend {
    AUTO,     // Auto-select best available (RGA > OpenCV)
    RGA,      // Force RGA (fails if unavailable)
    OPENCV    // Force OpenCV CPU
};

enum class FourChannelOrder {
    UNKNOWN,  // Ambiguous 4-channel input order
    BGRA,     // Byte layout: B,G,R,A
    RGBA      // Byte layout: R,G,B,A
};

class Preprocess {
public:
    // ========== Core Letterbox Functions ==========

    /**
     * @brief Letterbox resize with automatic backend selection
     *
     * On RK3588 with RGA available: Uses hardware acceleration (~0.3ms)
     * Fallback: Uses OpenCV CPU implementation (~3ms)
     *
     * @param src Input BGR image
     * @param target_size Square target size (e.g., 640)
     * @param info Output letterbox parameters for coordinate mapping
     * @param backend Acceleration backend (default: AUTO)
     * @return Letterboxed image
     */
    static cv::Mat letterbox(const cv::Mat& src, int target_size, LetterboxInfo& info,
                             AccelBackend backend = AccelBackend::AUTO);
    static cv::Mat letterbox(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info,
                             AccelBackend backend = AccelBackend::AUTO);

    // ========== Color Conversion ==========

    /**
     * @brief Color space conversion with optional RGA acceleration
     *
     * RGA-accelerated conversions: BGR<->RGB, YUV420->RGB
     * OpenCV fallback for other conversions
     */
    static cv::Mat convertColor(const cv::Mat& src, int code = cv::COLOR_BGR2RGB,
                                AccelBackend backend = AccelBackend::AUTO);

    // ========== Camera Calibration / Undistort ==========

    static bool loadCalibration(const std::string& calibration_path, CameraCalibration& calibration);
    static bool buildUndistortMaps(const CameraCalibration& calibration, cv::Size image_size,
                                   cv::Mat& map1, cv::Mat& map2);
    static cv::Mat undistort(const cv::Mat& src, const cv::Mat& map1, const cv::Mat& map2);

    // ========== Input Canonicalization ==========

    static cv::Mat ensureBgr8(const cv::Mat& src, AccelBackend backend = AccelBackend::AUTO,
                              FourChannelOrder four_channel_order = FourChannelOrder::UNKNOWN);

    // ========== ROI / Image Enhancement ==========

    static bool resolveRoiRect(cv::Size image_size, bool normalized_mode,
                               const cv::Rect2f& normalized_xywh, const cv::Rect& pixel_xywh,
                               bool clamp, int min_size, cv::Rect& roi_out);
    static cv::Mat cropRoi(const cv::Mat& src, const cv::Rect& roi);
    static cv::Mat applyGammaLut(const cv::Mat& src, float gamma);
    static cv::Mat whiteBalanceGrayWorld(const cv::Mat& src, float clip_percent = 0.0f);
    static cv::Mat denoiseBilateral(const cv::Mat& src, int d = 5, double sigma_color = 35.0,
                                    double sigma_space = 35.0);

    // ========== Normalization & Format Conversion ==========

    static cv::Mat normalize(const cv::Mat& src, float scale = 1.0f/255.0f);
    static cv::Mat hwc2chw(const cv::Mat& src);
    static cv::Mat blob(const cv::Mat& src);

    // ========== RGA Hardware Acceleration ==========

#if RKNN_USE_RGA
    /**
     * @brief Check if RGA hardware is available
     * @return true if RGA is available and initialized
     */
    static bool isRgaAvailable();

    /**
     * @brief RGA-accelerated letterbox (resize + pad)
     *
     * Performance: ~0.3ms for 1080p -> 640x640 (vs ~3ms OpenCV)
     * Uses im2d API: imresize + imfill for padding
     *
     * @note Only available when compiled with RKNN_USE_RGA=1
     */
    static cv::Mat letterboxRga(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info);

    /**
     * @brief RGA-accelerated color conversion
     *
     * Supported: BGR->RGB, RGB->BGR, YUV420SP->RGB, YUV420SP->BGR
     */
    static cv::Mat convertColorRga(const cv::Mat& src, int code);
#endif

private:
    // OpenCV CPU implementations (always available)
    static cv::Mat letterboxCpu(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info);
    static cv::Mat convertColorCpu(const cv::Mat& src, int code);

#if RKNN_USE_RGA
    static bool rga_initialized_;
    static bool initRga();
#endif
};

} // namespace rkapp::preprocess
