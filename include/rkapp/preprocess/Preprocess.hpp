#pragma once

#include <opencv2/opencv.hpp>
#include <string>

#include "rkapp/preprocess/LetterboxInfo.hpp"

namespace rkapp::preprocess {

struct CameraCalibration {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;

    bool isValid() const {
        return !camera_matrix.empty() && camera_matrix.rows == 3 &&
               camera_matrix.cols == 3 && !dist_coeffs.empty();
    }
};

/**
 * @brief 预处理加速后端选择
 */
enum class AccelBackend {
    AUTO,     // 自动选择（优先 RGA，否则 OpenCV）
    RGA,      // 强制使用 RGA（不可用时失败）
    OPENCV    // 强制使用 OpenCV CPU
};

enum class FourChannelOrder {
    UNKNOWN,  // 四通道顺序未知
    BGRA,     // 字节顺序：B,G,R,A
    RGBA      // 字节顺序：R,G,B,A
};

// 预处理工具集合：统一封装 letterbox、颜色转换、增强与格式整理。
class Preprocess {
public:
    // ========== 核心 Letterbox ==========

    /**
     * @brief Letterbox 缩放+补边（自动后端）
     *
     * 在 RK3588 且 RGA 可用时优先硬件加速，否则回退到 OpenCV CPU。
     */
    static cv::Mat letterbox(const cv::Mat& src, int target_size, LetterboxInfo& info,
                             AccelBackend backend = AccelBackend::AUTO);
    static cv::Mat letterbox(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info,
                             AccelBackend backend = AccelBackend::AUTO);

    // ========== 颜色空间转换 ==========

    /**
     * @brief 颜色转换（可选 RGA 加速）
     */
    static cv::Mat convertColor(const cv::Mat& src, int code = cv::COLOR_BGR2RGB,
                                AccelBackend backend = AccelBackend::AUTO);
    static cv::Mat convertYuv420spToBgr(const cv::Mat& src, cv::Size image_size,
                                        bool nv21 = false,
                                        AccelBackend backend = AccelBackend::AUTO);

    // ========== 相机标定/去畸变 ==========

    static bool loadCalibration(const std::string& calibration_path, CameraCalibration& calibration);
    static bool buildUndistortMaps(const CameraCalibration& calibration, cv::Size image_size,
                                   cv::Mat& map1, cv::Mat& map2);
    static cv::Mat undistort(const cv::Mat& src, const cv::Mat& map1, const cv::Mat& map2);

    // ========== 输入规范化 ==========

    static cv::Mat ensureBgr8(const cv::Mat& src, AccelBackend backend = AccelBackend::AUTO,
                              FourChannelOrder four_channel_order = FourChannelOrder::UNKNOWN);

    // ========== ROI 与图像增强 ==========

    static bool resolveRoiRect(cv::Size image_size, bool normalized_mode,
                               const cv::Rect2f& normalized_xywh, const cv::Rect& pixel_xywh,
                               bool clamp, int min_size, cv::Rect& roi_out);
    static cv::Mat cropRoi(const cv::Mat& src, const cv::Rect& roi);
    static cv::Mat applyGammaLut(const cv::Mat& src, float gamma);
    static cv::Mat whiteBalanceGrayWorld(const cv::Mat& src, float clip_percent = 0.0f);
    static cv::Mat denoiseBilateral(const cv::Mat& src, int d = 5, double sigma_color = 35.0,
                                    double sigma_space = 35.0);

    // ========== 归一化与格式转换 ==========

    static cv::Mat normalize(const cv::Mat& src, float scale = 1.0f/255.0f);
    static cv::Mat hwc2chw(const cv::Mat& src);
    static cv::Mat blob(const cv::Mat& src);

    // ========== RGA 硬件加速 ==========

#if RKNN_USE_RGA
    /**
     * @brief 检查 RGA 是否可用
     */
    static bool isRgaAvailable();

    /**
     * @brief RGA 加速 letterbox（缩放+补边）
     */
    static cv::Mat letterboxRga(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info);

    /**
     * @brief RGA 加速颜色转换
     */
    static cv::Mat convertColorRga(const cv::Mat& src, int code);
#endif

private:
    // OpenCV CPU 参考实现（始终可用）。
    static cv::Mat letterboxCpu(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info);
    static cv::Mat convertColorCpu(const cv::Mat& src, int code);

#if RKNN_USE_RGA
    static bool rga_initialized_;
    static bool initRga();
#endif
};

} // namespace rkapp::preprocess
