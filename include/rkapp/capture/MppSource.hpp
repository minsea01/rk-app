#pragma once

#include "rkapp/capture/ISource.hpp"
#include <memory>
#include <string>
#include <atomic>

namespace rkapp::capture {

/**
 * @brief MPP (Media Process Platform) hardware-accelerated video source
 *
 * Uses RK3588 VPU for hardware video decoding with the following benefits:
 * - H.264/H.265/VP9 decoding at up to 8K@60fps
 * - ~4ms decode latency vs ~25ms OpenCV software decode
 * - DMA-BUF output for zero-copy to RGA/NPU
 * - Low CPU usage (~5% vs ~30% software decode)
 *
 * Supported inputs:
 * - Video files (MP4, MKV, AVI with H.264/H.265)
 * - RTSP streams (rtsp://...)
 *
 * @note Requires librockchip_mpp.so and appropriate kernel drivers
 *
 * Usage:
 * @code
 *   MppSource src;
 *   src.open("video.mp4");  // or "rtsp://..." or "/dev/video0"
 *   cv::Mat frame;
 *   while (src.read(frame)) {
 *       // Process frame (BGR format)
 *   }
 * @endcode
 */
class MppSource : public ISource {
public:
    MppSource();
    ~MppSource() override;

    // Disable copy (MPP context is not copyable)
    MppSource(const MppSource&) = delete;
    MppSource& operator=(const MppSource&) = delete;

    // Disable move (std::atomic members cannot be moved)
    MppSource(MppSource&&) = delete;
    MppSource& operator=(MppSource&&) = delete;

    // ========== ISource Interface ==========

    /**
     * @brief Open video source with MPP hardware decoding
     *
     * @param uri Video file path or RTSP URL
     * @return true if successfully opened
     *
     * @note For RTSP streams, uses FFmpeg for demuxing + MPP for decoding
     */
    bool open(const std::string& uri) override;

    /**
     * @brief Read next frame with hardware decoding
     *
     * @param frame Output BGR image (converted from YUV via RGA if available)
     * @return true if frame was successfully read
     *
     * @note Output format is always BGR (cv::Mat CV_8UC3) for compatibility
     */
    bool read(cv::Mat& frame) override;

    void release() override;
    bool isOpened() const override;

    double getFPS() const override;
    cv::Size getSize() const override;
    int getTotalFrames() const override;
    int getCurrentFrame() const override;

    SourceType getType() const override;

    // ========== MPP-specific Methods ==========

    /**
     * @brief Check if MPP hardware decoding is available
     * @return true if librockchip_mpp is loaded and VPU is accessible
     */
    static bool isMppAvailable();

    /**
     * @brief Get decode latency statistics
     * @return Average decode time in milliseconds
     */
    double getDecodeLatencyMs() const;

    /**
     * @brief Enable/disable DMA-BUF output mode
     *
     * When enabled, frames are kept in DMA-BUF memory for zero-copy
     * transfer to RGA/NPU. Call getDmaBufFd() to get the file descriptor.
     *
     * @param enable true to enable DMA-BUF mode
     */
    void setDmaBufMode(bool enable);

    /**
     * @brief Get DMA-BUF file descriptor for current frame
     *
     * Only valid when DMA-BUF mode is enabled and after read() returns true.
     *
     * @return Duplicated DMA-BUF fd, or -1 if not available (caller must close)
     */
    int getDmaBufFd() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::string uri_;
    std::atomic<bool> is_opened_{false};
    std::atomic<int> current_frame_{0};

    double fps_ = 0.0;
    int width_ = 0;
    int height_ = 0;
    int total_frames_ = 0;
    bool dma_buf_mode_ = false;
};

} // namespace rkapp::capture
