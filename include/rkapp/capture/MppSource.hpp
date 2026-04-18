#pragma once

#include "rkapp/capture/ISource.hpp"
#include <memory>
#include <string>
#include <atomic>

namespace rkapp::capture {

/**
 * @brief MPP（Media Process Platform）硬件解码输入源
 *
 * 通过 RK3588 VPU 执行硬件解码，典型收益：
 * - 支持 H.264/H.265/VP9 等编码
 * - 解码延迟和 CPU 占用通常低于纯软件解码
 * - 可输出 DMA-BUF 以衔接零拷贝链路（RGA/NPU）
 *
 * 支持输入：
 * - 视频文件（MP4/MKV/AVI 等）
 * - RTSP 流（rtsp://...）
 *
 * @note 需要系统提供 `librockchip_mpp.so` 及对应内核驱动
 *
 * 使用示例：
 * @code
 *   MppSource src;
 *   src.open("video.mp4");
 *   cv::Mat frame;
 *   while (src.read(frame)) {
 *       // 处理 BGR 帧
 *   }
 * @endcode
 */
class MppSource : public ISource {
public:
    MppSource();
    ~MppSource() override;

    // 禁止拷贝（MPP 上下文不可安全复制）。
    MppSource(const MppSource&) = delete;
    MppSource& operator=(const MppSource&) = delete;

    // 禁止移动（包含原子成员与底层句柄状态）。
    MppSource(MppSource&&) = delete;
    MppSource& operator=(MppSource&&) = delete;

    // ========== ISource 接口 ==========

    /**
     * @brief 打开输入源并初始化 MPP 解码链路
     *
     * @param uri 视频文件路径或 RTSP URL
     * @return 是否打开成功
     *
     * @note RTSP/容器拆包由 FFmpeg 完成，MPP 负责纯解码
     */
    bool open(const std::string& uri) override;

    /**
     * @brief 读取下一帧（硬件解码）
     *
     * @param frame 输出 BGR 图像（必要时做 YUV->BGR 转换）
     * @return 读取是否成功
     *
     * @note 对外统一返回 BGR（`cv::Mat CV_8UC3`）
     */
    bool read(cv::Mat& frame) override;
    ReadStatus readFrameEx(CaptureFrame& frame) override;

    void release() override;
    bool isOpened() const override;

    double getFPS() const override;
    cv::Size getSize() const override;
    int getTotalFrames() const override;
    int getCurrentFrame() const override;

    SourceType getType() const override;

    // ========== MPP 扩展方法 ==========

    /**
     * @brief 检查 MPP 硬解码能力是否可用
     */
    static bool isMppAvailable();

    /**
     * @brief 获取平均解码时延（毫秒）
     */
    double getDecodeLatencyMs() const;

    /**
     * @brief 启用/关闭 DMA-BUF 输出模式
     *
     * 启用后，当前帧可通过 `getDmaBufFd()` 导出 fd 用于零拷贝传递。
     */
    void setDmaBufMode(bool enable);

    /**
     * @brief 获取当前帧 DMA-BUF 文件描述符
     *
     * 仅在 DMA-BUF 模式且 `read()` 成功后有效。
     * 返回的是 `dup` 后 fd，调用方负责 `close`。
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
