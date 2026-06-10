#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

namespace rkapp::capture {

// 输入源类型：覆盖离线文件、网络流与板端硬件采集。
enum class SourceType {
    FOLDER,
    VIDEO,
    RTSP,
    GIGE,
    MPP,    // RK3588 MPP 硬解码视频源
    CSI     // 通过 GStreamer v4l2src 采集的 MIPI CSI 相机
};

enum class PixelFormat {
    UNKNOWN,
    BGR888,
    RGB888,
    BGRA8888,
    RGBA8888,
    GRAY8,
    BAYER_RG8,
    BAYER_BG8,
    BAYER_GR8,
    BAYER_GB8,
    NV12,
    NV21,
};

enum class StorageKind {
    CPU_MAT,
    SHARED_MAPPED,
    DMA_BUF,
};

struct CaptureFrame {
    cv::Mat mat;
    std::shared_ptr<void> owner;
    PixelFormat pixel_format = PixelFormat::UNKNOWN;
    StorageKind storage_kind = StorageKind::CPU_MAT;
    cv::Size image_size{0, 0};
    size_t row_stride = 0;
    int dma_fd = -1;

    void reset() {
        mat.release();
        owner.reset();
        pixel_format = PixelFormat::UNKNOWN;
        storage_kind = StorageKind::CPU_MAT;
        image_size = {0, 0};
        row_stride = 0;
        dma_fd = -1;
    }

    void setMatFrame(cv::Mat frame_mat,
                     PixelFormat format = PixelFormat::BGR888,
                     StorageKind storage = StorageKind::CPU_MAT,
                     std::shared_ptr<void> frame_owner = nullptr,
                     cv::Size actual_size = {}) {
        mat = std::move(frame_mat);
        owner = std::move(frame_owner);
        pixel_format = format;
        storage_kind = storage;
        image_size = (actual_size.width > 0 && actual_size.height > 0) ? actual_size : mat.size();
        row_stride = mat.empty() ? 0 : mat.step;
        dma_fd = -1;
    }

    bool empty() const { return mat.empty(); }
};

enum class ReadStatus {
    FrameReady,
    EndOfStream,
    RecoverableError,
    FatalError,
};

// 统一采集接口：上层流水线只依赖 readFrameEx 这一个读取契约。
// 历史上的 read(cv::Mat&)/readFrame(CaptureFrame&) 不再属于接口；
// 具体源可按需提供同名便捷方法，但多态调用方一律使用 readFrameEx。
class ISource {
public:
    virtual ~ISource() = default;

    // 打开输入源（路径、URL、设备节点等）。
    virtual bool open(const std::string& uri) = 0;
    // 读取一帧（可携带像素格式/stride/底层所有权元数据）。
    // 实现方必须区分四种状态：FrameReady / EndOfStream / RecoverableError / FatalError。
    // 直播类源应把瞬时失败映射为 RecoverableError，上层的重连与退避策略依赖该语义；
    // 把失败一律报成 EndOfStream 会使流水线直接停机。
    virtual ReadStatus readFrameEx(CaptureFrame& frame) = 0;
    virtual void release() = 0;
    virtual bool isOpened() const = 0;
    
    // 基础流信息：供统计与业务控制使用。
    virtual double getFPS() const = 0;
    virtual cv::Size getSize() const = 0;
    virtual int getTotalFrames() const = 0;
    virtual int getCurrentFrame() const = 0;
    
    virtual SourceType getType() const = 0;
};

using SourcePtr = std::unique_ptr<ISource>;

} // namespace rkapp::capture
