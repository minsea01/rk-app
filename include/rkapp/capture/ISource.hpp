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

// 统一采集接口：上层流水线只依赖该抽象，不关心底层来源实现。
class ISource {
public:
    virtual ~ISource() = default;
    
    // 打开输入源（路径、URL、设备节点等）。
    virtual bool open(const std::string& uri) = 0;
    // 读取一帧到普通 cv::Mat（通用路径）。
    virtual bool read(cv::Mat& frame) = 0;
    // 读取一帧并可携带底层所有权（零拷贝路径可复用 owner）。
    virtual bool readFrame(CaptureFrame& frame) {
        frame.reset();
        cv::Mat tmp;
        if (!read(tmp)) return false;
        frame.setMatFrame(std::move(tmp), PixelFormat::BGR888);
        return true;
    }
    virtual ReadStatus readFrameEx(CaptureFrame& frame) {
        if (readFrame(frame)) {
            return ReadStatus::FrameReady;
        }
        return ReadStatus::EndOfStream;
    }
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
