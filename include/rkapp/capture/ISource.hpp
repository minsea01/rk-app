#pragma once

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

struct CaptureFrame {
    cv::Mat mat;
    std::shared_ptr<void> owner;
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
        cv::Mat tmp;
        if (!read(tmp)) return false;
        frame.mat = tmp;
        frame.owner.reset();
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
