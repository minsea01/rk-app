#pragma once

#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

namespace rkapp::capture {

enum class SourceType {
    FOLDER,
    VIDEO,
    RTSP,
    GIGE,
    MPP     // Hardware-accelerated video via RK3588 MPP
};

struct CaptureFrame {
    cv::Mat mat;
    std::shared_ptr<void> owner;
};

class ISource {
public:
    virtual ~ISource() = default;
    
    virtual bool open(const std::string& uri) = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual bool readFrame(CaptureFrame& frame) {
        cv::Mat tmp;
        if (!read(tmp)) return false;
        frame.mat = tmp;
        frame.owner.reset();
        return true;
    }
    virtual void release() = 0;
    virtual bool isOpened() const = 0;
    
    virtual double getFPS() const = 0;
    virtual cv::Size getSize() const = 0;
    virtual int getTotalFrames() const = 0;
    virtual int getCurrentFrame() const = 0;
    
    virtual SourceType getType() const = 0;
};

using SourcePtr = std::unique_ptr<ISource>;

} // namespace rkapp::capture
