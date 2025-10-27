#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace rkapp::infer {

struct Detection {
    float x, y, w, h;       // bbox in original image coordinates
    float confidence;       // confidence score [0-1]
    int class_id;          // class index
    std::string class_name; // class name
};

struct DecodeParams {
    float conf_thres = 0.25f;
    float iou_thres = 0.45f;
    int max_boxes = 0;          // 0 or negative means unlimited
};

class IInferEngine {
public:
    virtual ~IInferEngine() = default;
    
    virtual bool init(const std::string& model_path, int img_size = 640) = 0;
    virtual std::vector<Detection> infer(const cv::Mat& image) = 0;
    virtual void warmup() = 0;
    virtual void release() = 0;
    
    virtual int getInputWidth() const = 0;
    virtual int getInputHeight() const = 0;

    virtual void setDecodeParams(const DecodeParams& params) { (void)params; }
};

using InferEnginePtr = std::unique_ptr<IInferEngine>;

} // namespace rkapp::infer
