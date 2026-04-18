#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "rkapp/infer/ModelSpec.hpp"

namespace rkapp::infer {

struct Keypoint {
    float x = 0.0f;          // 原图坐标系 x
    float y = 0.0f;          // 原图坐标系 y
    float visibility = 0.0f; // 关键点可见性/置信度 [0,1]
};

struct Detection {
    float x, y, w, h;          // 检测框（原图坐标系）
    float confidence;          // 置信度 [0,1]
    int class_id;              // 类别索引
    std::string class_name;    // 类别名（可选）
    std::vector<Keypoint> keypoints; // 为空表示纯检测，非空表示姿态结果
};

struct DecodeParams {
    float conf_thres = 0.25f;
    float iou_thres = 0.45f;
    int max_boxes = 0;          // <=0 表示不限制数量
};

// 推理引擎统一接口：RKNN/ONNX 等后端都遵循该契约。
class IInferEngine {
public:
    virtual ~IInferEngine() = default;
    
    // 初始化模型与运行时。
    virtual bool init(const ModelSpec& model_spec) = 0;
    // 输入 BGR 图像，返回检测结果（原图坐标）。
    virtual std::vector<Detection> infer(const cv::Mat& image) = 0;
    // 预热模型，降低首帧抖动。
    virtual void warmup() = 0;
    // 释放后端资源。
    virtual void release() = 0;
    
    // 获取模型输入尺寸。
    virtual int getInputWidth() const = 0;
    virtual int getInputHeight() const = 0;

    // 动态更新解码/NMS参数（默认空实现，按后端可选覆盖）。
    virtual void setDecodeParams(const DecodeParams& params) { (void)params; }
};

using InferEnginePtr = std::unique_ptr<IInferEngine>;

} // namespace rkapp::infer
