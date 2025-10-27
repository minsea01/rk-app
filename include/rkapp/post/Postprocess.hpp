#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "rkapp/infer/IInferEngine.hpp"

namespace rkapp::post {

struct NMSConfig {
    float conf_thres = 0.25f;
    float iou_thres = 0.60f;
    int max_det = 1000;
    int topk = 1000; // 先按置信度截取 Top-K 再做 NMS
    float min_box_size = 0.0f;     // 最小宽高（像素）
    float max_box_size = 0.0f;     // 最大宽高（像素），0 表示不限
    float min_aspect_ratio = 0.0f; // H/W 下界，0 表示不限
    float max_aspect_ratio = 0.0f; // H/W 上界，0 表示不限
};

class Postprocess {
public:
    static std::vector<rkapp::infer::Detection> nms(
        const std::vector<rkapp::infer::Detection>& detections,
        const NMSConfig& config = {});
    
    static void rescaleDetections(
        std::vector<rkapp::infer::Detection>& detections,
        float scale_x, float scale_y, float dx, float dy);
    
    static void mapClassNames(
        std::vector<rkapp::infer::Detection>& detections,
        const std::vector<std::string>& class_names);
    
    static std::vector<std::string> loadClassNames(const std::string& path);
    
private:
    static float calculateIOU(const rkapp::infer::Detection& a, const rkapp::infer::Detection& b);
};

} // namespace rkapp::post
