#include "rkapp/post/Postprocess.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include "log.hpp"

namespace rkapp::post {

std::vector<rkapp::infer::Detection> Postprocess::nms(
    const std::vector<rkapp::infer::Detection>& detections,
    const NMSConfig& config) {
    
    std::vector<rkapp::infer::Detection> result;
    
    const float min_box = config.min_box_size > 0.0f ? config.min_box_size : 0.0f;
    const float max_box = config.max_box_size > 0.0f ? config.max_box_size : 0.0f;
    const float min_ar = config.min_aspect_ratio > 0.0f ? config.min_aspect_ratio : 0.0f;
    const float max_ar = config.max_aspect_ratio > 0.0f ? config.max_aspect_ratio : 0.0f;
    constexpr float kEps = 1e-6f;

    // Filter by confidence / geometry
    std::vector<rkapp::infer::Detection> filtered;
    for (const auto& det : detections) {
        if (det.confidence >= config.conf_thres) {
            float w = det.w;
            float h = det.h;
            if (w <= 0.0f || h <= 0.0f) {
                continue;
            }
            if (min_box > 0.0f && (w < min_box || h < min_box)) {
                continue;
            }
            if (max_box > 0.0f && (w > max_box || h > max_box)) {
                continue;
            }
            if (min_ar > 0.0f || max_ar > 0.0f) {
                float ratio = h / std::max(w, kEps);
                if (min_ar > 0.0f && ratio < min_ar) {
                    continue;
                }
                if (max_ar > 0.0f && ratio > max_ar) {
                    continue;
                }
            }
            filtered.push_back(det);
        }
    }
    
    // Sort by confidence (descending)
    std::sort(filtered.begin(), filtered.end(),
        [](const rkapp::infer::Detection& a, const rkapp::infer::Detection& b) {
            return a.confidence > b.confidence;
        });

    // Top-K prefilter to减小NMS开销
    if (config.topk > 0 && static_cast<int>(filtered.size()) > config.topk) {
        filtered.resize(config.topk);
    }
    
    std::vector<bool> suppressed(filtered.size(), false);
    
    for (size_t i = 0; i < filtered.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(filtered[i]);
        if (config.max_det > 0 &&
            result.size() >= static_cast<size_t>(config.max_det)) {
            break;
        }
        
        // Suppress overlapping boxes
        for (size_t j = i + 1; j < filtered.size(); ++j) {
            if (suppressed[j]) continue;
            
            if (filtered[i].class_id == filtered[j].class_id) {
                float iou = calculateIOU(filtered[i], filtered[j]);
                if (iou > config.iou_thres) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return result;
}

void Postprocess::rescaleDetections(
    std::vector<rkapp::infer::Detection>& detections,
    float scale_x, float scale_y, float dx, float dy) {
    if (std::fabs(scale_x) < 1e-6f || std::fabs(scale_y) < 1e-6f) {
        LOGE("Postprocess::rescaleDetections: invalid scale factors: ", scale_x, ", ", scale_y);
        return;
    }

    const float safe_scale_x = scale_x;
    const float safe_scale_y = scale_y;

    for (auto& det : detections) {
        // Rescale from input image coordinates to original image coordinates
        det.x = (det.x - dx) / safe_scale_x;
        det.y = (det.y - dy) / safe_scale_y;
        det.w = det.w / safe_scale_x;
        det.h = det.h / safe_scale_y;
    }
}

void Postprocess::mapClassNames(
    std::vector<rkapp::infer::Detection>& detections,
    const std::vector<std::string>& class_names) {
    
    for (auto& det : detections) {
        if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names.size())) {
            det.class_name = class_names[det.class_id];
        } else {
            det.class_name = "unknown";
        }
    }
}

std::vector<std::string> Postprocess::loadClassNames(const std::string& path) {
    std::vector<std::string> class_names;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open class names file: " << path << std::endl;
        return class_names;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Remove trailing whitespace; skip empty/whitespace-only lines safely
        const auto pos = line.find_last_not_of(" \t\r\n");
        if (pos == std::string::npos) continue;
        line.erase(pos + 1);
        if (!line.empty()) class_names.push_back(line);
    }
    
    std::cout << "Loaded " << class_names.size() << " class names from " << path << std::endl;
    return class_names;
}

float Postprocess::calculateIOU(const rkapp::infer::Detection& a, const rkapp::infer::Detection& b) {
    // Convert to x1, y1, x2, y2 format
    float a_x1 = a.x;
    float a_y1 = a.y;
    float a_x2 = a.x + a.w;
    float a_y2 = a.y + a.h;
    
    float b_x1 = b.x;
    float b_y1 = b.y;
    float b_x2 = b.x + b.w;
    float b_y2 = b.y + b.h;
    
    // Calculate intersection
    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);
    
    if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2) {
        return 0.0f;
    }
    
    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    float a_area = a.w * a.h;
    float b_area = b.w * b.h;
    float union_area = a_area + b_area - inter_area;
    
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

} // namespace rkapp::post
