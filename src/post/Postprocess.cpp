#include "rkapp/post/Postprocess.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include "rkapp/common/log.hpp"

// OpenMP for parallel NMS
#if defined(_OPENMP)
#include <omp.h>
#define RKAPP_HAS_OPENMP 1
#endif

// ARM NEON for SIMD acceleration
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define RKAPP_HAS_NEON 1
#endif

namespace rkapp::post {

// ============================================================================
// NEON-optimized IOU Calculation
// ============================================================================

#if RKAPP_HAS_NEON
/**
 * @brief Calculate IOU for 4 boxes at once using NEON SIMD
 *
 * Processes 4 candidate boxes against 1 reference box in parallel.
 * ~4x throughput improvement over scalar code on ARM Cortex-A76.
 *
 * @param ref_x1,ref_y1,ref_x2,ref_y2 Reference box coordinates (scalar)
 * @param ref_area Reference box area (scalar)
 * @param cand_x1,cand_y1,cand_x2,cand_y2 4 candidate boxes (float32x4_t)
 * @param cand_area 4 candidate areas (float32x4_t)
 * @return float32x4_t containing 4 IOU values
 */
static inline float32x4_t calculateIOU_NEON(
    float ref_x1, float ref_y1, float ref_x2, float ref_y2, float ref_area,
    float32x4_t cand_x1, float32x4_t cand_y1,
    float32x4_t cand_x2, float32x4_t cand_y2,
    float32x4_t cand_area) {

    // Broadcast reference box to vectors
    float32x4_t r_x1 = vdupq_n_f32(ref_x1);
    float32x4_t r_y1 = vdupq_n_f32(ref_y1);
    float32x4_t r_x2 = vdupq_n_f32(ref_x2);
    float32x4_t r_y2 = vdupq_n_f32(ref_y2);
    float32x4_t r_area = vdupq_n_f32(ref_area);

    // Calculate intersection rectangle
    float32x4_t inter_x1 = vmaxq_f32(r_x1, cand_x1);
    float32x4_t inter_y1 = vmaxq_f32(r_y1, cand_y1);
    float32x4_t inter_x2 = vminq_f32(r_x2, cand_x2);
    float32x4_t inter_y2 = vminq_f32(r_y2, cand_y2);

    // Calculate intersection dimensions (clamp to 0)
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t inter_w = vmaxq_f32(zero, vsubq_f32(inter_x2, inter_x1));
    float32x4_t inter_h = vmaxq_f32(zero, vsubq_f32(inter_y2, inter_y1));

    // Intersection area
    float32x4_t inter_area = vmulq_f32(inter_w, inter_h);

    // Union area = a_area + b_area - intersection
    float32x4_t union_area = vsubq_f32(vaddq_f32(r_area, cand_area), inter_area);

    // Avoid division by zero
    float32x4_t eps = vdupq_n_f32(1e-6f);
    union_area = vmaxq_f32(union_area, eps);

    // IOU = intersection / union
    return vdivq_f32(inter_area, union_area);
}
#endif  // RKAPP_HAS_NEON

// ============================================================================
// Optimized NMS Implementation
// ============================================================================

/**
 * @brief Internal box representation for optimized NMS
 *
 * Pre-computes x1,y1,x2,y2 and area to avoid redundant calculations.
 * 32-byte aligned for optimal NEON loading.
 */
struct alignas(32) BoxInfo {
    float x1, y1, x2, y2;  // Pre-computed corners
    float area;            // Pre-computed area
    float confidence;
    int class_id;
    int original_idx;      // Index in filtered array
};

std::vector<rkapp::infer::Detection> Postprocess::nms(
    const std::vector<rkapp::infer::Detection>& detections,
    const NMSConfig& config) {

    std::vector<rkapp::infer::Detection> result;

    const float min_box = config.min_box_size > 0.0f ? config.min_box_size : 0.0f;
    const float max_box = config.max_box_size > 0.0f ? config.max_box_size : 0.0f;
    const float min_ar = config.min_aspect_ratio > 0.0f ? config.min_aspect_ratio : 0.0f;
    const float max_ar = config.max_aspect_ratio > 0.0f ? config.max_aspect_ratio : 0.0f;
    constexpr float kEps = 1e-6f;

    // ========== Stage 1: Filter and pre-compute ==========
    std::vector<BoxInfo> boxes;
    boxes.reserve(detections.size());

    for (size_t idx = 0; idx < detections.size(); ++idx) {
        const auto& det = detections[idx];

        // Confidence filter
        if (det.confidence < config.conf_thres) continue;

        float w = det.w;
        float h = det.h;

        // Size validation
        if (w <= 0.0f || h <= 0.0f) continue;
        if (min_box > 0.0f && (w < min_box || h < min_box)) continue;
        if (max_box > 0.0f && (w > max_box || h > max_box)) continue;

        // Aspect ratio filter
        if (min_ar > 0.0f || max_ar > 0.0f) {
            float ratio = h / std::max(w, kEps);
            if (min_ar > 0.0f && ratio < min_ar) continue;
            if (max_ar > 0.0f && ratio > max_ar) continue;
        }

        // Pre-compute box info
        BoxInfo box;
        box.x1 = det.x;
        box.y1 = det.y;
        box.x2 = det.x + det.w;
        box.y2 = det.y + det.h;
        box.area = det.w * det.h;
        box.confidence = det.confidence;
        box.class_id = det.class_id;
        box.original_idx = static_cast<int>(idx);
        boxes.push_back(box);
    }

    if (boxes.empty()) {
        return result;
    }

    // ========== Stage 2: Sort by confidence ==========
    std::sort(boxes.begin(), boxes.end(),
        [](const BoxInfo& a, const BoxInfo& b) {
            return a.confidence > b.confidence;
        });

    // Top-K prefilter
    if (config.topk > 0 && static_cast<int>(boxes.size()) > config.topk) {
        boxes.resize(config.topk);
    }

    const size_t n = boxes.size();
    if (config.max_det > 0) {
        result.reserve(std::min(n, static_cast<size_t>(config.max_det)));
    } else {
        result.reserve(n);
    }
    // Use atomic<bool> for thread-safe parallel NMS
    std::vector<std::atomic<bool>> suppressed(n);
    for (size_t i = 0; i < n; ++i) {
        suppressed[i].store(false, std::memory_order_relaxed);
    }

    // ========== Stage 3: NMS with optional parallelization ==========
    // For small N (<100), serial is faster due to parallel overhead
    // For large N (>=100), OpenMP provides significant speedup

#if RKAPP_HAS_NEON
    // Pre-extract coordinates into contiguous arrays for NEON
    std::vector<float> all_x1(n);
    std::vector<float> all_y1(n);
    std::vector<float> all_x2(n);
    std::vector<float> all_y2(n);
    std::vector<float> all_area(n);
    for (size_t i = 0; i < n; ++i) {
        all_x1[i] = boxes[i].x1;
        all_y1[i] = boxes[i].y1;
        all_x2[i] = boxes[i].x2;
        all_y2[i] = boxes[i].y2;
        all_area[i] = boxes[i].area;
    }
#endif

    for (size_t i = 0; i < n; ++i) {
        if (suppressed[i].load(std::memory_order_acquire)) continue;

        result.push_back(detections[boxes[i].original_idx]);

        if (config.max_det > 0 &&
            result.size() >= static_cast<size_t>(config.max_det)) {
            break;
        }

        const int ref_class = boxes[i].class_id;
        const float ref_x1 = boxes[i].x1;
        const float ref_y1 = boxes[i].y1;
        const float ref_x2 = boxes[i].x2;
        const float ref_y2 = boxes[i].y2;
        const float ref_area = boxes[i].area;

#if RKAPP_HAS_NEON && RKAPP_HAS_OPENMP
        // NEON + OpenMP path for large N.
        // 步进为 4，每次处理一个 NEON 组（4 个 box）。
        // 注意：OpenMP parallel for 不允许 break，所以 tail(<4) 的处理放在循环外。
        if (n - i > 100) {
            const float iou_thres = config.iou_thres;
            // 计算有多少个完整的 4-box 组
            const size_t aligned_end = i + 1 + ((n - i - 1) / 4) * 4;

            #pragma omp parallel for schedule(static, 4) if(n - i > 200)
            for (size_t j = i + 1; j < aligned_end; j += 4) {
                // Check class match for all 4
                bool all_same_class =
                    (boxes[j].class_id == ref_class) &&
                    (boxes[j+1].class_id == ref_class) &&
                    (boxes[j+2].class_id == ref_class) &&
                    (boxes[j+3].class_id == ref_class);

                if (all_same_class) {
                    // NEON path: 4 路并行 IoU
                    float32x4_t cand_x1 = vld1q_f32(&all_x1[j]);
                    float32x4_t cand_y1 = vld1q_f32(&all_y1[j]);
                    float32x4_t cand_x2 = vld1q_f32(&all_x2[j]);
                    float32x4_t cand_y2 = vld1q_f32(&all_y2[j]);
                    float32x4_t cand_area = vld1q_f32(&all_area[j]);

                    float32x4_t iou = calculateIOU_NEON(
                        ref_x1, ref_y1, ref_x2, ref_y2, ref_area,
                        cand_x1, cand_y1, cand_x2, cand_y2, cand_area);

                    float iou_vals[4];
                    vst1q_f32(iou_vals, iou);

                    for (int k = 0; k < 4; ++k) {
                        if (!suppressed[j+k].load(std::memory_order_acquire) && iou_vals[k] > iou_thres) {
                            suppressed[j+k].store(true, std::memory_order_release);
                        }
                    }
                } else {
                    // 混合类别，标量回退
                    for (size_t k = j; k < j + 4; ++k) {
                        if (!suppressed[k].load(std::memory_order_acquire) && boxes[k].class_id == ref_class) {
                            float iou = calculateIOU(
                                detections[boxes[i].original_idx],
                                detections[boxes[k].original_idx]);
                            if (iou > iou_thres) {
                                suppressed[k].store(true, std::memory_order_release);
                            }
                        }
                    }
                }
            }

            // 处理尾部（不足 4 个的剩余 box），串行执行
            for (size_t k = aligned_end; k < n; ++k) {
                if (!suppressed[k].load(std::memory_order_acquire) && boxes[k].class_id == ref_class) {
                    float iou = calculateIOU(
                        detections[boxes[i].original_idx],
                        detections[boxes[k].original_idx]);
                    if (iou > config.iou_thres) {
                        suppressed[k].store(true, std::memory_order_release);
                    }
                }
            }
        } else
#endif
        {
            // Scalar path (small N or no SIMD)
#if RKAPP_HAS_OPENMP
            #pragma omp parallel for schedule(dynamic) if(n - i > 100)
#endif
            for (size_t j = i + 1; j < n; ++j) {
                if (!suppressed[j].load(std::memory_order_acquire) && boxes[j].class_id == ref_class) {
                    float iou = calculateIOU(
                        detections[boxes[i].original_idx],
                        detections[boxes[j].original_idx]);
                    if (iou > config.iou_thres) {
                        suppressed[j].store(true, std::memory_order_release);
                    }
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Coordinate Rescaling
// ============================================================================

void Postprocess::rescaleDetections(
    std::vector<rkapp::infer::Detection>& detections,
    float scale_x, float scale_y, float dx, float dy) {

    if (std::fabs(scale_x) < 1e-6f || std::fabs(scale_y) < 1e-6f) {
        LOGE("Postprocess::rescaleDetections: invalid scale factors: ", scale_x, ", ", scale_y);
        return;
    }

    const float inv_scale_x = 1.0f / scale_x;
    const float inv_scale_y = 1.0f / scale_y;

#if RKAPP_HAS_OPENMP
    #pragma omp parallel for if(detections.size() > 50)
#endif
    for (size_t i = 0; i < detections.size(); ++i) {
        auto& det = detections[i];
        det.x = (det.x - dx) * inv_scale_x;
        det.y = (det.y - dy) * inv_scale_y;
        det.w = det.w * inv_scale_x;
        det.h = det.h * inv_scale_y;
    }
}

// ============================================================================
// Class Name Mapping
// ============================================================================

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
        LOGE("Failed to open class names file: ", path);
        return class_names;
    }

    std::string line;
    while (std::getline(file, line)) {
        const auto pos = line.find_last_not_of(" \t\r\n");
        if (pos == std::string::npos) continue;
        line.erase(pos + 1);
        if (!line.empty()) class_names.push_back(line);
    }

    LOGI("Loaded ", class_names.size(), " class names from ", path);
    return class_names;
}

// ============================================================================
// IOU Calculation (Scalar Version)
// ============================================================================

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
