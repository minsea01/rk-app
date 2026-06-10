#pragma once

#include <cstdint>

namespace rkapp::common {

// 分阶段耗时（微秒）。由管线产出、输出层透传；两侧共用同一定义，
// 避免字段在多个结构体中重复声明后语义漂移。
// capture_us 是取帧等待，不属于处理耗时；处理耗时 = total_us - capture_us。
struct StageTimingUs {
    int64_t capture_us = 0;
    int64_t preprocess_us = 0;
    int64_t inference_us = 0;
    int64_t postprocess_us = 0;
    int64_t total_us = 0;

    int64_t processUs() const {
        const int64_t process = total_us - capture_us;
        return process > 0 ? process : 0;
    }
};

}  // namespace rkapp::common
