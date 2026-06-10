#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "rkapp/common/StageTiming.hpp"
#include "rkapp/infer/IInferEngine.hpp"

namespace rkapp::output {

struct FrameResult {
    int frame_id = -1;
    int64_t timestamp = 0;
    int width = 0, height = 0;
    std::vector<rkapp::infer::Detection> detections;
    std::string source_uri;
    std::vector<uint8_t> image_bytes;
    std::string image_encoding;
    bool image_contains_overlays = false;
    int image_width = 0;
    int image_height = 0;
    bool image_roi_applied = false;
    cv::Rect image_roi{0, 0, 0, 0};
    // 与 PipelineResult::Timing 共用同一定义；处理耗时经 timing.processUs() 推导。
    common::StageTimingUs timing;
};

enum class OutputType {
    TCP
};

class IOutput {
public:
    virtual ~IOutput() = default;

    virtual bool open(const std::string& config = "") = 0;
    // 提交一帧结果用于输出。实现可以异步投递：返回 true 表示“已接受”，
    // 不保证已送达；返回 false 表示输出已关闭或拒绝接受。
    virtual bool send(const FrameResult& result) = 0;
    virtual void close() = 0;
    virtual bool isOpened() const = 0;

    virtual OutputType getType() const = 0;
};

using OutputPtr = std::unique_ptr<IOutput>;

} // namespace rkapp::output
