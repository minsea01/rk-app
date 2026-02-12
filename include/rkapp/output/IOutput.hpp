#pragma once

#include <string>
#include <vector>
#include <memory>
#include "rkapp/infer/IInferEngine.hpp"

namespace rkapp::output {

struct FrameResult {
    int frame_id;
    int64_t timestamp;
    int width, height;
    std::vector<rkapp::infer::Detection> detections;
    std::string source_uri;
};

enum class OutputType {
    TCP
};

class IOutput {
public:
    virtual ~IOutput() = default;
    
    virtual bool open(const std::string& config = "") = 0;
    virtual bool send(const FrameResult& result) = 0;
    virtual void close() = 0;
    virtual bool isOpened() const = 0;
    
    virtual OutputType getType() const = 0;
};

using OutputPtr = std::unique_ptr<IOutput>;

} // namespace rkapp::output
