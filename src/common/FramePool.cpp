#include "rkapp/common/FramePool.hpp"

#include <chrono>

namespace rkapp::common {

FramePool::FramePool(int count, int width, int height, int type)
    : capacity_(count)
    , width_(width)
    , height_(height)
    , type_(type)
{
    // Pre-allocate all buffers
    buffers_.reserve(count);
    for (int i = 0; i < count; ++i) {
        buffers_.emplace_back(height, width, type);
        available_indices_.push(i);
    }
}

cv::Mat* FramePool::acquire(int timeout_ms) {
    stat_acquires_.fetch_add(1, std::memory_order_relaxed);

    std::unique_lock<std::mutex> lock(mutex_);

    if (available_indices_.empty()) {
        if (timeout_ms <= 0) {
            // Non-blocking mode
            stat_exhausted_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }

        // Wait with timeout
        stat_waits_.fetch_add(1, std::memory_order_relaxed);
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);

        bool got_buffer = cv_.wait_until(lock, deadline, [this]() {
            return !available_indices_.empty();
        });

        if (!got_buffer) {
            stat_exhausted_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
    }

    int idx = available_indices_.front();
    available_indices_.pop();

    return &buffers_[idx];
}

void FramePool::release(cv::Mat* buf) {
    if (!buf) return;

    stat_releases_.fetch_add(1, std::memory_order_relaxed);

    // Find the buffer index
    ptrdiff_t idx = buf - buffers_.data();
    if (idx < 0 || idx >= capacity_) {
        // Not from this pool - ignore
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        available_indices_.push(static_cast<int>(idx));
    }
    cv_.notify_one();
}

int FramePool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(available_indices_.size());
}

bool FramePool::owns(const cv::Mat* buf) const {
    if (!buf || buffers_.empty()) return false;
    ptrdiff_t idx = buf - buffers_.data();
    return idx >= 0 && idx < capacity_;
}

FramePool::Stats FramePool::getStats() const {
    Stats stats;
    stats.acquires = stat_acquires_.load(std::memory_order_relaxed);
    stats.releases = stat_releases_.load(std::memory_order_relaxed);
    stats.waits = stat_waits_.load(std::memory_order_relaxed);
    stats.exhausted = stat_exhausted_.load(std::memory_order_relaxed);
    return stats;
}

} // namespace rkapp::common
