#pragma once

#include <cstdint>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>

namespace rkapp::common {

/**
 * @brief DMA-BUF memory wrapper for zero-copy hardware pipeline
 *
 * This class manages DRM/GEM-backed memory that can be shared between
 * RK3588 hardware blocks (MPP decoder, RGA, NPU) without CPU copies.
 *
 * Memory allocation path: DRM GEM → DMA-BUF fd → import to RGA/RKNN
 *
 * Zero-copy pipeline:
 *   [MPP decode] → DMA-BUF → [RGA preprocess] → DMA-BUF → [NPU inference]
 *         ↓              (no copy)       ↓           (no copy)       ↓
 *   hardware output    shared fd    hardware in/out  shared fd   hardware input
 *
 * Performance improvement: ~3-5x reduction in memory bandwidth vs cv::Mat copies
 */
class DmaBuf {
public:
    /**
     * @brief Pixel format for DMA buffer
     */
    enum class PixelFormat {
        BGR888,      // 3-channel BGR (OpenCV default)
        RGB888,      // 3-channel RGB (RKNN input)
        NV12,        // YUV420SP (MPP decoder output)
        NV21,        // YUV420SP (alternate)
        RGBA8888,    // 4-channel RGBA
        BGRA8888     // 4-channel BGRA
    };

    /**
     * @brief Memory type for allocation
     */
    enum class MemType {
        DRM_GEM,     // DRM GEM buffer (default, best for cross-device sharing)
        CMA,         // Contiguous Memory Allocator
        SYSTEM       // System memory with DMA-BUF export (fallback)
    };

    DmaBuf();
    ~DmaBuf();

    // Non-copyable, but movable
    DmaBuf(const DmaBuf&) = delete;
    DmaBuf& operator=(const DmaBuf&) = delete;
    DmaBuf(DmaBuf&& other) noexcept;
    DmaBuf& operator=(DmaBuf&& other) noexcept;

    /**
     * @brief Allocate a DMA buffer
     *
     * @param width Width in pixels
     * @param height Height in pixels
     * @param format Pixel format
     * @param mem_type Memory allocation type
     * @return true if allocation succeeded
     */
    bool allocate(int width, int height, PixelFormat format = PixelFormat::RGB888,
                  MemType mem_type = MemType::DRM_GEM);

    /**
     * @brief Import an existing DMA-BUF fd (e.g., from MPP decoder)
     *
     * @param fd DMA-BUF file descriptor (duplicated internally; caller retains ownership)
     * @param width Width in pixels
     * @param height Height in pixels
     * @param format Pixel format
     * @param stride Row stride in bytes (0 = auto-calculate)
     * @return true if import succeeded
     */
    bool importFd(int fd, int width, int height, PixelFormat format, int stride = 0);

    /**
     * @brief Export as DMA-BUF fd for sharing with other hardware
     *
     * @return DMA-BUF file descriptor (-1 on error)
     * @note The returned fd is a dup; caller must close it
     */
    int exportFd() const;

    /**
     * @brief Get virtual address for CPU access (requires sync)
     *
     * @return Virtual address, or nullptr if not mapped
     */
    void* getVirtAddr();
    const void* getVirtAddr() const;

    /**
     * @brief Get physical address for hardware access
     *
     * @return Physical address (valid only for CMA/GEM allocations)
     */
    uint64_t getPhysAddr() const;

    /**
     * @brief Get RGA handle for im2d operations
     *
     * @return rga_buffer_handle_t value (0 on error)
     */
    uint64_t getRgaHandle() const;

    /**
     * @brief Sync for CPU read access (call before reading DMA buffer on CPU)
     */
    void syncForCpuReadStart();

    /**
     * @brief Sync end for CPU read access (call after CPU read completes)
     */
    void syncForCpuReadEnd();

    /**
     * @brief Sync for CPU write access (call before writing DMA buffer on CPU)
     */
    void syncForCpuWriteStart();

    /**
     * @brief Sync end for CPU write access (call after CPU write completes)
     */
    void syncForCpuWriteEnd();

    /**
     * @brief Legacy: Sync for CPU read access start
     */
    void syncForCpu();

    /**
     * @brief Legacy: Sync end for CPU write access
     */
    void syncForDevice();

    /**
     * @brief Wrap as cv::Mat for CPU processing (sync required)
     *
     * @return cv::Mat pointing to DMA buffer memory
     * @note The returned Mat does NOT own the memory
     */
    cv::Mat asMat();

    /**
     * @brief Copy from cv::Mat to DMA buffer
     *
     * @param src Source Mat.
     * For RGB/BGR/RGBA/BGRA: shape must be (height, width, channels).
     * For NV12/NV21: shape must be (height * 3 / 2, width), type CV_8UC1.
     * @return true on success
     */
    bool copyFrom(const cv::Mat& src);

    /**
     * @brief Copy to cv::Mat from DMA buffer
     *
     * @param dst Destination Mat.
     * For NV12/NV21 output shape is (height * 3 / 2, width), CV_8UC1.
     * @return true on success
     */
    bool copyTo(cv::Mat& dst) const;

    // Accessors
    int width() const { return width_; }
    int height() const { return height_; }
    int stride() const { return stride_; }
    size_t size() const { return size_; }
    PixelFormat format() const { return format_; }
    bool isValid() const { return fd_ >= 0 && size_ > 0; }

    /**
     * @brief Check if DMA-BUF allocation is supported on this platform
     */
    static bool isSupported();

    /**
     * @brief Get bytes per pixel for a format
     */
    static int bytesPerPixel(PixelFormat fmt);

private:
    void release();
    bool sync(uint64_t flags);

    int fd_ = -1;              // DMA-BUF file descriptor
    void* virt_addr_ = nullptr; // Mapped virtual address
    uint64_t phys_addr_ = 0;   // Physical address (if available)
    int width_ = 0;
    int height_ = 0;
    int stride_ = 0;           // Row stride in bytes
    size_t size_ = 0;          // Total buffer size
    PixelFormat format_ = PixelFormat::RGB888;
    MemType mem_type_ = MemType::DRM_GEM;

    // RGA handle (cached to avoid repeated imports)
    mutable uint64_t rga_handle_ = 0;

    // DRM handle for GEM allocations
    int drm_fd_ = -1;
    uint32_t gem_handle_ = 0;

    // Track ownership
    bool owns_fd_ = false;

    // DMA-BUF sync capability
    bool sync_supported_ = true;
    bool sync_warned_ = false;
};

/**
 * @brief Pool of DMA buffers for reuse
 *
 * Pre-allocates a set of DMA buffers to avoid allocation overhead
 * during inference. Thread-safe for producer-consumer patterns.
 */
class DmaBufPool {
public:
    DmaBufPool(int count, int width, int height,
               DmaBuf::PixelFormat format = DmaBuf::PixelFormat::RGB888);
    ~DmaBufPool();

    /**
     * @brief Acquire a buffer from the pool
     *
     * @return Pointer to DmaBuf, or nullptr if pool exhausted
     * @note Call release() when done
     */
    DmaBuf* acquire();

    /**
     * @brief Release a buffer back to the pool
     */
    void release(DmaBuf* buf);

    /**
     * @brief Get number of available buffers
     */
    int available() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rkapp::common
