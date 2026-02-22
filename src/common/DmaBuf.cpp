#include "rkapp/common/DmaBuf.hpp"
#include "rkapp/common/log.hpp"

#include <cstring>
#include <cerrno>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

// Linux DMA-BUF sync ioctls
#include <linux/dma-buf.h>

// DRM headers for GEM allocation (RK3588)
#if defined(__aarch64__) || defined(__arm__)
#include <drm/drm.h>
#include <drm/drm_mode.h>
// Rockchip-specific DRM extensions
#ifndef DRM_ROCKCHIP_GEM_CREATE
#define DRM_ROCKCHIP_GEM_CREATE 0x00
struct drm_rockchip_gem_create {
    uint64_t size;
    uint32_t flags;
    uint32_t handle;
};
#define DRM_IOCTL_ROCKCHIP_GEM_CREATE DRM_IOWR(DRM_COMMAND_BASE + DRM_ROCKCHIP_GEM_CREATE, struct drm_rockchip_gem_create)
#endif

#ifndef DRM_ROCKCHIP_GEM_MAP_OFFSET
#define DRM_ROCKCHIP_GEM_MAP_OFFSET 0x01
struct drm_rockchip_gem_map_offset {
    uint32_t handle;
    uint32_t pad;
    uint64_t offset;
};
#define DRM_IOCTL_ROCKCHIP_GEM_MAP_OFFSET DRM_IOWR(DRM_COMMAND_BASE + DRM_ROCKCHIP_GEM_MAP_OFFSET, struct drm_rockchip_gem_map_offset)
#endif
#endif

// RGA headers for handle import
#if defined(__aarch64__) || defined(__arm__)
#define RKAPP_HAS_RGA 1
#include <im2d.h>
#include <rga.h>
#endif

namespace rkapp::common {

namespace {

// Align size to page boundary
constexpr size_t alignToPage(size_t size) {
    constexpr size_t PAGE_SIZE = 4096;
    return (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
}

// Calculate buffer size for a given format
size_t calculateBufferSize(int width, int height, int stride, DmaBuf::PixelFormat format) {
    switch (format) {
        case DmaBuf::PixelFormat::BGR888:
        case DmaBuf::PixelFormat::RGB888:
            return static_cast<size_t>(stride) * height;
        case DmaBuf::PixelFormat::RGBA8888:
        case DmaBuf::PixelFormat::BGRA8888:
            return static_cast<size_t>(stride) * height;
        case DmaBuf::PixelFormat::NV12:
        case DmaBuf::PixelFormat::NV21:
            // Y plane + UV plane (half height)
            return static_cast<size_t>(stride) * height * 3 / 2;
        default:
            return static_cast<size_t>(stride) * height;
    }
}

// Open DRM device
int openDrmDevice() {
    // Try common DRM device paths on RK3588
    const char* devices[] = {
        "/dev/dri/card0",
        "/dev/dri/renderD128",
        "/dev/dri/renderD129"
    };

    for (const char* dev : devices) {
        int fd = open(dev, O_RDWR | O_CLOEXEC);
        if (fd >= 0) {
            LOGI("DmaBuf: Opened DRM device: ", dev);
            return fd;
        }
    }

    LOGE("DmaBuf: Failed to open any DRM device");
    return -1;
}

} // namespace

// ============================================================================
// DmaBuf Implementation
// ============================================================================

DmaBuf::DmaBuf() = default;

DmaBuf::~DmaBuf() {
    release();
}

DmaBuf::DmaBuf(DmaBuf&& other) noexcept
    : fd_(other.fd_)
    , virt_addr_(other.virt_addr_)
    , phys_addr_(other.phys_addr_)
    , width_(other.width_)
    , height_(other.height_)
    , stride_(other.stride_)
    , size_(other.size_)
    , format_(other.format_)
    , mem_type_(other.mem_type_)
    , rga_handle_(other.rga_handle_)
    , drm_fd_(other.drm_fd_)
    , gem_handle_(other.gem_handle_)
    , owns_fd_(other.owns_fd_)
{
    other.fd_ = -1;
    other.virt_addr_ = nullptr;
    other.phys_addr_ = 0;
    other.rga_handle_ = 0;
    other.drm_fd_ = -1;
    other.gem_handle_ = 0;
    other.owns_fd_ = false;
}

DmaBuf& DmaBuf::operator=(DmaBuf&& other) noexcept {
    if (this != &other) {
        release();
        fd_ = other.fd_;
        virt_addr_ = other.virt_addr_;
        phys_addr_ = other.phys_addr_;
        width_ = other.width_;
        height_ = other.height_;
        stride_ = other.stride_;
        size_ = other.size_;
        format_ = other.format_;
        mem_type_ = other.mem_type_;
        rga_handle_ = other.rga_handle_;
        drm_fd_ = other.drm_fd_;
        gem_handle_ = other.gem_handle_;
        owns_fd_ = other.owns_fd_;

        other.fd_ = -1;
        other.virt_addr_ = nullptr;
        other.phys_addr_ = 0;
        other.rga_handle_ = 0;
        other.drm_fd_ = -1;
        other.gem_handle_ = 0;
        other.owns_fd_ = false;
    }
    return *this;
}

void DmaBuf::release() {
    // Release RGA handle
#if RKAPP_HAS_RGA
    if (rga_handle_ != 0) {
        releasebuffer_handle(static_cast<rga_buffer_handle_t>(rga_handle_));
        rga_handle_ = 0;
    }
#endif

    // Unmap virtual address
    if (virt_addr_ != nullptr && virt_addr_ != MAP_FAILED) {
        munmap(virt_addr_, size_);
        virt_addr_ = nullptr;
    }

    // Close DMA-BUF fd
    if (fd_ >= 0 && owns_fd_) {
        close(fd_);
    }
    fd_ = -1;

    // Release GEM handle
#if defined(__aarch64__) || defined(__arm__)
    if (gem_handle_ != 0 && drm_fd_ >= 0) {
        struct drm_gem_close close_args = {};
        close_args.handle = gem_handle_;
        ioctl(drm_fd_, DRM_IOCTL_GEM_CLOSE, &close_args);
        gem_handle_ = 0;
    }
#endif

    // Close DRM device
    if (drm_fd_ >= 0) {
        close(drm_fd_);
        drm_fd_ = -1;
    }

    width_ = height_ = stride_ = 0;
    size_ = 0;
    phys_addr_ = 0;
    owns_fd_ = false;
    sync_supported_ = true;
    sync_warned_ = false;
}

bool DmaBuf::allocate(int width, int height, PixelFormat format, MemType mem_type) {
    release();

    if (width <= 0 || height <= 0) {
        LOGE("DmaBuf::allocate: Invalid dimensions: ", width, "x", height);
        return false;
    }

    width_ = width;
    height_ = height;
    format_ = format;
    mem_type_ = mem_type;
    sync_supported_ = true;
    sync_warned_ = false;

    // Calculate stride (align to 16 bytes for RGA/NPU efficiency)
    int bpp = bytesPerPixel(format);
    stride_ = (width * bpp + 15) & ~15;
    size_ = alignToPage(calculateBufferSize(width, height, stride_, format));

#if defined(__aarch64__) || defined(__arm__)
    if (mem_type == MemType::DRM_GEM) {
        // Open DRM device
        drm_fd_ = openDrmDevice();
        if (drm_fd_ < 0) {
            LOGW("DmaBuf::allocate: DRM not available, falling back to system memory");
            mem_type_ = MemType::SYSTEM;
        } else {
            // Allocate GEM buffer
            struct drm_rockchip_gem_create create_args = {};
            create_args.size = size_;
            create_args.flags = 0;  // Cacheable

            int ret = ioctl(drm_fd_, DRM_IOCTL_ROCKCHIP_GEM_CREATE, &create_args);
            if (ret < 0) {
                LOGE("DmaBuf::allocate: DRM_IOCTL_ROCKCHIP_GEM_CREATE failed: ", strerror(errno));
                close(drm_fd_);
                drm_fd_ = -1;
                mem_type_ = MemType::SYSTEM;
            } else {
                gem_handle_ = create_args.handle;

                // Export as DMA-BUF
                struct drm_prime_handle prime_args = {};
                prime_args.handle = gem_handle_;
                prime_args.flags = DRM_CLOEXEC | DRM_RDWR;

                ret = ioctl(drm_fd_, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime_args);
                if (ret < 0) {
                    LOGE("DmaBuf::allocate: DRM_IOCTL_PRIME_HANDLE_TO_FD failed: ", strerror(errno));
                    release();
                    return false;
                }

                fd_ = prime_args.fd;
                owns_fd_ = true;
                LOGI("DmaBuf::allocate: Allocated ", size_, " bytes via DRM GEM (fd=", fd_, ")");
            }
        }
    }
#endif

    // Fallback: allocate via memfd + mmap (works on all Linux)
    if (fd_ < 0 && mem_type_ == MemType::SYSTEM) {
        fd_ = memfd_create("dmabuf", MFD_CLOEXEC);
        if (fd_ < 0) {
            LOGE("DmaBuf::allocate: memfd_create failed: ", strerror(errno));
            return false;
        }

        if (ftruncate(fd_, size_) < 0) {
            LOGE("DmaBuf::allocate: ftruncate failed: ", strerror(errno));
            close(fd_);
            fd_ = -1;
            return false;
        }

        owns_fd_ = true;
        LOGI("DmaBuf::allocate: Allocated ", size_, " bytes via memfd (fd=", fd_, ")");
    }

    return isValid();
}

bool DmaBuf::importFd(int fd, int width, int height, PixelFormat format, int stride) {
    release();

    if (fd < 0 || width <= 0 || height <= 0) {
        LOGE("DmaBuf::importFd: Invalid parameters");
        return false;
    }

    int dup_fd = dup(fd);
    if (dup_fd < 0) {
        LOGE("DmaBuf::importFd: dup failed: ", strerror(errno));
        return false;
    }

    fd_ = dup_fd;
    owns_fd_ = true;  // Own the duplicated fd
    width_ = width;
    height_ = height;
    format_ = format;
    sync_supported_ = true;
    sync_warned_ = false;

    int bpp = bytesPerPixel(format);
    stride_ = (stride > 0) ? stride : (width * bpp);
    size_ = calculateBufferSize(width, height, stride_, format);

    LOGI("DmaBuf::importFd: Imported fd=", fd, " (", width, "x", height, ")");
    return true;
}

int DmaBuf::exportFd() const {
    if (fd_ < 0) return -1;
    return dup(fd_);  // Return a duplicate; caller owns it
}

void* DmaBuf::getVirtAddr() {
    if (virt_addr_ != nullptr) return virt_addr_;
    if (fd_ < 0) return nullptr;

#if defined(__aarch64__) || defined(__arm__)
    // For GEM buffers, get mmap offset first
    if (drm_fd_ >= 0 && gem_handle_ != 0) {
        struct drm_rockchip_gem_map_offset map_args = {};
        map_args.handle = gem_handle_;

        int ret = ioctl(drm_fd_, DRM_IOCTL_ROCKCHIP_GEM_MAP_OFFSET, &map_args);
        if (ret < 0) {
            LOGE("DmaBuf::getVirtAddr: DRM_IOCTL_ROCKCHIP_GEM_MAP_OFFSET failed");
            return nullptr;
        }

        virt_addr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                          drm_fd_, map_args.offset);
    } else
#endif
    {
        // Direct mmap for memfd or imported DMA-BUF
        virt_addr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    }

    if (virt_addr_ == MAP_FAILED) {
        LOGE("DmaBuf::getVirtAddr: mmap failed: ", strerror(errno));
        virt_addr_ = nullptr;
        return nullptr;
    }

    return virt_addr_;
}

const void* DmaBuf::getVirtAddr() const {
    return const_cast<DmaBuf*>(this)->getVirtAddr();
}

uint64_t DmaBuf::getPhysAddr() const {
    return phys_addr_;
}

uint64_t DmaBuf::getRgaHandle() const {
#if RKAPP_HAS_RGA
    // Fast path: 已初始化时无需加锁。
    if (rga_handle_ != 0) return rga_handle_;
    if (fd_ < 0) return 0;

    // Slow path: 用 rga_mutex_ 保护 lazy init，防止多线程重复 import fd。
    std::lock_guard<std::mutex> lock(rga_mutex_);
    // Double-check：另一个线程可能已完成初始化。
    if (rga_handle_ != 0) return rga_handle_;

    rga_buffer_handle_t handle = importbuffer_fd(fd_, size_);
    if (handle == 0) {
        LOGE("DmaBuf::getRgaHandle: importbuffer_fd failed");
        return 0;
    }

    rga_handle_ = static_cast<uint64_t>(handle);
    return rga_handle_;
#else
    return 0;
#endif
}

bool DmaBuf::sync(uint64_t flags) {
    if (fd_ < 0) return false;
    if (!sync_supported_) return false;

    struct dma_buf_sync sync = {};
    sync.flags = flags;
    if (ioctl(fd_, DMA_BUF_IOCTL_SYNC, &sync) < 0) {
        if (errno == ENOTTY || errno == EINVAL || errno == EOPNOTSUPP) {
            if (!sync_warned_) {
                LOGW("DmaBuf: DMA_BUF_IOCTL_SYNC not supported on fd=", fd_);
            }
            sync_supported_ = false;
        } else if (!sync_warned_) {
            LOGW("DmaBuf: DMA_BUF_IOCTL_SYNC failed: ", strerror(errno));
        }
        sync_warned_ = true;
        return false;
    }

    return true;
}

void DmaBuf::syncForCpuReadStart() {
    sync(DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
}

void DmaBuf::syncForCpuReadEnd() {
    sync(DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
}

void DmaBuf::syncForCpuWriteStart() {
    sync(DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
}

void DmaBuf::syncForCpuWriteEnd() {
    sync(DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
}

void DmaBuf::syncForCpu() {
    syncForCpuReadStart();
}

void DmaBuf::syncForDevice() {
    syncForCpuWriteEnd();
}

cv::Mat DmaBuf::asMat() {
    void* ptr = getVirtAddr();
    if (!ptr) return cv::Mat();

    int cv_type;
    int rows = height_;
    int cols = width_;
    switch (format_) {
        case PixelFormat::BGR888:
        case PixelFormat::RGB888:
            cv_type = CV_8UC3;
            break;
        case PixelFormat::RGBA8888:
        case PixelFormat::BGRA8888:
            cv_type = CV_8UC4;
            break;
        case PixelFormat::NV12:
        case PixelFormat::NV21:
            // Return full YUV420SP buffer view (Y + interleaved UV).
            cv_type = CV_8UC1;
            rows = height_ + height_ / 2;
            break;
        default:
            cv_type = CV_8UC3;
    }

    // Create Mat with external data (no ownership transfer)
    return cv::Mat(rows, cols, cv_type, ptr, stride_);
}

bool DmaBuf::copyFrom(const cv::Mat& src) {
    if (src.empty()) return false;
    void* ptr = getVirtAddr();
    if (!ptr) return false;

    syncForCpuWriteStart();
    bool ok = true;

    if (format_ == PixelFormat::NV12 || format_ == PixelFormat::NV21) {
        const int expected_rows = height_ + height_ / 2;
        if (src.type() != CV_8UC1 || src.cols != width_ || src.rows != expected_rows) {
            LOGE("DmaBuf::copyFrom: NV12/NV21 expects CV_8UC1 (", expected_rows, "x",
                 width_, "), got type=", src.type(), " shape=", src.rows, "x", src.cols);
            ok = false;
        } else {
            const size_t row_bytes = static_cast<size_t>(width_);
            const int uv_rows = height_ / 2;
            const uint8_t* src_ptr = src.ptr<uint8_t>(0);
            uint8_t* dst_ptr = static_cast<uint8_t*>(ptr);

            // Y plane
            for (int y = 0; y < height_; ++y) {
                std::memcpy(dst_ptr + static_cast<size_t>(y) * stride_,
                            src_ptr + static_cast<size_t>(y) * src.step, row_bytes);
            }

            // UV plane (interleaved)
            const uint8_t* src_uv = src_ptr + static_cast<size_t>(height_) * src.step;
            uint8_t* dst_uv = dst_ptr + static_cast<size_t>(height_) * stride_;
            for (int y = 0; y < uv_rows; ++y) {
                std::memcpy(dst_uv + static_cast<size_t>(y) * stride_,
                            src_uv + static_cast<size_t>(y) * src.step, row_bytes);
            }
        }
    } else {
        if (src.cols != width_ || src.rows != height_) {
            LOGE("DmaBuf::copyFrom: Dimension mismatch");
            ok = false;
        } else {
            // Copy row by row (handles stride differences)
            const size_t row_bytes = static_cast<size_t>(width_) * bytesPerPixel(format_);
            const uint8_t* src_ptr = src.data;
            uint8_t* dst_ptr = static_cast<uint8_t*>(ptr);

            for (int y = 0; y < height_; ++y) {
                std::memcpy(dst_ptr, src_ptr, row_bytes);
                src_ptr += src.step;
                dst_ptr += stride_;
            }
        }
    }

    syncForCpuWriteEnd();
    return ok;
}

bool DmaBuf::copyTo(cv::Mat& dst) const {
    const void* ptr = getVirtAddr();
    if (!ptr) return false;

    int cv_type;
    int rows = height_;
    int cols = width_;
    switch (format_) {
        case PixelFormat::BGR888:
        case PixelFormat::RGB888:
            cv_type = CV_8UC3;
            break;
        case PixelFormat::RGBA8888:
        case PixelFormat::BGRA8888:
            cv_type = CV_8UC4;
            break;
        case PixelFormat::NV12:
        case PixelFormat::NV21:
            cv_type = CV_8UC1;
            rows = height_ + height_ / 2;
            break;
        default:
            cv_type = CV_8UC3;
    }

    dst.create(rows, cols, cv_type);

    const_cast<DmaBuf*>(this)->syncForCpuReadStart();
    if (format_ == PixelFormat::NV12 || format_ == PixelFormat::NV21) {
        const size_t row_bytes = static_cast<size_t>(width_);
        const int uv_rows = height_ / 2;
        const uint8_t* src_ptr = static_cast<const uint8_t*>(ptr);
        uint8_t* dst_ptr = dst.ptr<uint8_t>(0);

        // Y plane
        for (int y = 0; y < height_; ++y) {
            std::memcpy(dst_ptr + static_cast<size_t>(y) * dst.step,
                        src_ptr + static_cast<size_t>(y) * stride_, row_bytes);
        }

        // UV plane (interleaved)
        const uint8_t* src_uv = src_ptr + static_cast<size_t>(height_) * stride_;
        uint8_t* dst_uv = dst_ptr + static_cast<size_t>(height_) * dst.step;
        for (int y = 0; y < uv_rows; ++y) {
            std::memcpy(dst_uv + static_cast<size_t>(y) * dst.step,
                        src_uv + static_cast<size_t>(y) * stride_, row_bytes);
        }
    } else {
        const size_t row_bytes = static_cast<size_t>(width_) * bytesPerPixel(format_);
        const uint8_t* src_ptr = static_cast<const uint8_t*>(ptr);
        uint8_t* dst_ptr = dst.data;

        for (int y = 0; y < height_; ++y) {
            std::memcpy(dst_ptr, src_ptr, row_bytes);
            src_ptr += stride_;
            dst_ptr += dst.step;
        }
    }

    const_cast<DmaBuf*>(this)->syncForCpuReadEnd();
    return true;
}

bool DmaBuf::isSupported() {
#if defined(RKAPP_WITH_DRM) && (defined(__aarch64__) || defined(__arm__))
    // Try to open DRM device
    int fd = openDrmDevice();
    if (fd >= 0) {
        close(fd);
        return true;
    }
#endif
    return false;
}

int DmaBuf::bytesPerPixel(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::BGR888:
        case PixelFormat::RGB888:
            return 3;
        case PixelFormat::RGBA8888:
        case PixelFormat::BGRA8888:
            return 4;
        case PixelFormat::NV12:
        case PixelFormat::NV21:
            return 1;  // Y plane only; actual buffer is 1.5x
        default:
            return 3;
    }
}

// ============================================================================
// DmaBufPool Implementation
// ============================================================================

struct DmaBufPool::Impl {
    std::mutex mutex;
    std::vector<std::unique_ptr<DmaBuf>> buffers;
    std::queue<DmaBuf*> available;
    std::unordered_set<DmaBuf*> all_buffers;
    std::unordered_set<DmaBuf*> checked_out;
    int width, height;
    DmaBuf::PixelFormat format;
};

DmaBufPool::DmaBufPool(int count, int width, int height, DmaBuf::PixelFormat format)
    : impl_(std::make_unique<Impl>())
{
    impl_->width = width;
    impl_->height = height;
    impl_->format = format;

    for (int i = 0; i < count; ++i) {
        auto buf = std::make_unique<DmaBuf>();
        if (buf->allocate(width, height, format)) {
            DmaBuf* raw = buf.get();
            impl_->available.push(raw);
            impl_->all_buffers.insert(raw);
            impl_->buffers.push_back(std::move(buf));
        } else {
            LOGW("DmaBufPool: Failed to allocate buffer ", i);
        }
    }

    LOGI("DmaBufPool: Created pool with ", impl_->buffers.size(), " buffers (",
         width, "x", height, ")");
}

DmaBufPool::~DmaBufPool() = default;

DmaBuf* DmaBufPool::acquire() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->available.empty()) {
        return nullptr;
    }
    DmaBuf* buf = impl_->available.front();
    impl_->available.pop();
    impl_->checked_out.insert(buf);
    return buf;
}

void DmaBufPool::release(DmaBuf* buf) {
    if (!buf) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->all_buffers.find(buf) == impl_->all_buffers.end()) {
        LOGW("DmaBufPool: Attempted to release foreign buffer");
        return;
    }
    if (impl_->checked_out.erase(buf) == 0) {
        LOGW("DmaBufPool: Attempted double-release for buffer ", static_cast<const void*>(buf));
        return;
    }
    impl_->available.push(buf);
}

int DmaBufPool::available() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return static_cast<int>(impl_->available.size());
}

} // namespace rkapp::common
