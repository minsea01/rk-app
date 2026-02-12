#include "rkapp/capture/MppSource.hpp"

#if RKAPP_WITH_MPP

#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_buffer.h>

// FFmpeg for demuxing (optional, for video files and RTSP)
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

// RGA for YUV->BGR conversion (optional)
#if RKNN_USE_RGA
#include <im2d.h>
#include <rga.h>
#endif

#include <iostream>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unistd.h>
#include <cerrno>

// Logging
#if __has_include("log.hpp")
#include "rkapp/common/log.hpp"
#else
#define LOGI(...) do { std::cout << "[INFO] MppSource: " << __VA_ARGS__ << std::endl; } while(0)
#define LOGW(...) do { std::cerr << "[WARN] MppSource: " << __VA_ARGS__ << std::endl; } while(0)
#define LOGE(...) do { std::cerr << "[ERROR] MppSource: " << __VA_ARGS__ << std::endl; } while(0)
#define LOGD(...) do { (void)0; } while(0)
#endif

namespace rkapp::capture {

// ============================================================================
// MPP Implementation Details
// ============================================================================

struct MppSource::Impl {
    // MPP context
    MppCtx mpp_ctx = nullptr;
    MppApi* mpi = nullptr;
    MppBufferGroup frame_group = nullptr;

    // FFmpeg demuxer (for video files / RTSP)
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;  // For extracting stream info only
    int video_stream_idx = -1;
    AVPacket* pkt = nullptr;

    // Frame conversion buffer
    cv::Mat bgr_frame;

    // Statistics
    double total_decode_time_ms = 0.0;
    int decode_count = 0;

    // Current DMA-BUF fd (when in DMA-BUF mode)
    int current_dma_fd = -1;

    // EOF handling for proper decoder flush
    bool eof_reached = false;

    // Thread safety
    std::mutex mtx;

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(mtx);

        if (mpi && mpp_ctx) {
            mpi->reset(mpp_ctx);
        }

        if (frame_group) {
            mpp_buffer_group_put(frame_group);
            frame_group = nullptr;
        }

        if (mpp_ctx) {
            mpp_destroy(mpp_ctx);
            mpp_ctx = nullptr;
            mpi = nullptr;
        }

        if (pkt) {
            av_packet_free(&pkt);
            pkt = nullptr;
        }

        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
            codec_ctx = nullptr;
        }

        if (fmt_ctx) {
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
        }

        if (current_dma_fd >= 0) {
            close(current_dma_fd);
            current_dma_fd = -1;
        }

        video_stream_idx = -1;
    }
};

// ============================================================================
// Static Methods
// ============================================================================

static std::once_flag mpp_check_flag;
static bool mpp_available = false;

bool MppSource::isMppAvailable() {
    std::call_once(mpp_check_flag, []() {
        // Try to create and immediately destroy an MPP context
        MppCtx ctx = nullptr;
        MppApi* mpi = nullptr;

        MPP_RET ret = mpp_create(&ctx, &mpi);
        if (ret == MPP_OK && ctx && mpi) {
            mpp_destroy(ctx);
            mpp_available = true;
            LOGI("MPP hardware decoding available");
        } else {
            mpp_available = false;
            LOGW("MPP hardware decoding not available");
        }
    });
    return mpp_available;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

MppSource::MppSource() : impl_(std::make_unique<Impl>()) {}

MppSource::~MppSource() {
    release();
}

// ============================================================================
// ISource Interface Implementation
// ============================================================================

bool MppSource::open(const std::string& uri) {
    if (!isMppAvailable()) {
        LOGE("MPP not available, cannot open: ", uri);
        return false;
    }

    release();  // Clean up any previous state
    uri_ = uri;
    impl_->eof_reached = false;  // Reset EOF flag for new stream

    // Determine codec type from file extension or probe
    MppCodingType coding_type = MPP_VIDEO_CodingAVC;  // Default H.264

    // Initialize FFmpeg demuxer for video files and RTSP
    if (uri.find("rtsp://") == 0 || uri.find(".mp4") != std::string::npos ||
        uri.find(".mkv") != std::string::npos || uri.find(".avi") != std::string::npos ||
        uri.find(".h264") != std::string::npos || uri.find(".h265") != std::string::npos) {

        // Open input
        int ret = avformat_open_input(&impl_->fmt_ctx, uri.c_str(), nullptr, nullptr);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOGE("Failed to open input: ", uri, " (", errbuf, ")");
            return false;
        }

        // Find stream info
        ret = avformat_find_stream_info(impl_->fmt_ctx, nullptr);
        if (ret < 0) {
            LOGE("Failed to find stream info");
            release();
            return false;
        }

        // Find video stream
        for (unsigned int i = 0; i < impl_->fmt_ctx->nb_streams; i++) {
            if (impl_->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                impl_->video_stream_idx = i;
                break;
            }
        }

        if (impl_->video_stream_idx < 0) {
            LOGE("No video stream found in: ", uri);
            release();
            return false;
        }

        AVStream* video_stream = impl_->fmt_ctx->streams[impl_->video_stream_idx];
        AVCodecParameters* codecpar = video_stream->codecpar;

        // Get video properties
        width_ = codecpar->width;
        height_ = codecpar->height;

        if (video_stream->avg_frame_rate.den > 0) {
            fps_ = av_q2d(video_stream->avg_frame_rate);
        } else if (video_stream->r_frame_rate.den > 0) {
            fps_ = av_q2d(video_stream->r_frame_rate);
        } else {
            fps_ = 30.0;  // Default
        }

        if (impl_->fmt_ctx->duration > 0 && fps_ > 0) {
            total_frames_ = static_cast<int>(
                (impl_->fmt_ctx->duration / AV_TIME_BASE) * fps_);
        } else {
            total_frames_ = -1;  // Unknown (streaming)
        }

        // Map FFmpeg codec to MPP codec type
        switch (codecpar->codec_id) {
            case AV_CODEC_ID_H264:
                coding_type = MPP_VIDEO_CodingAVC;
                break;
            case AV_CODEC_ID_HEVC:
                coding_type = MPP_VIDEO_CodingHEVC;
                break;
            case AV_CODEC_ID_VP9:
                coding_type = MPP_VIDEO_CodingVP9;
                break;
            case AV_CODEC_ID_VP8:
                coding_type = MPP_VIDEO_CodingVP8;
                break;
            case AV_CODEC_ID_MPEG4:
                coding_type = MPP_VIDEO_CodingMPEG4;
                break;
            case AV_CODEC_ID_MPEG2VIDEO:
                coding_type = MPP_VIDEO_CodingMPEG2;
                break;
            default:
                LOGE("Unsupported codec: ", avcodec_get_name(codecpar->codec_id));
                release();
                return false;
        }

        // Allocate packet
        impl_->pkt = av_packet_alloc();
        if (!impl_->pkt) {
            LOGE("Failed to allocate AVPacket");
            release();
            return false;
        }

        LOGI("Opened video: ", width_, "x", height_, " @ ", fps_, " fps, codec: ",
             avcodec_get_name(codecpar->codec_id));
    } else {
        // Unsupported URI format
        LOGE("MppSource::open: Unsupported URI format (not RTSP or known video file extension): ", uri);
        return false;
    }

    // Validate that fmt_ctx was initialized
    if (!impl_->fmt_ctx) {
        LOGE("MppSource::open: Failed to initialize demuxer for URI: ", uri);
        return false;
    }

    // Initialize MPP decoder
    MPP_RET ret = mpp_create(&impl_->mpp_ctx, &impl_->mpi);
    if (ret != MPP_OK) {
        LOGE("mpp_create failed: ", ret);
        release();
        return false;
    }

    // Configure MPP for split mode (feed complete frames)
    MppParam param = nullptr;
    int need_split = 1;
    ret = impl_->mpi->control(impl_->mpp_ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);
    if (ret != MPP_OK) {
        LOGW("Failed to set parser split mode");
    }

    // Initialize decoder
    ret = mpp_init(impl_->mpp_ctx, MPP_CTX_DEC, coding_type);
    if (ret != MPP_OK) {
        LOGE("mpp_init failed: ", ret);
        release();
        return false;
    }

    // Create external frame buffer group for DMA-BUF support
    ret = mpp_buffer_group_get_external(&impl_->frame_group, MPP_BUFFER_TYPE_DRM);
    if (ret != MPP_OK) {
        // Fallback to internal buffers
        LOGW("Failed to create DRM buffer group, using internal buffers");
        impl_->frame_group = nullptr;
    }

    is_opened_ = true;
    current_frame_ = 0;

    LOGI("MPP decoder initialized successfully");
    return true;
}

bool MppSource::read(cv::Mat& frame) {
    if (!is_opened_ || !impl_->mpi) {
        return false;
    }

    std::lock_guard<std::mutex> lock(impl_->mtx);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Retry limit to prevent infinite loop (was recursive, could cause stack overflow)
    constexpr int MAX_DECODE_RETRIES = 30;  // ~1 second at 30fps
    int retry_count = 0;

    MppFrame mpp_frame = nullptr;

    // Main decode loop - replaces recursive calls
    while (retry_count < MAX_DECODE_RETRIES) {
        // Safety check: ensure demuxer is initialized
        if (!impl_->fmt_ctx) {
            LOGE("MppSource::read: Demuxer not initialized. Call open() first.");
            return false;
        }

        // Read packet from demuxer
        while (true) {
            int ret = av_read_frame(impl_->fmt_ctx, impl_->pkt);
            if (ret < 0) {
                if (ret == AVERROR_EOF) {
                    // EOF reached - enter flush mode
                    if (impl_->eof_reached) {
                        // Already flushed, truly done
                        return false;
                    }
                    impl_->eof_reached = true;
                    LOGI("End of stream - flushing decoder");

                    // Send null packet to flush MPP decoder
                    av_packet_unref(impl_->pkt);
                    // Continue to MPP decode loop to drain buffered frames
                    break;
                } else {
                    // Other error
                    return false;
                }
            }

            // Skip non-video packets
            if (impl_->pkt->stream_index != impl_->video_stream_idx) {
                av_packet_unref(impl_->pkt);
                continue;
            }
            break;
        }

        // Create MPP packet
        MppPacket mpp_pkt = nullptr;
        mpp_packet_init(&mpp_pkt, impl_->pkt->data, impl_->pkt->size);

        // Set PTS if available
        if (impl_->pkt->pts != AV_NOPTS_VALUE) {
            mpp_packet_set_pts(mpp_pkt, impl_->pkt->pts);
        }

        // Send packet to decoder
        MPP_RET mpp_ret = impl_->mpi->decode_put_packet(impl_->mpp_ctx, mpp_pkt);
        mpp_packet_deinit(&mpp_pkt);
        av_packet_unref(impl_->pkt);

        if (mpp_ret != MPP_OK) {
            LOGW("decode_put_packet failed: ", mpp_ret);
            return false;
        }

        // Get decoded frame
        mpp_frame = nullptr;
        mpp_ret = impl_->mpi->decode_get_frame(impl_->mpp_ctx, &mpp_frame);

        if (mpp_ret != MPP_OK || !mpp_frame) {
            // Frame not ready yet, need more packets (B-frame reordering, etc.)
            retry_count++;
            continue;  // Loop back to read next packet
        }

        // Check for decode errors
        if (mpp_frame_get_errinfo(mpp_frame)) {
            LOGW("Decode error in frame, retry ", retry_count + 1, "/", MAX_DECODE_RETRIES);
            mpp_frame_deinit(&mpp_frame);
            mpp_frame = nullptr;
            retry_count++;
            continue;  // Loop back to read next packet
        }

        // Successfully got a valid frame, break out of retry loop
        break;
    }

    // Check if we exhausted retries
    if (retry_count >= MAX_DECODE_RETRIES || !mpp_frame) {
        LOGW("Decode failed after ", retry_count, " retries");
        if (mpp_frame) {
            mpp_frame_deinit(&mpp_frame);
        }
        return false;
    }

    // Get frame info
    int frm_width = mpp_frame_get_width(mpp_frame);
    int frm_height = mpp_frame_get_height(mpp_frame);
    int frm_h_stride = mpp_frame_get_hor_stride(mpp_frame);
    int frm_v_stride = mpp_frame_get_ver_stride(mpp_frame);
    MppFrameFormat frm_fmt = mpp_frame_get_fmt(mpp_frame);
    MppBuffer frm_buf = mpp_frame_get_buffer(mpp_frame);

    if (!frm_buf) {
        LOGW("Frame has no buffer");
        mpp_frame_deinit(&mpp_frame);
        return false;
    }

    // Get buffer pointer
    void* buf_ptr = mpp_buffer_get_ptr(frm_buf);

    // Store DMA-BUF fd if in DMA-BUF mode
    if (dma_buf_mode_) {
        int frame_fd = mpp_buffer_get_fd(frm_buf);
        int dup_fd = (frame_fd >= 0) ? dup(frame_fd) : -1;
        if (dup_fd < 0 && frame_fd >= 0) {
            LOGW("Failed to dup DMA-BUF fd: ", strerror(errno));
        } else {
            if (impl_->current_dma_fd >= 0) {
                close(impl_->current_dma_fd);
            }
            impl_->current_dma_fd = dup_fd;
        }
    }

    // Convert YUV to BGR
    // Most common format from MPP is NV12 (YUV420SP)
    if (frm_fmt == MPP_FMT_YUV420SP || frm_fmt == MPP_FMT_YUV420SP_10BIT) {
        bool converted = false;
#if RKNN_USE_RGA
        // Use RGA with stride-aware conversion (avoids CPU memcpy)
        // RGA can handle non-contiguous strides directly
        rga_buffer_t src_buf = {};
        src_buf.width = frm_width;
        src_buf.height = frm_height;
        src_buf.wstride = frm_h_stride;  // horizontal stride (may differ from width)
        src_buf.hstride = frm_v_stride;  // vertical stride (may differ from height)
        src_buf.format = RK_FORMAT_YCbCr_420_SP;
        src_buf.vir_addr = buf_ptr;

        frame.create(frm_height, frm_width, CV_8UC3);
        rga_buffer_t dst_buf = wrapbuffer_virtualaddr(
            frame.data, frm_width, frm_height,
            RK_FORMAT_BGR_888);

        IM_STATUS rga_ret = imcvtcolor(src_buf, dst_buf,
                                       RK_FORMAT_YCbCr_420_SP,
                                       RK_FORMAT_BGR_888,
                                       IM_YUV_TO_RGB_BT601_LIMIT);
        if (rga_ret == IM_STATUS_SUCCESS) {
            converted = true;
        } else {
            LOGW("RGA stride-aware cvtcolor failed (", imStrError(rga_ret), "), using CPU fallback");
        }
#endif
        if (!converted) {
            // CPU fallback path
            cv::Mat yuv_mat(frm_v_stride * 3 / 2, frm_h_stride, CV_8UC1, buf_ptr);

            // Crop to actual dimensions if stride differs
            cv::Mat yuv_cropped;
            if (frm_h_stride != frm_width || frm_v_stride != frm_height) {
                // Need to handle stride - create proper NV12 layout
                cv::Mat y_plane(frm_height, frm_width, CV_8UC1);
                cv::Mat uv_plane(frm_height / 2, frm_width / 2, CV_8UC2);

                // Copy Y plane row by row
                for (int i = 0; i < frm_height; i++) {
                    memcpy(y_plane.ptr(i),
                           (uint8_t*)buf_ptr + i * frm_h_stride,
                           frm_width);
                }

                // Copy UV plane row by row
                uint8_t* uv_src = (uint8_t*)buf_ptr + frm_h_stride * frm_v_stride;
                for (int i = 0; i < frm_height / 2; i++) {
                    memcpy(uv_plane.ptr(i),
                           uv_src + i * frm_h_stride,
                           frm_width);
                }

                // Combine for OpenCV cvtColor
                yuv_cropped = cv::Mat(frm_height * 3 / 2, frm_width, CV_8UC1);
                y_plane.copyTo(yuv_cropped(cv::Rect(0, 0, frm_width, frm_height)));

                // Reshape UV for copying
                cv::Mat uv_reshaped(frm_height / 2, frm_width, CV_8UC1,
                                   uv_plane.data);
                uv_reshaped.copyTo(
                    yuv_cropped(cv::Rect(0, frm_height, frm_width, frm_height / 2)));
            } else {
                yuv_cropped = yuv_mat(cv::Rect(0, 0, frm_width, frm_height * 3 / 2));
            }

            cv::cvtColor(yuv_cropped, frame, cv::COLOR_YUV2BGR_NV12);
        }
    } else {
        LOGW("Unsupported frame format: ", frm_fmt);
        mpp_frame_deinit(&mpp_frame);
        return false;
    }

    mpp_frame_deinit(&mpp_frame);

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    impl_->total_decode_time_ms += decode_ms;
    impl_->decode_count++;

    current_frame_++;
    return true;
}

void MppSource::release() {
    is_opened_ = false;
    if (impl_) {
        impl_->cleanup();
    }
}

bool MppSource::isOpened() const {
    return is_opened_;
}

double MppSource::getFPS() const {
    return fps_;
}

cv::Size MppSource::getSize() const {
    return cv::Size(width_, height_);
}

int MppSource::getTotalFrames() const {
    return total_frames_;
}

int MppSource::getCurrentFrame() const {
    return current_frame_;
}

SourceType MppSource::getType() const {
    if (uri_.find("rtsp://") == 0) {
        return SourceType::RTSP;
    }
    return SourceType::VIDEO;
}

// ============================================================================
// MPP-specific Methods
// ============================================================================

double MppSource::getDecodeLatencyMs() const {
    if (impl_->decode_count == 0) return 0.0;
    return impl_->total_decode_time_ms / impl_->decode_count;
}

void MppSource::setDmaBufMode(bool enable) {
    dma_buf_mode_ = enable;
}

int MppSource::getDmaBufFd() const {
    if (!impl_ || impl_->current_dma_fd < 0) return -1;
    int dup_fd = dup(impl_->current_dma_fd);
    if (dup_fd < 0) {
        LOGW("Failed to dup current DMA-BUF fd: ", strerror(errno));
        return -1;
    }
    return dup_fd;
}

} // namespace rkapp::capture

#else  // !RKAPP_WITH_MPP

// Stub implementation when MPP is not available
namespace rkapp::capture {

struct MppSource::Impl {};

bool MppSource::isMppAvailable() { return false; }
MppSource::MppSource() : impl_(std::make_unique<Impl>()) {}
MppSource::~MppSource() = default;

bool MppSource::open(const std::string&) {
    std::cerr << "[ERROR] MppSource: MPP support not compiled in. Rebuild with -DENABLE_MPP=ON\n";
    return false;
}

bool MppSource::read(cv::Mat&) { return false; }
void MppSource::release() { is_opened_ = false; }
bool MppSource::isOpened() const { return false; }
double MppSource::getFPS() const { return 0.0; }
cv::Size MppSource::getSize() const { return {}; }
int MppSource::getTotalFrames() const { return 0; }
int MppSource::getCurrentFrame() const { return 0; }
SourceType MppSource::getType() const { return SourceType::VIDEO; }
double MppSource::getDecodeLatencyMs() const { return 0.0; }
void MppSource::setDmaBufMode(bool) {}
int MppSource::getDmaBufFd() const { return -1; }

} // namespace rkapp::capture

#endif  // RKAPP_WITH_MPP
