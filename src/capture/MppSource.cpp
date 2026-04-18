#include "rkapp/capture/MppSource.hpp"

#if RKAPP_WITH_MPP
// 本文件是“硬件视频解码采集源”：
// FFmpeg 负责拆包（demux），MPP 负责硬解码，最终输出 OpenCV BGR 帧。
// 可选 DMA-BUF 模式用于零拷贝链路。

#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_buffer.h>

// FFmpeg 拆包（可选，用于视频文件和 RTSP）
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

// RGA 用于 YUV->BGR 转换（可选）
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

// 日志
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
// MPP 内部实现细节
// ============================================================================

struct MppSource::Impl {
    // MPP 上下文
    MppCtx mpp_ctx = nullptr;
    MppApi* mpi = nullptr;
    MppBufferGroup frame_group = nullptr;

    // FFmpeg 拆包器（视频文件 / RTSP）
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;  // 仅用于获取流信息
    int video_stream_idx = -1;
    AVPacket* pkt = nullptr;

    // 帧转换缓存
    cv::Mat bgr_frame;

    // 统计信息
    double total_decode_time_ms = 0.0;
    int decode_count = 0;

    // 当前帧 DMA-BUF fd（仅在 DMA 模式下有效）
    // 注意这里保存的是 dup 后的 fd，生命周期由本对象管理。
    int current_dma_fd = -1;

    // EOF 处理：用于正确 flush 解码器
    bool eof_reached = false;

    // 线程安全
    std::mutex mtx;

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        // cleanup 可能由析构/显式 release 多次调用，按“可重入”方式写防御逻辑。
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
// 静态方法
// ============================================================================

static std::once_flag mpp_check_flag;
static bool mpp_available = false;

bool MppSource::isMppAvailable() {
    std::call_once(mpp_check_flag, []() {
        // 用最小代价探测 MPP 是否可用：创建并立即销毁上下文。
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
// 构造/析构
// ============================================================================

MppSource::MppSource() : impl_(std::make_unique<Impl>()) {}

MppSource::~MppSource() {
    release();
}

// ============================================================================
// ISource 接口实现
// ============================================================================

bool MppSource::open(const std::string& uri) {
    if (!isMppAvailable()) {
        LOGE("MPP not available, cannot open: ", uri);
        return false;
    }

    release();  // 清理旧状态
    uri_ = uri;
    impl_->eof_reached = false;  // 新流打开时重置 EOF 状态

    // 默认按 H.264 处理，后续会根据流实际 codec_id 纠正。
    MppCodingType coding_type = MPP_VIDEO_CodingAVC;  // 默认按 H.264

    // 先用 FFmpeg 打开流并做拆包，MPP 只负责解码，不负责容器解析。
    if (uri.find("rtsp://") == 0 || uri.find(".mp4") != std::string::npos ||
        uri.find(".mkv") != std::string::npos || uri.find(".avi") != std::string::npos ||
        uri.find(".h264") != std::string::npos || uri.find(".h265") != std::string::npos) {

        // 打开输入流
        int ret = avformat_open_input(&impl_->fmt_ctx, uri.c_str(), nullptr, nullptr);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOGE("Failed to open input: ", uri, " (", errbuf, ")");
            return false;
        }

        // 获取流信息
        ret = avformat_find_stream_info(impl_->fmt_ctx, nullptr);
        if (ret < 0) {
            LOGE("Failed to find stream info");
            release();
            return false;
        }

        // 找到第一个视频流索引（忽略音频/字幕等）。
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

        // 从 stream 参数提取基础信息。
        width_ = codecpar->width;
        height_ = codecpar->height;

        if (video_stream->avg_frame_rate.den > 0) {
            fps_ = av_q2d(video_stream->avg_frame_rate);
        } else if (video_stream->r_frame_rate.den > 0) {
            fps_ = av_q2d(video_stream->r_frame_rate);
        } else {
            fps_ = 30.0;  // 默认帧率
        }

        if (impl_->fmt_ctx->duration > 0 && fps_ > 0) {
            total_frames_ = static_cast<int>(
                (impl_->fmt_ctx->duration / AV_TIME_BASE) * fps_);
        } else {
            total_frames_ = -1;  // 流媒体场景通常未知
        }

        // 把 FFmpeg codec_id 映射为 MPP 识别的编码类型。
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

        // AVPacket 复用：每次 av_read_frame 填充后再 unref。
        impl_->pkt = av_packet_alloc();
        if (!impl_->pkt) {
            LOGE("Failed to allocate AVPacket");
            release();
            return false;
        }

        LOGI("Opened video: ", width_, "x", height_, " @ ", fps_, " fps, codec: ",
             avcodec_get_name(codecpar->codec_id));
    } else {
        // 不支持的 URI 形式
        LOGE("MppSource::open: Unsupported URI format (not RTSP or known video file extension): ", uri);
        return false;
    }

    // 校验拆包器是否成功初始化。
    if (!impl_->fmt_ctx) {
        LOGE("MppSource::open: Failed to initialize demuxer for URI: ", uri);
        return false;
    }

    // 初始化 MPP 解码器上下文。
    MPP_RET ret = mpp_create(&impl_->mpp_ctx, &impl_->mpi);
    if (ret != MPP_OK) {
        LOGE("mpp_create failed: ", ret);
        release();
        return false;
    }

    // split mode=1：让 parser 对输入码流做更稳健的帧切分。
    int need_split = 1;
    ret = impl_->mpi->control(impl_->mpp_ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);
    if (ret != MPP_OK) {
        LOGW("Failed to set parser split mode");
    }

    // 初始化解码器
    ret = mpp_init(impl_->mpp_ctx, MPP_CTX_DEC, coding_type);
    if (ret != MPP_OK) {
        LOGE("mpp_init failed: ", ret);
        release();
        return false;
    }

    // 尝试创建外部 DRM buffer group，成功时更有利于后续零拷贝传递。
    ret = mpp_buffer_group_get_external(&impl_->frame_group, MPP_BUFFER_TYPE_DRM);
    if (ret != MPP_OK) {
        // 回退到内部 buffer
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

    // 重试上限：防止极端流异常导致死循环。
    // 旧版本递归重试容易栈增长，这里改为 while 循环更安全。
    constexpr int MAX_DECODE_RETRIES = 30;  // 约等于 30fps 下 1 秒窗口
    int retry_count = 0;

    MppFrame mpp_frame = nullptr;

    // 主解码循环：读取包 -> 投喂解码器 -> 取帧。
    while (retry_count < MAX_DECODE_RETRIES) {
        // 安全检查：确保拆包器可用。
        if (!impl_->fmt_ctx) {
            LOGE("MppSource::read: Demuxer not initialized. Call open() first.");
            return false;
        }

        // 先从容器/流中读取一个 packet（可能是音频或视频）。
        while (true) {
            int ret = av_read_frame(impl_->fmt_ctx, impl_->pkt);
            if (ret < 0) {
                if (ret == AVERROR_EOF) {
                    // EOF 后需要进入 flush 阶段，把解码器内部缓存帧“榨干”。
                    if (impl_->eof_reached) {
                        // 已完成 flush，流真正结束。
                        return false;
                    }
                    impl_->eof_reached = true;
                    LOGI("End of stream - flushing decoder");

                    // 这里不再提供有效码流，后续 decode_get_frame 会持续取残留帧。
                    av_packet_unref(impl_->pkt);
                    // 继续进入解码循环，榨干缓冲帧。
                    break;
                } else {
                    // 其他读取错误
                    return false;
                }
            }

            // 非视频流包（音频等）直接丢弃。
            if (impl_->pkt->stream_index != impl_->video_stream_idx) {
                av_packet_unref(impl_->pkt);
                continue;
            }
            break;
        }

        // 把 AVPacket 包装成 MPP packet 后送入硬解码器。
        // EOF flush 路径：发送带 EOS 标记的空包触发解码器 drain；
        // 正常路径：包装真实码流数据。
        MppPacket mpp_pkt = nullptr;
        if (impl_->eof_reached) {
            // av_packet_unref 已被调用，data=nullptr/size=0。
            // MPP 需要显式 EOS 包才能 drain 内部缓冲帧。
            mpp_packet_init(&mpp_pkt, nullptr, 0);
            mpp_packet_set_eos(mpp_pkt, 1);
        } else {
            mpp_packet_init(&mpp_pkt, impl_->pkt->data, impl_->pkt->size);
            // 保留时间戳，便于后续做同步/统计（如果上层需要）。
            if (impl_->pkt->pts != AV_NOPTS_VALUE) {
                mpp_packet_set_pts(mpp_pkt, impl_->pkt->pts);
            }
        }

        // 投喂 packet 到解码器
        MPP_RET mpp_ret = impl_->mpi->decode_put_packet(impl_->mpp_ctx, mpp_pkt);
        mpp_packet_deinit(&mpp_pkt);
        av_packet_unref(impl_->pkt);

        if (mpp_ret != MPP_OK) {
            LOGW("decode_put_packet failed: ", mpp_ret);
            return false;
        }

        // 取解码帧：某些编解码器存在重排序，可能本轮拿不到帧。
        mpp_frame = nullptr;
        mpp_ret = impl_->mpi->decode_get_frame(impl_->mpp_ctx, &mpp_frame);

        if (mpp_ret != MPP_OK || !mpp_frame) {
            // 帧尚未就绪，继续喂包（例如 B 帧重排场景）。
            retry_count++;
            continue;  // 继续读下一个包
        }

        // 检查帧级解码错误
        if (mpp_frame_get_errinfo(mpp_frame)) {
            LOGW("Decode error in frame, retry ", retry_count + 1, "/", MAX_DECODE_RETRIES);
            mpp_frame_deinit(&mpp_frame);
            mpp_frame = nullptr;
            retry_count++;
            continue;  // 继续读下一个包
        }

        // 成功拿到有效帧，退出重试循环。
        break;
    }

    // 超过重试上限则判定失败。
    if (retry_count >= MAX_DECODE_RETRIES || !mpp_frame) {
        LOGW("Decode failed after ", retry_count, " retries");
        if (mpp_frame) {
            mpp_frame_deinit(&mpp_frame);
        }
        return false;
    }

    // 读取帧信息：width/height 是有效尺寸，stride 可能更大（内存对齐）。
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

    // 获取底层缓冲区地址（通常是 NV12/NV12-10bit）。
    void* buf_ptr = mpp_buffer_get_ptr(frm_buf);

    // DMA-BUF 模式：保存当前帧 fd（dup 一份，避免上游释放后句柄失效）。
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

    // 把 MPP 输出的 YUV（常见 NV12）转换为 OpenCV 的 BGR。
    if (frm_fmt == MPP_FMT_YUV420SP || frm_fmt == MPP_FMT_YUV420SP_10BIT) {
        bool converted = false;
#if RKNN_USE_RGA
        // 优先用 RGA 硬件色彩空间转换，能减少 CPU 开销。
        // 关键是把 stride 正确传入，否则会出现图像错位。
        rga_buffer_t src_buf = {};
        src_buf.width = frm_width;
        src_buf.height = frm_height;
        src_buf.wstride = frm_h_stride;  // 水平 stride（可能大于 width）
        src_buf.hstride = frm_v_stride;  // 垂直 stride（可能大于 height）
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
            // RGA 不可用或失败时回退到 CPU 路径，保证功能可用性。
            cv::Mat yuv_mat(frm_v_stride * 3 / 2, frm_h_stride, CV_8UC1, buf_ptr);

            // stride 与 width/height 不一致时，需要按行拷贝到紧凑内存布局。
            cv::Mat yuv_cropped;
            if (frm_h_stride != frm_width || frm_v_stride != frm_height) {
                // stride 不一致时按行拷贝为紧凑 NV12 布局。
                cv::Mat y_plane(frm_height, frm_width, CV_8UC1);
                cv::Mat uv_plane(frm_height / 2, frm_width / 2, CV_8UC2);

                // 按行拷贝 Y 平面
                for (int i = 0; i < frm_height; i++) {
                    memcpy(y_plane.ptr(i),
                           (uint8_t*)buf_ptr + i * frm_h_stride,
                           frm_width);
                }

                // 按行拷贝 UV 平面
                uint8_t* uv_src = (uint8_t*)buf_ptr + frm_h_stride * frm_v_stride;
                for (int i = 0; i < frm_height / 2; i++) {
                    memcpy(uv_plane.ptr(i),
                           uv_src + i * frm_h_stride,
                           frm_width);
                }

                // 重新拼成 OpenCV 期望的 NV12 连续布局后再做 cvtColor。
                yuv_cropped = cv::Mat(frm_height * 3 / 2, frm_width, CV_8UC1);
                y_plane.copyTo(yuv_cropped(cv::Rect(0, 0, frm_width, frm_height)));

                // 重排 UV 形状后拼接到 NV12 连续缓冲
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

    // 更新解码耗时统计（可用于监控平均解码延迟）。
    auto end_time = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    impl_->total_decode_time_ms += decode_ms;
    impl_->decode_count++;

    current_frame_++;
    return true;
}

ReadStatus MppSource::readFrameEx(CaptureFrame& frame) {
    frame.owner.reset();
    frame.mat.release();

    cv::Mat mat;
    if (read(mat)) {
        frame.mat = std::move(mat);
        return ReadStatus::FrameReady;
    }

    const bool stream_like = uri_.find("rtsp://") == 0 || uri_.find("rtmp://") == 0 ||
                             uri_.find("http://") == 0 || uri_.find("https://") == 0;
    if (stream_like && is_opened_.load()) {
        return ReadStatus::RecoverableError;
    }
    return ReadStatus::EndOfStream;
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
// MPP 扩展方法
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
    // 返回新的 dup fd：调用者拿到后可独立 close，不影响 MppSource 内部 fd。
    int dup_fd = dup(impl_->current_dma_fd);
    if (dup_fd < 0) {
        LOGW("Failed to dup current DMA-BUF fd: ", strerror(errno));
        return -1;
    }
    return dup_fd;
}

} // namespace rkapp::capture

#else  // !RKAPP_WITH_MPP

// MPP 不可用时的桩实现
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
ReadStatus MppSource::readFrameEx(CaptureFrame&) { return ReadStatus::FatalError; }
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
