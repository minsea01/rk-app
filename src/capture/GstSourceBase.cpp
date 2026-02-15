#include "rkapp/capture/GstSourceBase.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include "rkapp/common/log.hpp"
#include "pixel_format_utils.hpp"

#if defined(RKAPP_WITH_GIGE) || defined(RKAPP_WITH_CSI)
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/gstcaps.h>
#include <gst/gstmessage.h>

#if !GST_CHECK_VERSION(1, 14, 0)
#define gst_app_sink_try_pull_sample(s, t) gst_app_sink_pull_sample(s)
#endif
#endif

namespace rkapp::capture {

#if defined(RKAPP_WITH_GIGE) || defined(RKAPP_WITH_CSI)
namespace {

using detail::PixelFormatKind;
using detail::bayerToBgrCode;
using detail::classifyPixelFormat;
using detail::isBayerFormat;

struct GstSampleHolder {
  GstSample* sample = nullptr;
  GstBuffer* buffer = nullptr;
  GstMapInfo map{};
  bool mapped = false;

  ~GstSampleHolder() {
    if (mapped && buffer) {
      gst_buffer_unmap(buffer, &map);
    }
    if (sample) {
      gst_sample_unref(sample);
    }
  }
};

struct GstSampleGuard {
  GstAppSink* sink = nullptr;
  GstSample* sample = nullptr;
  GstBuffer* buffer = nullptr;
  GstMapInfo map{};
  bool mapped = false;

  ~GstSampleGuard() {
    if (mapped && buffer) {
      gst_buffer_unmap(buffer, &map);
    }
    if (sample) {
      gst_sample_unref(sample);
    }
    if (sink) {
      gst_object_unref(sink);
    }
  }
};

constexpr std::chrono::milliseconds kInitialReconnectBackoff(500);
constexpr std::chrono::milliseconds kMaxReconnectBackoff(5000);

}  // namespace
#endif

GstSourceBase::GstSourceBase(const char* source_name)
    : source_name_(source_name ? source_name : "GstSource") {}

bool GstSourceBase::openWithConfig(const OpenConfig& config) {
#if defined(RKAPP_WITH_GIGE) || defined(RKAPP_WITH_CSI)
  std::lock_guard<std::mutex> lock(mtx_);
  destroyPipelineLocked();
  opened_ = false;
  count_ = 0;
  consecutive_failures_ = 0;

  source_uri_ = config.uri;
  pipeline_desc_ = config.pipeline_desc;
  pull_timeout_ = config.pull_timeout;
  max_consecutive_failures_ = std::max(1, config.max_consecutive_failures);
  reconnect_backoff_ = kInitialReconnectBackoff;
  last_reconnect_ = std::chrono::steady_clock::now() - reconnect_backoff_;
  unknown_format_fallback_ = config.unknown_format_fallback;
  reconnect_immediately_on_failure_ = config.reconnect_immediately_on_failure;

  size_ = {0, 0};
  fps_ = config.fps;

  if (!gst_is_initialized()) {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
  }

  if (!createPipelineLocked()) {
    opened_ = false;
    return false;
  }

  opened_ = true;
  count_ = 0;
  return true;
#else
  (void)config;
  std::lock_guard<std::mutex> lock(mtx_);
  opened_ = false;
  count_ = 0;
  size_ = {0, 0};
  return false;
#endif
}

bool GstSourceBase::read(cv::Mat& frame) {
  CaptureFrame capture;
  if (!readFrame(capture)) {
    return false;
  }
  capture.mat.copyTo(frame);
  return true;
}

bool GstSourceBase::readFrame(CaptureFrame& frame) {
  frame.owner.reset();
  frame.mat.release();

#if defined(RKAPP_WITH_GIGE) || defined(RKAPP_WITH_CSI)
  std::unique_lock<std::mutex> lock(mtx_);
  if (!opened_) {
    return false;
  }
  if (!ensurePipelineLocked()) {
    return false;
  }

  if (!checkPipelineHealthLocked()) {
    if (!attemptReconnectLocked()) {
      return false;
    }
    if (!ensurePipelineLocked()) {
      return false;
    }
  }

  GstSampleGuard sample_guard;
  sample_guard.sink = GST_APP_SINK(gst_object_ref(appsink_));
  const auto pull_timeout = pull_timeout_;
  lock.unlock();

  sample_guard.sample =
      gst_app_sink_try_pull_sample(sample_guard.sink, pull_timeout.count() * GST_MSECOND);
  if (!sample_guard.sample) {
    LOGW(source_name_, ": pull_sample timeout, scheduling reconnect");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  sample_guard.buffer = gst_sample_get_buffer(sample_guard.sample);
  if (!sample_guard.buffer) {
    LOGW(source_name_, ": sample has no GstBuffer");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  GstCaps* caps = gst_sample_get_caps(sample_guard.sample);
  if (!caps) {
    LOGW(source_name_, ": sample has no caps");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  const GstStructure* structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    LOGW(source_name_, ": sample caps has no structure");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  int width = 0;
  int height = 0;
  if (!gst_structure_get_int(structure, "width", &width) ||
      !gst_structure_get_int(structure, "height", &height) || width <= 0 || height <= 0 ||
      width > 8192 || height > 8192) {
    LOGW(source_name_, ": invalid frame dimensions: ", width, "x", height);
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  const gchar* fmt = gst_structure_get_string(structure, "format");
  std::string sample_format = fmt ? fmt : "BGR";
  PixelFormatKind pixel_kind = classifyPixelFormat(sample_format);
  if (!fmt) {
    pixel_kind = PixelFormatKind::BGR;
  }

  if (pixel_kind == PixelFormatKind::UNKNOWN) {
    lock.lock();
    const std::string fallback = unknown_format_fallback_;
    lock.unlock();
    if (!fallback.empty()) {
      const PixelFormatKind fallback_kind = classifyPixelFormat(fallback);
      if (fallback_kind != PixelFormatKind::UNKNOWN) {
        sample_format = fallback;
        pixel_kind = fallback_kind;
      }
    }
  }

  if (pixel_kind == PixelFormatKind::UNKNOWN) {
    LOGW(source_name_, ": unsupported sample format: ", sample_format);
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  if (!gst_buffer_map(sample_guard.buffer, &sample_guard.map, GST_MAP_READ)) {
    LOGW(source_name_, ": failed to map buffer");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }
  sample_guard.mapped = true;

  int src_channels = 3;
  int src_type = CV_8UC3;
  switch (pixel_kind) {
    case PixelFormatKind::GRAY:
    case PixelFormatKind::BAYER_RG:
    case PixelFormatKind::BAYER_BG:
    case PixelFormatKind::BAYER_GR:
    case PixelFormatKind::BAYER_GB:
      src_channels = 1;
      src_type = CV_8UC1;
      break;
    case PixelFormatKind::BGRA:
    case PixelFormatKind::RGBA:
      src_channels = 4;
      src_type = CV_8UC4;
      break;
    case PixelFormatKind::RGB:
    case PixelFormatKind::BGR:
    default:
      src_channels = 3;
      src_type = CV_8UC3;
      break;
  }

  const size_t row_stride =
      static_cast<size_t>(sample_guard.map.size) / static_cast<size_t>(height);
  const size_t expected_stride = static_cast<size_t>(width) * static_cast<size_t>(src_channels);
  if (row_stride < expected_stride) {
    LOGW(source_name_, ": buffer stride smaller than expected");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  cv::Mat mapped(height, width, src_type, static_cast<void*>(sample_guard.map.data), row_stride);
  if (pixel_kind == PixelFormatKind::BGR) {
    auto holder = std::make_shared<GstSampleHolder>();
    holder->sample = sample_guard.sample;
    holder->buffer = sample_guard.buffer;
    holder->map = sample_guard.map;
    holder->mapped = sample_guard.mapped;
    sample_guard.sample = nullptr;
    sample_guard.buffer = nullptr;
    sample_guard.mapped = false;
    frame.mat = mapped;
    frame.owner = holder;
  } else {
    try {
      if (pixel_kind == PixelFormatKind::GRAY) {
        cv::cvtColor(mapped, frame.mat, cv::COLOR_GRAY2BGR);
      } else if (pixel_kind == PixelFormatKind::RGB) {
        cv::cvtColor(mapped, frame.mat, cv::COLOR_RGB2BGR);
      } else if (pixel_kind == PixelFormatKind::BGRA) {
        cv::cvtColor(mapped, frame.mat, cv::COLOR_BGRA2BGR);
      } else if (pixel_kind == PixelFormatKind::RGBA) {
        cv::cvtColor(mapped, frame.mat, cv::COLOR_RGBA2BGR);
      } else if (isBayerFormat(pixel_kind)) {
        const int bayer_code = bayerToBgrCode(pixel_kind);
        if (bayer_code < 0) {
          LOGW(source_name_, ": unsupported Bayer format: ", sample_format);
          lock.lock();
          handleReadFailureLocked();
          return false;
        }
        cv::cvtColor(mapped, frame.mat, bayer_code);
      }
    } catch (const cv::Exception& e) {
      LOGW(source_name_, ": color conversion failed for format ", sample_format, ": ", e.what());
      lock.lock();
      handleReadFailureLocked();
      return false;
    }
    frame.owner.reset();
  }

  lock.lock();
  consecutive_failures_ = 0;
  if (size_.width == 0) {
    size_ = {width, height};
  }
  ++count_;
  return true;
#else
  return false;
#endif
}

void GstSourceBase::release() {
  std::lock_guard<std::mutex> lock(mtx_);

#if defined(RKAPP_WITH_GIGE) || defined(RKAPP_WITH_CSI)
  destroyPipelineLocked();
  source_uri_.clear();
  pipeline_desc_.clear();
  reconnect_backoff_ = kInitialReconnectBackoff;
  last_reconnect_ = {};
  pull_timeout_ = std::chrono::milliseconds(200);
  max_consecutive_failures_ = 5;
  consecutive_failures_ = 0;
  unknown_format_fallback_.clear();
  reconnect_immediately_on_failure_ = true;
#endif

  opened_ = false;
  count_ = 0;
  size_ = {0, 0};
  fps_ = 30.0;
}

bool GstSourceBase::isOpened() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return opened_;
}

double GstSourceBase::getFPS() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return fps_;
}

cv::Size GstSourceBase::getSize() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return size_;
}

int GstSourceBase::getTotalFrames() const { return 0; }

int GstSourceBase::getCurrentFrame() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return count_;
}

#if defined(RKAPP_WITH_GIGE) || defined(RKAPP_WITH_CSI)
bool GstSourceBase::createPipelineLocked() {
  destroyPipelineLocked();

  GError* err = nullptr;
  pipeline_ = gst_parse_launch(pipeline_desc_.c_str(), &err);
  if (!pipeline_) {
    LOGE(source_name_, ": failed to create GStreamer pipeline: ",
         (err ? err->message : "unknown"));
    if (err) {
      g_error_free(err);
    }
    return false;
  }

  sink_element_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!sink_element_) {
    LOGE(source_name_, ": appsink element not found");
    destroyPipelineLocked();
    return false;
  }

  appsink_ = GST_APP_SINK(sink_element_);
  gst_app_sink_set_emit_signals(appsink_, false);
  gst_app_sink_set_drop(appsink_, true);
  gst_app_sink_set_max_buffers(appsink_, 3);

  const GstStateChangeReturn state_ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (state_ret == GST_STATE_CHANGE_FAILURE) {
    LOGE(source_name_, ": pipeline failed to enter PLAYING state");
    destroyPipelineLocked();
    return false;
  }

  const GstStateChangeReturn wait_ret =
      gst_element_get_state(pipeline_, nullptr, nullptr, 2 * GST_SECOND);
  if (wait_ret != GST_STATE_CHANGE_SUCCESS && wait_ret != GST_STATE_CHANGE_NO_PREROLL) {
    LOGE(source_name_, ": pipeline failed to reach PLAYING within timeout (ret=", wait_ret,
         ")");
    destroyPipelineLocked();
    return false;
  }

  opened_ = true;
  consecutive_failures_ = 0;
  return true;
}

void GstSourceBase::destroyPipelineLocked() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_element_get_state(pipeline_, nullptr, nullptr, 1 * GST_SECOND);
  }
  if (sink_element_) {
    gst_object_unref(sink_element_);
    sink_element_ = nullptr;
  }
  appsink_ = nullptr;
  if (pipeline_) {
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  opened_ = false;
}

bool GstSourceBase::ensurePipelineLocked() {
  if (pipeline_ && appsink_) {
    return true;
  }
  return attemptReconnectLocked();
}

bool GstSourceBase::checkPipelineHealthLocked() {
  if (!pipeline_) {
    return false;
  }

  GstBus* bus = gst_element_get_bus(pipeline_);
  if (!bus) {
    return true;
  }

  bool healthy = true;
  while (true) {
    GstMessage* msg = gst_bus_timed_pop_filtered(
        bus, 0, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (!msg) {
      break;
    }

    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_ERROR: {
        GError* err = nullptr;
        gchar* debug_info = nullptr;
        gst_message_parse_error(msg, &err, &debug_info);
        LOGW(source_name_, ": pipeline error: ", (err ? err->message : "unknown"));
        if (debug_info) {
          LOGW(source_name_, ": debug info: ", debug_info);
        }
        if (err) {
          g_error_free(err);
        }
        if (debug_info) {
          g_free(debug_info);
        }
        healthy = false;
        gst_message_unref(msg);
        break;
      }
      case GST_MESSAGE_EOS:
        LOGW(source_name_, ": pipeline reached EOS");
        healthy = false;
        gst_message_unref(msg);
        break;
      default:
        gst_message_unref(msg);
        break;
    }

    if (!healthy) {
      break;
    }
  }

  gst_object_unref(bus);
  return healthy;
}

bool GstSourceBase::attemptReconnectLocked() {
  const auto now = std::chrono::steady_clock::now();
  if (now - last_reconnect_ < reconnect_backoff_) {
    return false;
  }

  last_reconnect_ = now;
  LOGW(source_name_, ": reconnecting with backoff=", reconnect_backoff_.count(), " ms");
  destroyPipelineLocked();
  if (!createPipelineLocked()) {
    reconnect_backoff_ = std::min(reconnect_backoff_ * 2, kMaxReconnectBackoff);
    opened_ = false;
    return false;
  }

  reconnect_backoff_ = kInitialReconnectBackoff;
  opened_ = true;
  consecutive_failures_ = 0;
  return true;
}

void GstSourceBase::handleReadFailureLocked() {
  ++consecutive_failures_;
  if (consecutive_failures_ >= max_consecutive_failures_) {
    LOGW(source_name_, ": too many failures (", consecutive_failures_,
         "), forcing pipeline restart");
    destroyPipelineLocked();
    consecutive_failures_ = 0;
  }

  if (reconnect_immediately_on_failure_) {
    attemptReconnectLocked();
  }
}
#endif

}  // namespace rkapp::capture
