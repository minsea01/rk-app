#include "rkapp/capture/CsiSource.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <iostream>
#include <sstream>
#include <thread>
#include <utility>

#include <opencv2/opencv.hpp>

#include "log.hpp"
#include "pixel_format_utils.hpp"

#ifdef RKAPP_WITH_CSI
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/gstcaps.h>
#include <gst/gstmessage.h>

#if !GST_CHECK_VERSION(1, 14, 0)
#define gst_app_sink_try_pull_sample(s, t) gst_app_sink_pull_sample(s)
#endif
#endif

namespace rkapp::capture {

namespace {

std::string trimCopy(const std::string& input) {
  auto first = std::find_if_not(input.begin(), input.end(),
                                [](unsigned char ch) { return std::isspace(ch); });
  auto last = std::find_if_not(input.rbegin(), input.rend(),
                               [](unsigned char ch) { return std::isspace(ch); })
                  .base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

std::string toLowerCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool parseInt(const std::string& input, int& out) {
  const auto trimmed = trimCopy(input);
  if (trimmed.empty()) {
    return false;
  }
  int value = 0;
  const char* begin = trimmed.data();
  const char* end = trimmed.data() + trimmed.size();
  if (auto [ptr, ec] = std::from_chars(begin, end, value); ec == std::errc{} && ptr == end) {
    out = value;
    return true;
  }
  return false;
}

int parseFramerate(const std::string& value, int fallback) {
  int parsed = 0;
  if (parseInt(value, parsed)) {
    return std::max(1, parsed);
  }
  const size_t slash = value.find('/');
  if (slash == std::string::npos) {
    return fallback;
  }
  int num = 0;
  int den = 1;
  if (!parseInt(value.substr(0, slash), num) || !parseInt(value.substr(slash + 1), den)) {
    return fallback;
  }
  if (den <= 0) {
    return fallback;
  }
  return std::max(1, num / den);
}

using detail::PixelFormatKind;
using detail::bayerToBgrCode;
using detail::canonicalCapsFormat;
using detail::classifyPixelFormat;
using detail::isBayerFormat;
using detail::mediaTypeForFormat;
using detail::shouldUseVideoConvert;

#ifdef RKAPP_WITH_CSI
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
#endif

}  // namespace

CsiSource::~CsiSource() {
  try {
    release();
  } catch (...) {
    // Destructors must not throw.
  }
}

bool CsiSource::open(const std::string& uri) {
#ifdef RKAPP_WITH_CSI
  std::lock_guard<std::mutex> lock(mtx_);
  destroyPipeline();
  opened_ = false;
  count_ = 0;
  consecutive_failures_ = 0;
  device_uri_ = uri;

  if (!gst_is_initialized()) {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
  }

  uri_config_ = parseUri(uri);
  pipeline_desc_ = buildPipelineDescription(uri_config_);
  pull_timeout_ = uri_config_.pull_timeout;
  max_consecutive_failures_ = std::max(1, uri_config_.max_consecutive_failures);
  reconnect_backoff_ = std::chrono::milliseconds(500);
  last_reconnect_ = std::chrono::steady_clock::now() - reconnect_backoff_;

  if (!createPipeline()) {
    opened_ = false;
    return false;
  }

  opened_ = true;
  size_ = {0, 0};
  fps_ = static_cast<double>(uri_config_.framerate);
  count_ = 0;
  return true;
#else
  (void)uri;
  std::cerr << "CsiSource: CSI source is not enabled. Rebuild with -DENABLE_CSI=ON."
            << std::endl;
  opened_ = false;
  return false;
#endif
}

bool CsiSource::read(cv::Mat& frame) {
#ifdef RKAPP_WITH_CSI
  CaptureFrame capture;
  if (!readFrame(capture)) {
    return false;
  }
  capture.mat.copyTo(frame);
  return true;
#else
  (void)frame;
  return false;
#endif
}

bool CsiSource::readFrame(CaptureFrame& frame) {
#ifdef RKAPP_WITH_CSI
  frame.owner.reset();
  frame.mat.release();

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
  auto pull_timeout = pull_timeout_;
  lock.unlock();

  sample_guard.sample =
      gst_app_sink_try_pull_sample(sample_guard.sink, pull_timeout.count() * GST_MSECOND);
  if (!sample_guard.sample) {
    LOGW("CsiSource: pull_sample timeout, scheduling reconnect");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  sample_guard.buffer = gst_sample_get_buffer(sample_guard.sample);
  if (!sample_guard.buffer) {
    LOGW("CsiSource: sample has no GstBuffer");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  GstCaps* caps = gst_sample_get_caps(sample_guard.sample);
  if (!caps) {
    LOGW("CsiSource: sample has no caps");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  const GstStructure* structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    LOGW("CsiSource: sample caps has no structure");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  int width = 0;
  int height = 0;
  if (!gst_structure_get_int(structure, "width", &width) ||
      !gst_structure_get_int(structure, "height", &height) || width <= 0 || height <= 0 ||
      width > 8192 || height > 8192) {
    LOGW("CsiSource: invalid frame dimensions: ", width, "x", height);
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  const gchar* fmt = gst_structure_get_string(structure, "format");
  const std::string sample_format = fmt ? fmt : "BGR";
  PixelFormatKind pixel_kind = classifyPixelFormat(sample_format);
  if (!fmt) {
    pixel_kind = PixelFormatKind::BGR;
  }
  if (pixel_kind == PixelFormatKind::UNKNOWN) {
    LOGW("CsiSource: unsupported sample format: ", sample_format);
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  if (!gst_buffer_map(sample_guard.buffer, &sample_guard.map, GST_MAP_READ)) {
    LOGW("CsiSource: failed to map buffer");
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
    LOGW("CsiSource: buffer stride smaller than expected");
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
          LOGW("CsiSource: unsupported Bayer format: ", sample_format);
          lock.lock();
          handleReadFailureLocked();
          return false;
        }
        cv::cvtColor(mapped, frame.mat, bayer_code);
      }
    } catch (const cv::Exception& e) {
      LOGW("CsiSource: color conversion failed for format ", sample_format, ": ", e.what());
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
  (void)frame;
  return false;
#endif
}

void CsiSource::release() {
#ifdef RKAPP_WITH_CSI
  std::lock_guard<std::mutex> lock(mtx_);
  destroyPipeline();
#endif
  opened_ = false;
  count_ = 0;
  size_ = {0, 0};
}

CsiSource::UriConfig CsiSource::parseUri(const std::string& uri) {
  UriConfig config;

  auto applyFormat = [&](const std::string& format) {
    const std::string normalized = canonicalCapsFormat(format);
    if (normalized.empty()) {
      return;
    }
    config.format = normalized;
    config.use_videoconvert = shouldUseVideoConvert(normalized);
  };

  if (!uri.empty()) {
    size_t start = 0;
    while (start <= uri.size()) {
      const size_t comma = uri.find(',', start);
      std::string token =
          (comma == std::string::npos) ? uri.substr(start) : uri.substr(start, comma - start);
      start = (comma == std::string::npos) ? uri.size() + 1 : comma + 1;
      token = trimCopy(token);
      if (token.empty()) {
        continue;
      }

      const size_t eq = token.find('=');
      if (eq == std::string::npos) {
        continue;
      }

      const std::string key = toLowerCopy(trimCopy(token.substr(0, eq)));
      const std::string value = trimCopy(token.substr(eq + 1));
      if (key.empty()) {
        continue;
      }

      if (key == "device") {
        if (!value.empty()) {
          config.device = sanitizeDevicePath(value);
        }
        continue;
      }
      if (key == "width") {
        int parsed = 0;
        if (parseInt(value, parsed)) {
          config.width = std::clamp(parsed, 32, 8192);
        }
        continue;
      }
      if (key == "height") {
        int parsed = 0;
        if (parseInt(value, parsed)) {
          config.height = std::clamp(parsed, 32, 8192);
        }
        continue;
      }
      if (key == "framerate" || key == "fps") {
        config.framerate = std::clamp(parseFramerate(value, config.framerate), 1, 240);
        continue;
      }
      if (key == "format") {
        if (!value.empty()) {
          applyFormat(value);
        }
        continue;
      }
      if (key == "pull-timeout-ms") {
        int parsed = 0;
        if (parseInt(value, parsed)) {
          config.pull_timeout = std::chrono::milliseconds(std::clamp(parsed, 50, 5000));
        }
        continue;
      }
      if (key == "max-failures" || key == "max-consecutive-failures") {
        int parsed = 0;
        if (parseInt(value, parsed)) {
          config.max_consecutive_failures = std::clamp(parsed, 1, 100);
        }
        continue;
      }
    }
  }

  if (config.device.empty()) {
    config.device = "/dev/video0";
  }
  if (config.format.empty()) {
    config.format = "NV12";
  }
  config.use_videoconvert = shouldUseVideoConvert(config.format);
  return config;
}

std::string CsiSource::buildPipelineDescription(const UriConfig& config) {
  const std::string media_type = mediaTypeForFormat(config.format);
  const std::string safe_device = sanitizeDevicePath(config.device);
  const std::string safe_format = sanitizeCapsToken(config.format);
  const int safe_width = std::clamp(config.width, 32, 8192);
  const int safe_height = std::clamp(config.height, 32, 8192);
  const int safe_fps = std::clamp(config.framerate, 1, 240);

  std::ostringstream pipeline;
  pipeline << "v4l2src device=" << safe_device << " ! "
           << media_type << ",format=" << (safe_format.empty() ? "NV12" : safe_format)
           << ",width=" << safe_width << ",height=" << safe_height << ",framerate=" << safe_fps
           << "/1 ! ";

  if (config.use_videoconvert) {
    pipeline << "videoconvert ! video/x-raw,format=BGR ! ";
    pipeline << "appsink name=sink sync=false max-buffers=2 drop=true";
  } else {
    pipeline << "appsink name=sink caps=" << media_type << ",format="
             << (safe_format.empty() ? "NV12" : safe_format)
             << " sync=false max-buffers=2 drop=true";
  }
  return pipeline.str();
}

std::string CsiSource::sanitizeDevicePath(const std::string& value) {
  std::string safe;
  safe.reserve(value.size());
  for (char ch : value) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (std::isalnum(c) || ch == '/' || ch == '_' || ch == '-' || ch == '.' || ch == ':') {
      safe.push_back(ch);
    }
  }
  if (safe.empty()) {
    return "/dev/video0";
  }
  return safe;
}

std::string CsiSource::sanitizeCapsToken(const std::string& value) {
  std::string safe;
  safe.reserve(value.size());
  for (char ch : value) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (std::isalnum(c) || ch == '_' || ch == '-') {
      safe.push_back(ch);
    }
  }
  return safe;
}

bool CsiSource::isOpened() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return opened_;
}

double CsiSource::getFPS() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return fps_;
}

cv::Size CsiSource::getSize() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return size_;
}

int CsiSource::getTotalFrames() const { return 0; }

int CsiSource::getCurrentFrame() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return count_;
}

#ifdef RKAPP_WITH_CSI
bool CsiSource::createPipeline() {
  destroyPipeline();

  GError* err = nullptr;
  pipeline_ = gst_parse_launch(pipeline_desc_.c_str(), &err);
  if (!pipeline_) {
    LOGE("CsiSource: failed to create GStreamer pipeline: ", (err ? err->message : "unknown"));
    if (err) {
      g_error_free(err);
    }
    return false;
  }

  sink_element_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!sink_element_) {
    LOGE("CsiSource: appsink element not found");
    destroyPipeline();
    return false;
  }

  appsink_ = GST_APP_SINK(sink_element_);
  gst_app_sink_set_emit_signals(appsink_, false);
  gst_app_sink_set_drop(appsink_, true);
  gst_app_sink_set_max_buffers(appsink_, 3);

  const GstStateChangeReturn state_ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (state_ret == GST_STATE_CHANGE_FAILURE) {
    LOGE("CsiSource: pipeline failed to enter PLAYING state");
    destroyPipeline();
    return false;
  }

  const GstStateChangeReturn wait_ret =
      gst_element_get_state(pipeline_, nullptr, nullptr, 2 * GST_SECOND);
  if (wait_ret != GST_STATE_CHANGE_SUCCESS && wait_ret != GST_STATE_CHANGE_NO_PREROLL) {
    LOGE("CsiSource: pipeline failed to reach PLAYING within timeout (ret=", wait_ret, ")");
    destroyPipeline();
    return false;
  }

  opened_ = true;
  consecutive_failures_ = 0;
  return true;
}

void CsiSource::destroyPipeline() {
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

bool CsiSource::ensurePipelineLocked() {
  if (pipeline_ && appsink_) {
    return true;
  }
  return attemptReconnectLocked();
}

bool CsiSource::checkPipelineHealthLocked() {
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
        LOGW("CsiSource: pipeline error: ", (err ? err->message : "unknown"));
        if (debug_info) {
          LOGW("CsiSource: debug info: ", debug_info);
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
        LOGW("CsiSource: pipeline reached EOS");
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

bool CsiSource::attemptReconnectLocked() {
  const auto now = std::chrono::steady_clock::now();
  if (now - last_reconnect_ < reconnect_backoff_) {
    return false;
  }
  last_reconnect_ = now;

  LOGW("CsiSource: reconnecting with backoff=", reconnect_backoff_.count(), " ms");
  destroyPipeline();
  if (!createPipeline()) {
    reconnect_backoff_ = std::min(reconnect_backoff_ * 2, std::chrono::milliseconds(5000));
    return false;
  }

  reconnect_backoff_ = std::chrono::milliseconds(500);
  consecutive_failures_ = 0;
  return true;
}

void CsiSource::handleReadFailureLocked() {
  ++consecutive_failures_;
  if (consecutive_failures_ >= max_consecutive_failures_) {
    LOGW("CsiSource: too many failures (", consecutive_failures_,
         "), forcing pipeline restart");
    destroyPipeline();
    consecutive_failures_ = 0;
  }
}
#endif

}  // namespace rkapp::capture
