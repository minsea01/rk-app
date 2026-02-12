#include "rkapp/capture/GigeSource.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <iostream>
#include <sstream>
#include <utility>

#include <opencv2/opencv.hpp>

#include "rkapp/common/StringUtils.hpp"
#include "rkapp/common/log.hpp"
#include "pixel_format_utils.hpp"

#ifdef RKAPP_WITH_GIGE
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
                               [](unsigned char ch) { return std::isspace(ch); }).base();
  if (first >= last) return {};
  return std::string(first, last);
}

bool parseInt(const std::string& input, int& out) {
  const auto trimmed = trimCopy(input);
  if (trimmed.empty()) return false;
  int value = 0;
  const char* begin = trimmed.data();
  const char* end = trimmed.data() + trimmed.size();
  if (auto [ptr, ec] = std::from_chars(begin, end, value); ec == std::errc{} && ptr == end) {
    out = value;
    return true;
  }
  return false;
}

bool parseBool(const std::string& input) {
  const auto lowered = rkapp::common::toLowerCopy(trimCopy(input));
  return lowered == "1" || lowered == "true" || lowered == "yes" ||
         lowered == "on" || lowered == "color" || lowered == "colour" ||
         lowered == "bgr" || lowered == "rgb";
}

using detail::PixelFormatKind;
using detail::bayerToBgrCode;
using detail::canonicalCapsFormat;
using detail::classifyPixelFormat;
using detail::isBayerFormat;
using detail::mediaTypeForFormat;
using detail::shouldUseVideoConvert;

#ifdef RKAPP_WITH_GIGE
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

GigeSource::~GigeSource() {
  try {
    release();
  } catch (...) {
    // destructors must not throw
  }
}

bool GigeSource::open(const std::string& uri) {
#ifdef RKAPP_WITH_GIGE
  std::lock_guard<std::mutex> lock(mtx_);
  destroyPipeline();
  opened_ = false;
  count_ = 0;
  consecutive_failures_ = 0;
  camera_uri_ = uri;

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
  fps_ = 30.0;
  count_ = 0;
  return true;
#else
  (void)uri;
  std::cerr << "GigeSource: 未启用 GIGE (Aravis)。请以 -DENABLE_GIGE=ON 并安装 aravis-0.8 后重编译。" << std::endl;
  opened_ = false;
  return false;
#endif
}

bool GigeSource::read(cv::Mat& frame) {
#ifdef RKAPP_WITH_GIGE
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

bool GigeSource::readFrame(CaptureFrame& frame) {
#ifdef RKAPP_WITH_GIGE
  frame.owner.reset();
  frame.mat.release();

  std::unique_lock<std::mutex> lock(mtx_);
  if (!opened_) return false;
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
    LOGW("GigeSource: pull_sample timeout, scheduling reconnect");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  sample_guard.buffer = gst_sample_get_buffer(sample_guard.sample);
  GstCaps* caps = gst_sample_get_caps(sample_guard.sample);
  if (!caps) {
    LOGE("GigeSource: missing caps in sample");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }
  const GstStructure* structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    LOGE("GigeSource: missing structure in caps");
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  int width = 0;
  int height = 0;
  if (!gst_structure_get_int(structure, "width", &width) ||
      !gst_structure_get_int(structure, "height", &height) ||
      width <= 0 || height <= 0 || width > 8192 || height > 8192) {
    LOGE("GigeSource: invalid frame dimensions: ", width, "x", height);
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  const gchar* fmt = gst_structure_get_string(structure, "format");
  if (!fmt) {
    LOGW("GigeSource: sample format unspecified, assuming BGR");
  }
  const std::string sample_format = fmt ? fmt : "BGR";
  PixelFormatKind pixel_kind = classifyPixelFormat(sample_format);
  if (!fmt) {
    pixel_kind = PixelFormatKind::BGR;
  } else if (pixel_kind == PixelFormatKind::UNKNOWN && !uri_config_.desired_format.empty()) {
    pixel_kind = classifyPixelFormat(uri_config_.desired_format);
  }
  if (pixel_kind == PixelFormatKind::UNKNOWN) {
    LOGW("GigeSource: unsupported sample format: ", sample_format);
    lock.lock();
    handleReadFailureLocked();
    return false;
  }

  if (!gst_buffer_map(sample_guard.buffer, &sample_guard.map, GST_MAP_READ)) {
    LOGW("GigeSource: failed to map buffer");
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
    LOGW("GigeSource: buffer stride smaller than expected");
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
          LOGW("GigeSource: unsupported Bayer format: ", sample_format);
          lock.lock();
          handleReadFailureLocked();
          return false;
        }
        cv::cvtColor(mapped, frame.mat, bayer_code);
      }
    } catch (const cv::Exception& e) {
      LOGW("GigeSource: color conversion failed for format ", sample_format,
           ": ", e.what());
      lock.lock();
      handleReadFailureLocked();
      return false;
    }
    frame.owner.reset();
  }

  lock.lock();
  consecutive_failures_ = 0;
  if (size_.width == 0) size_ = {width, height};
  ++count_;
  return true;
#else
  (void)frame;
  return false;
#endif
}

void GigeSource::release() {
#ifdef RKAPP_WITH_GIGE
  std::lock_guard<std::mutex> lock(mtx_);
  destroyPipeline();
#endif
  opened_ = false;
  count_ = 0;
  size_ = {0, 0};
}

GigeSource::UriConfig GigeSource::parseUri(const std::string& uri) {
  UriConfig config;
  bool format_explicit = false;
  bool color_requested = false;

  auto applyFormat = [&](const std::string& fmt) {
    const auto caps_fmt = canonicalCapsFormat(fmt);
    if (caps_fmt.empty()) return;
    const auto sanitized = sanitizeCaps("format=" + caps_fmt);
    if (sanitized.empty()) return;
    config.desired_format = caps_fmt;
    config.use_videoconvert = shouldUseVideoConvert(caps_fmt);
    format_explicit = true;
    config.caps_kv.erase(
        std::remove_if(config.caps_kv.begin(), config.caps_kv.end(),
                       [](const std::string& entry) {
                         return entry.rfind("format=", 0) == 0;
                       }),
        config.caps_kv.end());
    config.caps_kv.push_back("format=" + caps_fmt);
  };

  if (!uri.empty()) {
    size_t start = 0;
    while (start <= uri.size()) {
      size_t comma = uri.find(',', start);
      std::string token =
          (comma == std::string::npos) ? uri.substr(start) : uri.substr(start, comma - start);
      start = (comma == std::string::npos) ? uri.size() + 1 : comma + 1;
      token = trimCopy(token);
      if (token.empty()) continue;

      const size_t eq = token.find('=');
      if (eq == std::string::npos) {
        continue;
      }
      std::string key = trimCopy(token.substr(0, eq));
      std::string value = trimCopy(token.substr(eq + 1));
      if (key.empty()) continue;

      const std::string lowered = rkapp::common::toLowerCopy(key);
      if (lowered == "camera-name") {
        if (!value.empty()) config.camera_name = value;
        continue;
      }

      if (lowered == "pull-timeout-ms") {
        int parsed = 0;
        if (parseInt(value, parsed)) {
          parsed = std::clamp(parsed, 50, 5000);
          config.pull_timeout = std::chrono::milliseconds(parsed);
        }
        continue;
      }

      if (lowered == "max-failures" || lowered == "max-consecutive-failures") {
        int parsed = 0;
        if (parseInt(value, parsed)) {
          config.max_consecutive_failures = std::clamp(parsed, 1, 100);
        }
        continue;
      }

      if (lowered == "color" || lowered == "colour" || lowered == "mode") {
        color_requested = parseBool(value);
        continue;
      }

      if (lowered == "format") {
        if (!value.empty()) {
          applyFormat(value);
        }
        continue;
      }

      auto sanitized = sanitizeCaps(token);
      if (!sanitized.empty()) {
        config.caps_kv.push_back(std::move(sanitized));
      }
    }
  }

  if (!format_explicit) {
    config.caps_kv.erase(
        std::remove_if(config.caps_kv.begin(), config.caps_kv.end(),
                       [](const std::string& entry) {
                         return entry.rfind("format=", 0) == 0;
                       }),
        config.caps_kv.end());
    if (color_requested) {
      config.desired_format = "BGR";
      config.use_videoconvert = true;
      config.caps_kv.push_back("format=BGR");
    } else {
      config.desired_format = "GRAY8";
      config.use_videoconvert = false;
      config.caps_kv.push_back("format=GRAY8");
    }
  }

  std::vector<std::string> deduped;
  deduped.reserve(config.caps_kv.size());
  for (const auto& entry : config.caps_kv) {
    if (entry.empty()) continue;
    if (std::find(deduped.begin(), deduped.end(), entry) == deduped.end()) {
      deduped.push_back(entry);
    }
  }
  config.caps_kv = std::move(deduped);

  return config;
}

std::string GigeSource::buildPipelineDescription(const UriConfig& config) {
  const std::string media_type = mediaTypeForFormat(config.desired_format);
  std::string caps = media_type;
  for (const auto& kv : config.caps_kv) {
    auto sanitized = sanitizeCaps(kv);
    if (sanitized.empty()) continue;
    if (sanitized.rfind("video/", 0) == 0) continue;
    caps += ",";
    caps += sanitized;
  }

  std::ostringstream pipeline;
  pipeline << "aravissrc camera-name=\"" << sanitizeCameraName(config.camera_name) << "\" ! "
           << caps << " ! ";
  if (config.use_videoconvert) {
    pipeline << "videoconvert ! video/x-raw,format=" << config.desired_format
             << " ! appsink name=sink sync=false max-buffers=2 drop=true";
  } else {
    pipeline << "appsink name=sink caps=" << media_type << ",format=" << config.desired_format
             << " sync=false max-buffers=2 drop=true";
  }
  return pipeline.str();
}

std::string GigeSource::sanitizeCameraName(const std::string& name) {
  std::string safe;
  safe.reserve(name.size());
  for (char ch : name) {
    unsigned char c = static_cast<unsigned char>(ch);
    if (ch == '"' || ch == '\\') {
      safe.push_back('\\');
      safe.push_back(ch);
    } else if (std::isalnum(c) || ch == '-' || ch == '_' || ch == '.' || ch == ' ') {
      safe.push_back(ch);
    }
  }
  if (safe.empty()) safe = "camera";
  return safe;
}

std::string GigeSource::sanitizeCaps(const std::string& caps) {
  std::string safe;
  safe.reserve(caps.size());
  for (char ch : caps) {
    unsigned char c = static_cast<unsigned char>(ch);
    if (std::isalnum(c) || ch == '_' || ch == '-' || ch == ':' || ch == '/' ||
        ch == '=' || ch == ',' || ch == '.') {
      safe.push_back(ch);
    }
  }
  return safe;
}

bool GigeSource::isOpened() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return opened_;
}

double GigeSource::getFPS() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return fps_;
}

cv::Size GigeSource::getSize() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return size_;
}

int GigeSource::getTotalFrames() const { return 0; }

int GigeSource::getCurrentFrame() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return count_;
}

#ifdef RKAPP_WITH_GIGE
bool GigeSource::createPipeline() {
  destroyPipeline();

  GError* err = nullptr;
  pipeline_ = gst_parse_launch(pipeline_desc_.c_str(), &err);
  if (!pipeline_) {
    LOGE("GigeSource: 创建GStreamer管道失败: ", (err ? err->message : "unknown"));
    if (err) g_error_free(err);
    return false;
  }

  sink_element_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!sink_element_) {
    LOGE("GigeSource: 未找到appsink");
    destroyPipeline();
    return false;
  }

  appsink_ = GST_APP_SINK(sink_element_);
  gst_app_sink_set_emit_signals(appsink_, false);
  gst_app_sink_set_drop(appsink_, true);
  gst_app_sink_set_max_buffers(appsink_, 3);

  GstStateChangeReturn sret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (sret == GST_STATE_CHANGE_FAILURE) {
    LOGE("GigeSource: 管道无法进入PLAYING");
    destroyPipeline();
    return false;
  }

  // Bound the wait for READY->PLAYING to avoid hanging the capture loop
  GstStateChangeReturn wait_ret =
      gst_element_get_state(pipeline_, nullptr, nullptr, 2 * GST_SECOND);
  if (wait_ret != GST_STATE_CHANGE_SUCCESS && wait_ret != GST_STATE_CHANGE_NO_PREROLL) {
    LOGE("GigeSource: pipeline failed to reach PLAYING within timeout (ret=", wait_ret, ")");
    destroyPipeline();
    return false;
  }

  opened_ = true;
  consecutive_failures_ = 0;
  return true;
}

void GigeSource::destroyPipeline() {
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

bool GigeSource::ensurePipelineLocked() {
  if (pipeline_ && appsink_) {
    return true;
  }
  return attemptReconnectLocked();
}

bool GigeSource::checkPipelineHealthLocked() {
  if (!pipeline_) return false;
  GstBus* bus = gst_element_get_bus(pipeline_);
  if (!bus) return true;

  bool healthy = true;
  while (true) {
    GstMessage* msg = gst_bus_timed_pop_filtered(
        bus, 0, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (!msg) break;

    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_ERROR: {
        GError* err = nullptr;
        gchar* debug_info = nullptr;
        gst_message_parse_error(msg, &err, &debug_info);
        LOGW("GigeSource: pipeline error: ", (err ? err->message : "unknown"));
        if (debug_info) {
          LOGW("GigeSource: debug info: ", debug_info);
        }
        if (err) g_error_free(err);
        if (debug_info) g_free(debug_info);
        healthy = false;
        gst_message_unref(msg);
        break;
      }
      case GST_MESSAGE_EOS:
        LOGW("GigeSource: received EOS, restarting pipeline");
        healthy = false;
        gst_message_unref(msg);
        break;
      default:
        gst_message_unref(msg);
        break;
    }
    if (!healthy) break;
  }

  gst_object_unref(bus);
  return healthy;
}

bool GigeSource::attemptReconnectLocked() {
  auto now = std::chrono::steady_clock::now();
  if (now - last_reconnect_ < reconnect_backoff_) {
    return false;
  }

  last_reconnect_ = now;
  if (!createPipeline()) {
    reconnect_backoff_ = std::min(reconnect_backoff_ * 2, std::chrono::milliseconds(5000));
    opened_ = false;
    return false;
  }

  reconnect_backoff_ = std::chrono::milliseconds(500);
  opened_ = true;
  consecutive_failures_ = 0;
  return true;
}

void GigeSource::handleReadFailureLocked() {
  ++consecutive_failures_;
  if (consecutive_failures_ >= max_consecutive_failures_) {
    LOGW("GigeSource: consecutive failures reached threshold (", consecutive_failures_,
         "), forcing pipeline recreation");
    destroyPipeline();
    consecutive_failures_ = 0;
  }
  attemptReconnectLocked();
}
#endif  // RKAPP_WITH_GIGE

}  // namespace rkapp::capture
