#include "rkapp/capture/CsiSource.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <iostream>
#include <sstream>

#include "rkapp/common/StringUtils.hpp"
#include "pixel_format_utils.hpp"

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

using detail::canonicalCapsFormat;
using detail::mediaTypeForFormat;
using detail::shouldUseVideoConvert;

}  // namespace

CsiSource::CsiSource() : GstSourceBase("CsiSource") {}

CsiSource::~CsiSource() {
  try {
    release();
  } catch (...) {
    // Destructors must not throw.
  }
}

bool CsiSource::open(const std::string& uri) {
#ifdef RKAPP_WITH_CSI
  const UriConfig uri_config = parseUri(uri);

  OpenConfig config;
  config.uri = uri;
  config.pipeline_desc = buildPipelineDescription(uri_config);
  config.pull_timeout = uri_config.pull_timeout;
  config.max_consecutive_failures = uri_config.max_consecutive_failures;
  config.fps = static_cast<double>(uri_config.framerate);
  config.reconnect_immediately_on_failure = false;

  return openWithConfig(config);
#else
  (void)uri;
  std::cerr << "CsiSource: CSI source is not enabled. Rebuild with -DENABLE_CSI=ON."
            << std::endl;
  release();
  return false;
#endif
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

      const std::string key = rkapp::common::toLowerCopy(trimCopy(token.substr(0, eq)));
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

}  // namespace rkapp::capture
