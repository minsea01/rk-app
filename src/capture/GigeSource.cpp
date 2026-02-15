#include "rkapp/capture/GigeSource.hpp"

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

using detail::canonicalCapsFormat;
using detail::mediaTypeForFormat;
using detail::shouldUseVideoConvert;

}  // namespace

GigeSource::GigeSource() : GstSourceBase("GigeSource") {}

GigeSource::~GigeSource() {
  try {
    release();
  } catch (...) {
    // Destructors must not throw.
  }
}

bool GigeSource::open(const std::string& uri) {
#ifdef RKAPP_WITH_GIGE
  const UriConfig uri_config = parseUri(uri);

  OpenConfig config;
  config.uri = uri;
  config.pipeline_desc = buildPipelineDescription(uri_config);
  config.pull_timeout = uri_config.pull_timeout;
  config.max_consecutive_failures = uri_config.max_consecutive_failures;
  config.fps = 30.0;
  config.unknown_format_fallback = uri_config.desired_format;
  config.reconnect_immediately_on_failure = true;

  return openWithConfig(config);
#else
  (void)uri;
  std::cerr << "GigeSource: 未启用 GIGE (Aravis)。请以 -DENABLE_GIGE=ON 并安装 aravis-0.8 后重编译。"
            << std::endl;
  release();
  return false;
#endif
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

}  // namespace rkapp::capture
