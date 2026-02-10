#pragma once

#include <algorithm>
#include <cctype>
#include <string>

#include <opencv2/opencv.hpp>

namespace rkapp::capture::detail {

enum class PixelFormatKind {
  BGR,
  RGB,
  BGRA,
  RGBA,
  GRAY,
  BAYER_RG,
  BAYER_BG,
  BAYER_GR,
  BAYER_GB,
  UNKNOWN
};

inline std::string trimCopy(const std::string& input) {
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

inline std::string toUpperCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

inline std::string normalizeFormatToken(std::string value) {
  value = toUpperCopy(trimCopy(value));
  value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch) {
                return std::isspace(ch) || ch == '_' || ch == '-';
              }),
              value.end());
  return value;
}

inline PixelFormatKind classifyPixelFormat(const std::string& format) {
  const std::string token = normalizeFormatToken(format);
  if (token.empty()) {
    return PixelFormatKind::UNKNOWN;
  }

  if (token == "GRAY8" || token == "MONO8" || token == "Y8") {
    return PixelFormatKind::GRAY;
  }
  if (token == "BGR" || token == "BGR8" || token == "BGR888") {
    return PixelFormatKind::BGR;
  }
  if (token == "RGB" || token == "RGB8" || token == "RGB888") {
    return PixelFormatKind::RGB;
  }
  if (token == "BGRA" || token == "BGRA8" || token == "BGRA8888" || token == "BGRX" ||
      token == "BGRX8") {
    return PixelFormatKind::BGRA;
  }
  if (token == "RGBA" || token == "RGBA8" || token == "RGBA8888" || token == "RGBX" ||
      token == "RGBX8") {
    return PixelFormatKind::RGBA;
  }
  if (token == "RGGB" || token == "RGGB8" || token == "BAYERRG8" || token == "BAYERRG" ||
      token == "BAYERRGGB" || token == "BAYERRGGB8") {
    return PixelFormatKind::BAYER_RG;
  }
  if (token == "BGGR" || token == "BGGR8" || token == "BAYERBG8" || token == "BAYERBG" ||
      token == "BAYERBGGR" || token == "BAYERBGGR8") {
    return PixelFormatKind::BAYER_BG;
  }
  if (token == "GRBG" || token == "GRBG8" || token == "BAYERGR8" || token == "BAYERGR" ||
      token == "BAYERGRBG" || token == "BAYERGRBG8") {
    return PixelFormatKind::BAYER_GR;
  }
  if (token == "GBRG" || token == "GBRG8" || token == "BAYERGB8" || token == "BAYERGB" ||
      token == "BAYERGBRG" || token == "BAYERGBRG8") {
    return PixelFormatKind::BAYER_GB;
  }
  return PixelFormatKind::UNKNOWN;
}

inline bool isBayerFormat(PixelFormatKind kind) {
  return kind == PixelFormatKind::BAYER_RG || kind == PixelFormatKind::BAYER_BG ||
         kind == PixelFormatKind::BAYER_GR || kind == PixelFormatKind::BAYER_GB;
}

inline int bayerToBgrCode(PixelFormatKind kind) {
  switch (kind) {
    case PixelFormatKind::BAYER_RG:
      return cv::COLOR_BayerRG2BGR;
    case PixelFormatKind::BAYER_BG:
      return cv::COLOR_BayerBG2BGR;
    case PixelFormatKind::BAYER_GR:
      return cv::COLOR_BayerGR2BGR;
    case PixelFormatKind::BAYER_GB:
      return cv::COLOR_BayerGB2BGR;
    default:
      return -1;
  }
}

inline std::string canonicalCapsFormat(const std::string& format) {
  switch (classifyPixelFormat(format)) {
    case PixelFormatKind::GRAY:
      return "GRAY8";
    case PixelFormatKind::BGR:
      return "BGR";
    case PixelFormatKind::RGB:
      return "RGB";
    case PixelFormatKind::BGRA:
      return "BGRA";
    case PixelFormatKind::RGBA:
      return "RGBA";
    case PixelFormatKind::BAYER_RG:
      return "rggb";
    case PixelFormatKind::BAYER_BG:
      return "bggr";
    case PixelFormatKind::BAYER_GR:
      return "grbg";
    case PixelFormatKind::BAYER_GB:
      return "gbrg";
    case PixelFormatKind::UNKNOWN:
    default:
      return toUpperCopy(trimCopy(format));
  }
}

inline bool shouldUseVideoConvert(const std::string& format) {
  const PixelFormatKind kind = classifyPixelFormat(format);
  if (kind == PixelFormatKind::BGR || kind == PixelFormatKind::GRAY || isBayerFormat(kind)) {
    return false;
  }
  return true;
}

inline std::string mediaTypeForFormat(const std::string& format) {
  return isBayerFormat(classifyPixelFormat(format)) ? "video/x-bayer" : "video/x-raw";
}

}  // namespace rkapp::capture::detail
