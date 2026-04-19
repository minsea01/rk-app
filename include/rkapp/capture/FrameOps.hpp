#pragma once

#include <opencv2/opencv.hpp>

#include "rkapp/capture/ISource.hpp"
#include "rkapp/common/log.hpp"
#include "rkapp/preprocess/Preprocess.hpp"

namespace rkapp::capture {

struct BgrFrameView {
  cv::Mat image;
  bool aliases_source = false;

  bool empty() const { return image.empty(); }
};

inline int bayerToBgrCode(PixelFormat format) {
  switch (format) {
    case PixelFormat::BAYER_RG8:
      return cv::COLOR_BayerRG2BGR;
    case PixelFormat::BAYER_BG8:
      return cv::COLOR_BayerBG2BGR;
    case PixelFormat::BAYER_GR8:
      return cv::COLOR_BayerGR2BGR;
    case PixelFormat::BAYER_GB8:
      return cv::COLOR_BayerGB2BGR;
    default:
      return -1;
  }
}

inline bool hasRawSharedBacking(const CaptureFrame& frame) {
  return frame.owner != nullptr || frame.storage_kind != StorageKind::CPU_MAT || frame.dma_fd >= 0;
}

inline BgrFrameView convertToBgr(
    const CaptureFrame& frame,
    preprocess::AccelBackend backend = preprocess::AccelBackend::AUTO) {
  if (frame.mat.empty()) {
    return {};
  }

  switch (frame.pixel_format) {
    case PixelFormat::BGR888:
      return {frame.mat, true};
    case PixelFormat::RGB888:
      return {preprocess::Preprocess::convertColor(frame.mat, cv::COLOR_RGB2BGR, backend), false};
    case PixelFormat::BGRA8888:
      return {preprocess::Preprocess::convertColor(frame.mat, cv::COLOR_BGRA2BGR, backend),
              false};
    case PixelFormat::RGBA8888:
      return {preprocess::Preprocess::convertColor(frame.mat, cv::COLOR_RGBA2BGR, backend),
              false};
    case PixelFormat::GRAY8:
      return {preprocess::Preprocess::convertColor(frame.mat, cv::COLOR_GRAY2BGR, backend), false};
    case PixelFormat::NV12:
    case PixelFormat::NV21: {
      if (frame.image_size.width <= 0 || frame.image_size.height <= 0) {
        LOGW("capture::convertToBgr: NV12/NV21 frame missing image_size metadata");
        return {};
      }
      return {preprocess::Preprocess::convertYuv420spToBgr(
                  frame.mat, frame.image_size, frame.pixel_format == PixelFormat::NV21, backend),
              false};
    }
    case PixelFormat::BAYER_RG8:
    case PixelFormat::BAYER_BG8:
    case PixelFormat::BAYER_GR8:
    case PixelFormat::BAYER_GB8: {
      const int code = bayerToBgrCode(frame.pixel_format);
      if (code < 0) {
        return {};
      }
      cv::Mat converted;
      cv::cvtColor(frame.mat, converted, code);
      return {std::move(converted), false};
    }
    case PixelFormat::UNKNOWN:
    default: {
      cv::Mat converted = preprocess::Preprocess::ensureBgr8(frame.mat, backend);
      const bool aliases_source = !converted.empty() &&
          converted.data == frame.mat.data &&
          converted.rows == frame.mat.rows &&
          converted.cols == frame.mat.cols &&
          converted.type() == frame.mat.type();
      return {std::move(converted), aliases_source};
    }
  }
}

inline cv::Mat detachFrameIfAliased(const cv::Mat& frame,
                                    const CaptureFrame& source_frame,
                                    bool aliases_source) {
  if (frame.empty()) {
    return {};
  }
  if (aliases_source && hasRawSharedBacking(source_frame)) {
    return frame.clone();
  }
  return frame;
}

}  // namespace rkapp::capture
