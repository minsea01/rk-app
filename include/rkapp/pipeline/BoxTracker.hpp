#pragma once

#include <vector>

#include "rkapp/infer/IInferEngine.hpp"

namespace rkapp::pipeline {

class BoxTracker {
public:
  struct Config {
    bool enable = false;
    float match_iou = 0.30f;
    float ema_alpha = 0.65f;
    int confirm_hits = 2;
    int max_misses = 4;
    bool keep_missing_tracks = true;
    float missing_conf_decay = 0.08f;
  };

  BoxTracker() = default;
  explicit BoxTracker(const Config& config);

  void configure(const Config& config);
  void reset();

  std::vector<infer::Detection> update(const std::vector<infer::Detection>& detections);

private:
  struct Track {
    int id = 0;
    infer::Detection detection;
    int hits = 0;
    int misses = 0;
    bool confirmed = false;
  };

  struct MatchCandidate {
    int track_index = -1;
    int detection_index = -1;
    float iou = 0.0f;
  };

  static float computeIoU(const infer::Detection& lhs, const infer::Detection& rhs);
  static infer::Detection smoothDetection(const infer::Detection& previous,
                                          const infer::Detection& current,
                                          float alpha);

  Config config_;
  int next_track_id_ = 1;
  std::vector<Track> tracks_;
};

}  // namespace rkapp::pipeline
