#include "rkapp/pipeline/BoxTracker.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace rkapp::pipeline {
namespace {

float clampUnit(float value) {
  return std::clamp(value, 0.0f, 1.0f);
}

}  // namespace

BoxTracker::BoxTracker(const Config& config) {
  configure(config);
}

void BoxTracker::configure(const Config& config) {
  config_ = config;
  config_.match_iou = clampUnit(config_.match_iou);
  config_.ema_alpha = clampUnit(config_.ema_alpha);
  config_.confirm_hits = std::max(1, config_.confirm_hits);
  config_.max_misses = std::max(0, config_.max_misses);
  config_.missing_conf_decay = std::max(0.0f, config_.missing_conf_decay);
  reset();
}

void BoxTracker::reset() {
  tracks_.clear();
  next_track_id_ = 1;
}

float BoxTracker::computeIoU(const infer::Detection& lhs, const infer::Detection& rhs) {
  const float lhs_x2 = lhs.x + lhs.w;
  const float lhs_y2 = lhs.y + lhs.h;
  const float rhs_x2 = rhs.x + rhs.w;
  const float rhs_y2 = rhs.y + rhs.h;

  const float inter_x1 = std::max(lhs.x, rhs.x);
  const float inter_y1 = std::max(lhs.y, rhs.y);
  const float inter_x2 = std::min(lhs_x2, rhs_x2);
  const float inter_y2 = std::min(lhs_y2, rhs_y2);
  const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
  const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
  const float inter_area = inter_w * inter_h;
  const float lhs_area = std::max(0.0f, lhs.w) * std::max(0.0f, lhs.h);
  const float rhs_area = std::max(0.0f, rhs.w) * std::max(0.0f, rhs.h);
  const float union_area = lhs_area + rhs_area - inter_area;
  if (union_area <= 0.0f) {
    return 0.0f;
  }
  return inter_area / union_area;
}

infer::Detection BoxTracker::smoothDetection(const infer::Detection& previous,
                                             const infer::Detection& current,
                                             float alpha) {
  infer::Detection blended = current;
  blended.x = previous.x + alpha * (current.x - previous.x);
  blended.y = previous.y + alpha * (current.y - previous.y);
  blended.w = previous.w + alpha * (current.w - previous.w);
  blended.h = previous.h + alpha * (current.h - previous.h);
  blended.confidence = previous.confidence + alpha * (current.confidence - previous.confidence);
  if (!current.keypoints.empty() && current.keypoints.size() == previous.keypoints.size()) {
    for (size_t i = 0; i < current.keypoints.size(); ++i) {
      blended.keypoints[i].x =
          previous.keypoints[i].x + alpha * (current.keypoints[i].x - previous.keypoints[i].x);
      blended.keypoints[i].y =
          previous.keypoints[i].y + alpha * (current.keypoints[i].y - previous.keypoints[i].y);
      blended.keypoints[i].visibility = previous.keypoints[i].visibility +
          alpha * (current.keypoints[i].visibility - previous.keypoints[i].visibility);
    }
  }
  return blended;
}

std::vector<infer::Detection> BoxTracker::update(const std::vector<infer::Detection>& detections) {
  if (!config_.enable) {
    return detections;
  }

  std::vector<MatchCandidate> candidates;
  candidates.reserve(tracks_.size() * detections.size());
  for (int track_index = 0; track_index < static_cast<int>(tracks_.size()); ++track_index) {
    for (int detection_index = 0; detection_index < static_cast<int>(detections.size());
         ++detection_index) {
      if (tracks_[track_index].detection.class_id != detections[detection_index].class_id) {
        continue;
      }
      const float iou = computeIoU(tracks_[track_index].detection, detections[detection_index]);
      if (iou >= config_.match_iou) {
        candidates.push_back({track_index, detection_index, iou});
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [&](const MatchCandidate& lhs, const MatchCandidate& rhs) {
              if (lhs.iou != rhs.iou) {
                return lhs.iou > rhs.iou;
              }
              return detections[lhs.detection_index].confidence >
                  detections[rhs.detection_index].confidence;
            });

  std::vector<int> assigned_track(detections.size(), -1);
  std::vector<int> assigned_detection(tracks_.size(), -1);
  for (const auto& candidate : candidates) {
    if (assigned_track[candidate.detection_index] != -1 ||
        assigned_detection[candidate.track_index] != -1) {
      continue;
    }
    assigned_track[candidate.detection_index] = candidate.track_index;
    assigned_detection[candidate.track_index] = candidate.detection_index;
  }

  for (int track_index = 0; track_index < static_cast<int>(tracks_.size()); ++track_index) {
    auto& track = tracks_[track_index];
    const int detection_index = assigned_detection[track_index];
    if (detection_index >= 0) {
      track.detection = smoothDetection(track.detection, detections[detection_index],
                                        config_.ema_alpha);
      track.hits += 1;
      track.misses = 0;
      track.confirmed = track.hits >= config_.confirm_hits;
    } else {
      track.misses += 1;
    }
  }

  for (int detection_index = 0; detection_index < static_cast<int>(detections.size());
       ++detection_index) {
    if (assigned_track[detection_index] != -1) {
      continue;
    }
    Track track;
    track.id = next_track_id_++;
    track.detection = detections[detection_index];
    track.hits = 1;
    track.misses = 0;
    track.confirmed = track.hits >= config_.confirm_hits;
    tracks_.push_back(std::move(track));
  }

  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track& track) {
                                 return track.misses > config_.max_misses;
                               }),
                tracks_.end());

  std::vector<infer::Detection> stabilized;
  stabilized.reserve(tracks_.size());
  for (const auto& track : tracks_) {
    if (!track.confirmed) {
      continue;
    }
    if (track.misses > 0 && !config_.keep_missing_tracks) {
      continue;
    }
    infer::Detection detection = track.detection;
    if (track.misses > 0) {
      detection.confidence =
          std::max(0.0f, detection.confidence - config_.missing_conf_decay * track.misses);
      if (detection.confidence <= 0.0f) {
        continue;
      }
    }
    stabilized.push_back(std::move(detection));
  }

  std::sort(stabilized.begin(), stabilized.end(),
            [](const infer::Detection& lhs, const infer::Detection& rhs) {
              return lhs.confidence > rhs.confidence;
            });
  return stabilized;
}

}  // namespace rkapp::pipeline
