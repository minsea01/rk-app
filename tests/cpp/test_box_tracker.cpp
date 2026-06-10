#include <gtest/gtest.h>

#include "rkapp/pipeline/BoxTracker.hpp"

namespace {

rkapp::infer::Detection makeDetection(float x, float y, float w, float h, float confidence) {
  rkapp::infer::Detection detection;
  detection.x = x;
  detection.y = y;
  detection.w = w;
  detection.h = h;
  detection.confidence = confidence;
  detection.class_id = 0;
  detection.class_name = "person";
  return detection;
}

}  // namespace

TEST(BoxTrackerTests, DisabledTrackerPassesThroughDetections) {
  rkapp::pipeline::BoxTracker tracker;
  rkapp::pipeline::BoxTracker::Config config;
  config.enable = false;
  tracker.configure(config);

  const auto detections = std::vector<rkapp::infer::Detection>{
      makeDetection(10.0f, 20.0f, 30.0f, 40.0f, 0.8f)};

  const auto stabilized = tracker.update(detections);

  ASSERT_EQ(stabilized.size(), 1u);
  EXPECT_FLOAT_EQ(stabilized.front().x, 10.0f);
  EXPECT_FLOAT_EQ(stabilized.front().confidence, 0.8f);
}

TEST(BoxTrackerTests, ConfirmsAndSmoothsMatchedTrack) {
  rkapp::pipeline::BoxTracker tracker;
  rkapp::pipeline::BoxTracker::Config config;
  config.enable = true;
  config.confirm_hits = 2;
  config.max_misses = 2;
  config.match_iou = 0.3f;
  config.ema_alpha = 0.5f;
  tracker.configure(config);

  EXPECT_TRUE(tracker.update({makeDetection(10.0f, 10.0f, 20.0f, 30.0f, 0.80f)}).empty());

  const auto stabilized = tracker.update({makeDetection(12.0f, 12.0f, 20.0f, 30.0f, 0.90f)});

  ASSERT_EQ(stabilized.size(), 1u);
  EXPECT_NEAR(stabilized.front().x, 11.0f, 1e-3f);
  EXPECT_NEAR(stabilized.front().y, 11.0f, 1e-3f);
  EXPECT_NEAR(stabilized.front().confidence, 0.85f, 1e-3f);
}

TEST(BoxTrackerTests, KeepsConfirmedTrackAcrossShortMisses) {
  rkapp::pipeline::BoxTracker tracker;
  rkapp::pipeline::BoxTracker::Config config;
  config.enable = true;
  config.confirm_hits = 2;
  config.max_misses = 1;
  config.keep_missing_tracks = true;
  config.missing_conf_decay = 0.10f;
  tracker.configure(config);

  EXPECT_TRUE(tracker.update({makeDetection(40.0f, 20.0f, 16.0f, 32.0f, 0.90f)}).empty());
  ASSERT_EQ(tracker.update({makeDetection(41.0f, 21.0f, 16.0f, 32.0f, 0.88f)}).size(), 1u);

  const auto held = tracker.update({});
  ASSERT_EQ(held.size(), 1u);
  EXPECT_GT(held.front().confidence, 0.0f);

  EXPECT_TRUE(tracker.update({}).empty());
}
