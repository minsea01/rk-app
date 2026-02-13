#!/usr/bin/env python3
"""Unit tests for pose estimation extensions.

Tests cover:
- decode_meta.py: task/num_keypoints parsing, resolve_dfl_layout with keypoints
- yolo_post.py: postprocess_yolov8_pose() end-to-end decode
"""

import json

import numpy as np
import pytest

from apps.utils.decode_meta import (
    load_decode_meta,
    normalize_decode_meta,
    resolve_dfl_layout,
    resolve_head,
)
from apps.utils.yolo_post import (
    dfl_decode,
    letterbox,
    make_anchors,
    postprocess_yolov8_pose,
    sigmoid,
)


# ============================================================================
# decode_meta: pose metadata extensions
# ============================================================================


class TestDecodeMetaPose:
    """Tests for pose-related metadata normalization and resolution."""

    def test_normalize_pose_meta(self):
        meta = normalize_decode_meta(
            {
                "head": "dfl",
                "reg_max": 16,
                "strides": [8, 16, 32],
                "num_classes": 1,
                "task": "pose",
                "num_keypoints": 17,
            }
        )
        assert meta["task"] == "pose"
        assert meta["num_keypoints"] == 17
        assert meta["num_classes"] == 1
        assert meta["head"] == "dfl"

    def test_normalize_detect_defaults(self):
        """Detection-only metadata should have None for task/num_keypoints."""
        meta = normalize_decode_meta(
            {"head": "dfl", "reg_max": 16, "num_classes": 80}
        )
        assert meta["task"] is None
        assert meta["num_keypoints"] is None

    def test_normalize_invalid_task(self):
        meta = normalize_decode_meta({"task": "invalid_task"})
        assert meta["task"] is None

    def test_normalize_kpt_shape_alias(self):
        """kpt_shape should be recognized as alias for num_keypoints."""
        meta = normalize_decode_meta({"kpt_shape": "17"})
        assert meta["num_keypoints"] == 17

    def test_resolve_dfl_layout_pose(self):
        """Pose model: 116 channels = 64 DFL + 1 cls + 51 kpts."""
        pose_meta = {
            "head": "dfl",
            "reg_max": 16,
            "strides": [8, 16, 32],
            "num_classes": 1,
            "num_keypoints": 17,
        }
        result = resolve_dfl_layout(116, pose_meta)
        assert result is not None
        reg_max, strides = result
        assert reg_max == 16
        assert strides == (8, 16, 32)

    def test_resolve_dfl_layout_pose_wrong_channels(self):
        """Mismatch channels should fail."""
        pose_meta = {
            "reg_max": 16,
            "num_classes": 1,
            "num_keypoints": 17,
        }
        # 144 channels is for 80-class detect, not pose
        assert resolve_dfl_layout(144, pose_meta) is None

    def test_resolve_dfl_layout_detect_unchanged(self):
        """Standard detection layout should still work."""
        detect_meta = {
            "reg_max": 16,
            "strides": [8, 16, 32],
            "num_classes": 80,
        }
        result = resolve_dfl_layout(144, detect_meta)
        assert result == (16, (8, 16, 32))

    def test_resolve_dfl_layout_10class(self):
        """10-class detect: 74 channels = 64 DFL + 10 cls."""
        meta = {"reg_max": 16, "num_classes": 10}
        result = resolve_dfl_layout(74, meta)
        assert result is not None
        assert result[0] == 16

    def test_resolve_head_pose_model(self):
        """resolve_head should work for pose channel count."""
        pose_meta = {
            "reg_max": 16,
            "num_classes": 1,
            "num_keypoints": 17,
        }
        # 116 = 64 + 1 + 51
        result = resolve_head("auto", channels=116, decode_meta=pose_meta)
        assert result == "dfl"

    def test_load_decode_meta_pose_sidecar(self, tmp_path):
        model_path = tmp_path / "pose.rknn"
        model_path.write_bytes(b"fake")
        sidecar = tmp_path / "pose.rknn.json"
        sidecar.write_text(
            json.dumps(
                {
                    "head": "dfl",
                    "reg_max": 16,
                    "strides": [8, 16, 32],
                    "num_classes": 1,
                    "task": "pose",
                    "num_keypoints": 17,
                }
            )
        )
        meta = load_decode_meta(model_path)
        assert meta["task"] == "pose"
        assert meta["num_keypoints"] == 17
        assert meta["num_classes"] == 1


# ============================================================================
# yolo_post: pose postprocessing
# ============================================================================


def _make_pose_predictions(
    img_size: int = 416,
    reg_max: int = 16,
    num_keypoints: int = 17,
    strides: tuple = (8, 16, 32),
    target_anchor_idx: int = 0,
    person_logit: float = 3.0,
) -> np.ndarray:
    """Create synthetic pose model output with one strong detection.

    Returns shape (1, N, C) where C = 4*reg_max + 1 + num_keypoints*3.
    """
    nc = 1
    c = 4 * reg_max + nc + num_keypoints * 3

    # Total anchors
    n = sum((img_size // s) ** 2 for s in strides)
    pred = np.zeros((1, n, c), dtype=np.float32)

    # DFL: uniform distribution (each bin gets equal weight -> expected value ~7.5)
    for side in range(4):
        for k in range(reg_max):
            pred[0, target_anchor_idx, side * reg_max + k] = 1.0

    # Person confidence (sigmoid(3.0) ≈ 0.95)
    pred[0, target_anchor_idx, 4 * reg_max] = person_logit

    # Keypoints: small offsets from anchor center
    kpt_base = 4 * reg_max + nc
    for k in range(num_keypoints):
        pred[0, target_anchor_idx, kpt_base + k * 3 + 0] = 0.1  # raw_x offset
        pred[0, target_anchor_idx, kpt_base + k * 3 + 1] = 0.2  # raw_y offset
        pred[0, target_anchor_idx, kpt_base + k * 3 + 2] = 2.0  # visibility logit

    return pred


class TestPostprocessYolov8Pose:
    """Tests for pose model post-processing."""

    def test_basic_pose_output_shape(self):
        """Test that pose postprocessing returns correct output shapes."""
        img_size = 416
        preds = _make_pose_predictions(img_size=img_size)

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, ratio, pad = letterbox(img, img_size)

        boxes, confs, class_ids, keypoints = postprocess_yolov8_pose(
            preds,
            img_size=img_size,
            orig_shape=(480, 640),
            ratio_pad=(ratio, pad),
            conf_thres=0.5,
        )

        assert boxes.ndim == 2 and boxes.shape[1] == 4
        assert confs.ndim == 1
        assert class_ids.ndim == 1
        assert keypoints.ndim == 3 and keypoints.shape[1] == 17 and keypoints.shape[2] == 3
        assert len(boxes) == len(confs) == len(class_ids) == len(keypoints)
        # Should have at least one detection (logit=3.0 -> ~0.95 conf)
        assert len(boxes) >= 1

    def test_all_class_ids_are_zero(self):
        """Pose model only has person class (id=0)."""
        preds = _make_pose_predictions()
        img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        _, ratio, pad = letterbox(img, 416)

        _, _, class_ids, _ = postprocess_yolov8_pose(
            preds,
            img_size=416,
            orig_shape=(416, 416),
            ratio_pad=(ratio, pad),
            conf_thres=0.5,
        )
        assert np.all(class_ids == 0)

    def test_keypoint_visibility_range(self):
        """Keypoint visibility should be in [0, 1] (sigmoid output)."""
        preds = _make_pose_predictions()
        img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        _, ratio, pad = letterbox(img, 416)

        _, _, _, keypoints = postprocess_yolov8_pose(
            preds,
            img_size=416,
            orig_shape=(416, 416),
            ratio_pad=(ratio, pad),
            conf_thres=0.5,
        )
        if len(keypoints) > 0:
            vis = keypoints[:, :, 2]
            assert np.all(vis >= 0.0)
            assert np.all(vis <= 1.0)

    def test_keypoints_within_image_bounds(self):
        """Keypoint coordinates should be clipped to original image."""
        orig_h, orig_w = 480, 640
        preds = _make_pose_predictions()
        img = np.random.randint(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)
        _, ratio, pad = letterbox(img, 416)

        _, _, _, keypoints = postprocess_yolov8_pose(
            preds,
            img_size=416,
            orig_shape=(orig_h, orig_w),
            ratio_pad=(ratio, pad),
            conf_thres=0.5,
        )
        if len(keypoints) > 0:
            assert np.all(keypoints[:, :, 0] >= 0)
            assert np.all(keypoints[:, :, 0] <= orig_w)
            assert np.all(keypoints[:, :, 1] >= 0)
            assert np.all(keypoints[:, :, 1] <= orig_h)

    def test_high_conf_threshold_filters_all(self):
        """Very high confidence threshold should filter out all detections."""
        preds = _make_pose_predictions(person_logit=0.0)  # sigmoid(0) = 0.5
        img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        _, ratio, pad = letterbox(img, 416)

        boxes, confs, class_ids, keypoints = postprocess_yolov8_pose(
            preds,
            img_size=416,
            orig_shape=(416, 416),
            ratio_pad=(ratio, pad),
            conf_thres=0.9,
        )
        assert len(boxes) == 0
        assert len(keypoints) == 0
        assert keypoints.shape == (0, 17, 3)

    def test_channel_mismatch_raises(self):
        """Wrong number of channels should raise ValueError."""
        # Create wrong-shape predictions (detection model output)
        n = sum((416 // s) ** 2 for s in (8, 16, 32))
        wrong_preds = np.zeros((1, n, 144), dtype=np.float32)  # 80-class detect

        with pytest.raises(ValueError, match="Channel mismatch"):
            postprocess_yolov8_pose(
                wrong_preds,
                img_size=416,
                orig_shape=(416, 416),
                ratio_pad=(1.0, (0.0, 0.0)),
            )

    def test_batch_dim_validation(self):
        """Non-batch-1 input should raise ValueError."""
        n = sum((416 // s) ** 2 for s in (8, 16, 32))
        bad_preds = np.zeros((2, n, 116), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected predictions shape"):
            postprocess_yolov8_pose(
                bad_preds,
                img_size=416,
                orig_shape=(416, 416),
                ratio_pad=(1.0, (0.0, 0.0)),
            )

    def test_keypoint_decode_formula(self):
        """Verify keypoint decode: kpt = (anchor + raw * 2) * stride."""
        img_size = 416
        reg_max = 16
        nc = 1
        nk = 17
        strides = (8, 16, 32)
        c = 4 * reg_max + nc + nk * 3
        n = sum((img_size // s) ** 2 for s in strides)

        pred = np.zeros((1, n, c), dtype=np.float32)

        # Set anchor index 0 (stride=8, center at (4, 4))
        # Strong person confidence
        pred[0, 0, 4 * reg_max] = 5.0

        # First keypoint: raw_x=0.5, raw_y=0.25, vis_logit=3.0
        kpt_base = 4 * reg_max + nc
        pred[0, 0, kpt_base + 0] = 0.5   # raw_x
        pred[0, 0, kpt_base + 1] = 0.25  # raw_y
        pred[0, 0, kpt_base + 2] = 3.0   # visibility logit

        # Identity letterbox (no padding, no scaling)
        boxes, confs, class_ids, keypoints = postprocess_yolov8_pose(
            pred,
            img_size=img_size,
            orig_shape=(img_size, img_size),
            ratio_pad=(1.0, (0.0, 0.0)),
            conf_thres=0.5,
        )

        assert len(keypoints) >= 1
        kpt = keypoints[0, 0]  # first detection, first keypoint

        # anchor center for idx 0, stride 8: cx=4, cy=4
        # kpt_x = (4 + 0.5 * 2) * 8 = 5 * 8 = 40
        # kpt_y = (4 + 0.25 * 2) * 8 = 4.5 * 8 = 36
        assert abs(kpt[0] - 40.0) < 0.5, f"kpt_x={kpt[0]}, expected ~40"
        assert abs(kpt[1] - 36.0) < 0.5, f"kpt_y={kpt[1]}, expected ~36"
        # visibility: sigmoid(3.0) ≈ 0.953
        assert 0.94 < kpt[2] < 0.97, f"kpt_vis={kpt[2]}, expected ~0.953"
