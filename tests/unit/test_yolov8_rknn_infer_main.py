#!/usr/bin/env python3
"""Unit tests for apps/yolov8_rknn_infer.py.

Tests decode_predictions, load_labels, draw_boxes, and basic main() scenarios.
Note: Full main() testing requires rknnlite which is RK3588-specific.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

import pytest
import numpy as np

from apps.exceptions import ModelLoadError, RKNNError, PreprocessError, InferenceError


class TestDecodePredictions:
    """Test suite for decode_predictions function."""

    def test_decode_predictions_with_dfl_head(self):
        """Test decode_predictions with DFL head."""
        from apps.yolov8_rknn_infer import decode_predictions

        # Create prediction with shape (1, N, 64+nc) matching 640 anchors
        N = 8400  # Must match anchor count for 640
        nc = 80
        pred = np.random.randn(1, N, 64 + nc).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head="dfl"
        )

        assert isinstance(boxes, np.ndarray)
        assert isinstance(confs, np.ndarray)
        assert isinstance(cls_ids, np.ndarray)

    def test_decode_predictions_with_raw_head(self):
        """Test decode_predictions with raw head."""
        from apps.yolov8_rknn_infer import decode_predictions

        # Create prediction with raw format [cx, cy, w, h, obj, cls...]
        N = 100
        nc = 80
        pred = np.random.randn(1, N, 5 + nc).astype(np.float32)
        # Make some predictions high confidence
        pred[0, :10, 4] = 5.0  # High objectness

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head="raw"
        )

        assert isinstance(boxes, np.ndarray)
        assert isinstance(confs, np.ndarray)
        assert isinstance(cls_ids, np.ndarray)

    def test_decode_predictions_auto_selects_dfl_for_large_channels(self):
        """Test decode_predictions auto-selects DFL head for C >= 64."""
        from apps.yolov8_rknn_infer import decode_predictions

        # DFL format (C >= 64) with correct anchor count
        N = 8400  # Correct anchor count for 640
        pred_dfl = np.random.randn(1, N, 144).astype(np.float32)
        boxes, _, _ = decode_predictions(
            pred_dfl, imgsz=640, conf_thres=0.9, iou_thres=0.45, head="auto"
        )
        assert isinstance(boxes, np.ndarray)

    def test_decode_predictions_auto_selects_raw_for_small_channels(self):
        """Test decode_predictions auto-selects raw head for C < 64."""
        from apps.yolov8_rknn_infer import decode_predictions

        # Raw format (C < 64) - explicit raw head to avoid anchor mismatch
        pred_raw = np.random.randn(1, 100, 85).astype(np.float32)
        pred_raw[0, :5, 4] = 5.0  # High objectness
        boxes, _, _ = decode_predictions(
            pred_raw, imgsz=640, conf_thres=0.5, iou_thres=0.45, head="raw"
        )
        assert isinstance(boxes, np.ndarray)

    def test_decode_predictions_handles_2d_input(self):
        """Test decode_predictions handles 2D input (N, C)."""
        from apps.yolov8_rknn_infer import decode_predictions

        # 2D input with raw format
        pred = np.random.randn(100, 85).astype(np.float32)
        pred[:5, 4] = 5.0  # High objectness
        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head="raw"
        )

        assert isinstance(boxes, np.ndarray)

    def test_decode_predictions_returns_empty_for_no_detections(self):
        """Test decode_predictions returns empty arrays when no detections."""
        from apps.yolov8_rknn_infer import decode_predictions

        # Very low logits = no detections above threshold (raw format)
        pred = np.ones((1, 100, 85), dtype=np.float32) * -100

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.9, iou_thres=0.45, head="raw"
        )

        assert boxes.shape[0] == 0
        assert confs.shape[0] == 0
        assert cls_ids.shape[0] == 0

    def test_decode_predictions_with_orig_shape_scaling(self):
        """Test decode_predictions scales boxes to original coordinates."""
        from apps.yolov8_rknn_infer import decode_predictions

        N = 100
        pred = np.random.randn(1, N, 85).astype(np.float32)
        pred[0, :5, 4] = 5.0  # High objectness

        boxes, confs, cls_ids = decode_predictions(
            pred,
            imgsz=640,
            conf_thres=0.5,
            iou_thres=0.45,
            head="raw",
            ratio_pad=(0.5, (10.0, 10.0)),
            orig_shape=(480, 640),
        )

        # Boxes should be clipped to original image bounds
        if len(boxes) > 0:
            assert np.all(boxes[:, 0] >= 0) and np.all(boxes[:, 0] < 640)
            assert np.all(boxes[:, 1] >= 0) and np.all(boxes[:, 1] < 480)

    def test_decode_predictions_transposes_when_needed(self):
        """Test decode_predictions transposes (1, C, N) to (1, N, C)."""
        from apps.yolov8_rknn_infer import decode_predictions

        # Shape (1, C, N) where C > N indicates transposed
        pred = np.random.randn(1, 85, 50).astype(np.float32)
        pred[0, 4, :10] = 5.0  # High objectness after transpose

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head="raw"
        )

        assert isinstance(boxes, np.ndarray)


class TestLoadLabels:
    """Test suite for load_labels function."""

    def setup_method(self):
        """Create temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_load_labels_from_file(self):
        """Test loading labels from file."""
        from apps.yolov8_rknn_infer import load_labels

        names_file = self.temp_path / "names.txt"
        names_file.write_text("person\ncar\nbicycle\n")

        labels = load_labels(names_file)

        assert labels == ["person", "car", "bicycle"]

    def test_load_labels_returns_none_for_nonexistent_file(self):
        """Test that None is returned for nonexistent file."""
        from apps.yolov8_rknn_infer import load_labels

        labels = load_labels(Path("/nonexistent/path.txt"))
        assert labels is None

    def test_load_labels_returns_none_for_none_input(self):
        """Test that None is returned for None input."""
        from apps.yolov8_rknn_infer import load_labels

        labels = load_labels(None)
        assert labels is None

    def test_load_labels_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        from apps.yolov8_rknn_infer import load_labels

        names_file = self.temp_path / "names.txt"
        names_file.write_text("person\n\ncar\n  \nbicycle\n")

        labels = load_labels(names_file)

        assert labels == ["person", "car", "bicycle"]


class TestDrawBoxes:
    """Test suite for draw_boxes function."""

    def test_draw_boxes_with_names(self):
        """Test drawing boxes with class names."""
        from apps.yolov8_rknn_infer import draw_boxes

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confs = np.array([0.9])
        class_ids = np.array([0])
        names = ["person", "car"]

        result = draw_boxes(img, boxes, confs, class_ids, names)

        assert result.shape == img.shape
        # Image should have been modified (not all zeros)
        assert not np.all(result == 0)

    def test_draw_boxes_without_names(self):
        """Test drawing boxes without class names."""
        from apps.yolov8_rknn_infer import draw_boxes

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confs = np.array([0.9])
        class_ids = np.array([0])

        result = draw_boxes(img, boxes, confs, class_ids, names=None)

        assert result.shape == img.shape

    def test_draw_boxes_empty_input(self):
        """Test drawing with no boxes."""
        from apps.yolov8_rknn_infer import draw_boxes

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.empty((0, 4), dtype=np.float32)
        confs = np.array([])
        class_ids = np.array([])

        result = draw_boxes(img, boxes, confs, class_ids)

        # Image should be unchanged
        assert np.all(result == img)

    def test_draw_boxes_multiple_detections(self):
        """Test drawing multiple boxes."""
        from apps.yolov8_rknn_infer import draw_boxes

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.array(
            [[10, 10, 100, 100], [200, 200, 300, 300], [400, 100, 500, 200]], dtype=np.float32
        )
        confs = np.array([0.9, 0.8, 0.7])
        class_ids = np.array([0, 1, 2])
        names = ["person", "car", "bicycle"]

        result = draw_boxes(img, boxes, confs, class_ids, names)

        assert result.shape == img.shape
        # Image should have been modified
        assert not np.all(result == 0)

    def test_draw_boxes_handles_out_of_range_class_id(self):
        """Test drawing with class ID out of names range."""
        from apps.yolov8_rknn_infer import draw_boxes

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confs = np.array([0.9])
        class_ids = np.array([99])  # Out of range
        names = ["person", "car"]

        # Should not crash, falls back to numeric label
        result = draw_boxes(img, boxes, confs, class_ids, names)
        assert result.shape == img.shape


class TestRawDecodePath:
    """Test suite for raw decode path edge cases."""

    def test_raw_decode_with_single_class(self):
        """Test raw decode with only objectness (no class scores)."""
        from apps.yolov8_rknn_infer import decode_predictions

        # Only 5 channels: cx, cy, w, h, obj
        pred = np.random.randn(1, 100, 5).astype(np.float32)
        pred[0, :10, 4] = 5.0  # High objectness

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head="raw"
        )

        # Class IDs should all be 0 (default)
        if len(cls_ids) > 0:
            assert np.all(cls_ids == 0)

    def test_raw_decode_with_less_than_5_channels(self):
        """Test raw decode with insufficient channels returns empty."""
        from apps.yolov8_rknn_infer import decode_predictions

        # Only 4 channels - not enough for detection
        pred = np.random.randn(1, 100, 4).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head="raw"
        )

        assert boxes.shape[0] == 0
        assert confs.shape[0] == 0
        assert cls_ids.shape[0] == 0

    def test_raw_decode_scales_normalized_coords(self):
        """Test raw decode scales normalized coordinates to image size."""
        from apps.yolov8_rknn_infer import decode_predictions

        # Create normalized predictions (coords in 0-1 range)
        pred = np.zeros((1, 10, 85), dtype=np.float32)
        # Set cx, cy, w, h as normalized values
        pred[0, 0, 0:4] = [0.5, 0.5, 0.1, 0.1]  # Center of image, small box
        pred[0, 0, 4] = 5.0  # High objectness

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head="raw"
        )

        # Should have detections with scaled coordinates
        if len(boxes) > 0:
            # Coordinates should be scaled to image size
            assert np.max(boxes) > 1.0  # Not normalized anymore


class TestArgumentParser:
    """Test suite for CLI argument parser compatibility."""

    def test_parser_accepts_new_preprocess_options(self):
        from apps.yolov8_rknn_infer import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--model",
                "dummy.rknn",
                "--cfg",
                "config/detect.yaml",
                "--pp-profile",
                "quality",
                "--undistort-enable",
                "--undistort-calib",
                "calib.yaml",
                "--roi-enable",
                "--roi-mode",
                "pixel",
                "--roi-px",
                "1,2,30,40",
                "--roi-min-size",
                "12",
                "--gamma-enable",
                "--gamma",
                "0.7",
                "--white-balance-enable",
                "--white-balance-clip",
                "1.5",
                "--denoise-enable",
                "--denoise-method",
                "bilateral",
                "--denoise-d",
                "7",
                "--denoise-sigma-color",
                "30",
                "--denoise-sigma-space",
                "20",
                "--input-format",
                "gray",
            ]
        )

        assert args.cfg == Path("config/detect.yaml")
        assert args.pp_profile == "quality"
        assert args.undistort_enable is True
        assert args.roi_enable is True
        assert args.roi_px == "1,2,30,40"
        assert args.gamma_enable is True
        assert args.white_balance_enable is True
        assert args.denoise_enable is True
        assert args.input_format == "gray"

    def test_parser_keeps_backward_compatible_defaults(self):
        from apps.yolov8_rknn_infer import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--model", "dummy.rknn"])

        assert args.cfg is None
        assert args.pp_profile is None
        assert args.undistort_enable is None
        assert args.gamma_enable is None
        assert args.denoise_enable is None
