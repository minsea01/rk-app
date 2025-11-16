#!/usr/bin/env python3
"""Unit tests for decode_predictions function."""
import pytest
import numpy as np
from apps.yolov8_rknn_infer import decode_predictions, load_labels, draw_boxes
import tempfile
from pathlib import Path
import cv2


class TestDecodePredictions:
    """Test suite for decode_predictions function."""

    def test_decode_predictions_auto_dfl_detection(self):
        """Test auto-detection of DFL head (C >= 64)."""
        # Create prediction with DFL head (84 channels for COCO)
        N = 8400  # Match anchor grid for 640
        C = 84  # 64 (DFL) + 20 (classes)
        pred = np.random.randn(1, N, C).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head='auto'
        )

        # Should return valid outputs
        assert isinstance(boxes, np.ndarray)
        assert isinstance(confs, np.ndarray)
        assert isinstance(cls_ids, np.ndarray)

    def test_decode_predictions_auto_raw_detection(self):
        """Test auto-detection of raw head (C < 64)."""
        # Create prediction with raw head (5 channels: cx, cy, w, h, obj)
        N = 100
        C = 5
        pred = np.random.randn(1, N, C).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head='auto'
        )

        assert isinstance(boxes, np.ndarray)
        assert isinstance(confs, np.ndarray)
        assert isinstance(cls_ids, np.ndarray)

    def test_decode_predictions_2d_input(self):
        """Test decode_predictions handles 2D input (N, C)."""
        N = 8400  # Match anchor grid for 640
        C = 84
        pred = np.random.randn(N, C).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head='dfl'
        )

        # Should handle 2D input by adding batch dimension
        assert len(boxes) == len(confs) == len(cls_ids)

    def test_decode_predictions_transpose_handling(self):
        """Test decode_predictions handles transposed input (1, C, N)."""
        N = 8400  # Match anchor grid for 640
        C = 84
        # Create transposed input (channels first)
        pred = np.random.randn(1, C, N).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head='dfl'
        )

        # Should correctly transpose to (1, N, C)
        assert isinstance(boxes, np.ndarray)

    def test_decode_predictions_dfl_mode_explicit(self):
        """Test decode_predictions with explicit DFL mode."""
        N = 8400  # Match anchor grid for 640
        C = 84
        pred = np.random.randn(1, N, C).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head='dfl'
        )

        if len(boxes) > 0:
            assert boxes.shape[1] == 4  # xyxy format

    def test_decode_predictions_raw_mode_explicit(self):
        """Test decode_predictions with explicit raw mode."""
        N = 100
        C = 85  # cx, cy, w, h, obj, 80 classes
        pred = np.random.randn(1, N, C).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head='raw'
        )

        if len(boxes) > 0:
            assert boxes.shape[1] == 4

    def test_decode_predictions_confidence_threshold(self):
        """Test decode_predictions respects confidence threshold."""
        N = 8400  # Match anchor grid for 640
        C = 84
        # Create predictions with very negative values (low confidence)
        pred = np.random.randn(1, N, C).astype(np.float32) - 10

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.95, iou_thres=0.45, head='dfl'
        )

        # Should filter out most/all low-confidence detections
        assert len(boxes) == 0 or np.all(confs >= 0.95)

    def test_decode_predictions_with_ratio_pad(self):
        """Test decode_predictions with custom ratio and padding."""
        N = 8400  # Match anchor grid for 640
        C = 84
        pred = np.random.randn(1, N, C).astype(np.float32)

        ratio_pad = (0.75, (50.0, 100.0))  # Custom ratio and padding
        orig_shape = (480, 640)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.25, iou_thres=0.45,
            head='dfl', ratio_pad=ratio_pad, orig_shape=orig_shape
        )

        # Boxes should be scaled to original image coordinates
        if len(boxes) > 0:
            assert np.all(boxes[:, 0::2] >= 0)  # x coords
            assert np.all(boxes[:, 1::2] >= 0)  # y coords

    def test_decode_predictions_raw_no_classes(self):
        """Test raw mode with no class scores (C=5)."""
        N = 100
        C = 5  # cx, cy, w, h, obj only
        pred = np.random.randn(1, N, C).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.25, iou_thres=0.45, head='raw'
        )

        # Should handle case with no class scores
        if len(boxes) > 0:
            assert len(cls_ids) == len(boxes)
            # Class IDs should default to 0
            assert np.all(cls_ids == 0)

    def test_decode_predictions_very_low_channel_count(self):
        """Test decode_predictions with too few channels."""
        N = 100
        C = 3  # Less than minimum 5 for raw head
        pred = np.random.randn(1, N, C).astype(np.float32)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.25, iou_thres=0.45, head='raw'
        )

        # Should return empty results for invalid input
        assert len(boxes) == 0
        assert len(confs) == 0
        assert len(cls_ids) == 0


class TestLoadLabels:
    """Test suite for load_labels function."""

    def test_load_labels_valid_file(self):
        """Test loading labels from valid file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('person\n')
            f.write('bicycle\n')
            f.write('car\n')
            f.flush()
            path = Path(f.name)

        try:
            labels = load_labels(path)
            assert labels == ['person', 'bicycle', 'car']
        finally:
            path.unlink()

    def test_load_labels_empty_lines(self):
        """Test load_labels skips empty lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('person\n')
            f.write('\n')
            f.write('bicycle\n')
            f.write('  \n')
            f.write('car\n')
            f.flush()
            path = Path(f.name)

        try:
            labels = load_labels(path)
            assert labels == ['person', 'bicycle', 'car']
        finally:
            path.unlink()

    def test_load_labels_nonexistent_file(self):
        """Test load_labels with nonexistent file."""
        labels = load_labels(Path('/nonexistent/file.txt'))
        assert labels is None

    def test_load_labels_none_path(self):
        """Test load_labels with None path."""
        labels = load_labels(None)
        assert labels is None

    def test_load_labels_strips_whitespace(self):
        """Test load_labels strips whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('  person  \n')
            f.write('\tbicycle\t\n')
            f.write('car   \n')
            f.flush()
            path = Path(f.name)

        try:
            labels = load_labels(path)
            assert labels == ['person', 'bicycle', 'car']
        finally:
            path.unlink()


class TestDrawBoxes:
    """Test suite for draw_boxes function."""

    def test_draw_boxes_returns_image(self):
        """Test draw_boxes returns an image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confs = np.array([0.95])
        class_ids = np.array([0])

        result = draw_boxes(img, boxes, confs, class_ids)

        assert isinstance(result, np.ndarray)
        assert result.shape == img.shape

    def test_draw_boxes_with_names(self):
        """Test draw_boxes with class names."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confs = np.array([0.95])
        class_ids = np.array([0])
        names = ['person', 'bicycle', 'car']

        result = draw_boxes(img, boxes, confs, class_ids, names)

        assert isinstance(result, np.ndarray)
        # Image should be modified (not all zeros anymore)
        assert not np.all(result == 0)

    def test_draw_boxes_multiple_detections(self):
        """Test draw_boxes with multiple detections."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.array([
            [10, 10, 100, 100],
            [200, 200, 300, 300],
            [400, 50, 500, 150]
        ], dtype=np.float32)
        confs = np.array([0.95, 0.85, 0.75])
        class_ids = np.array([0, 1, 2])

        result = draw_boxes(img, boxes, confs, class_ids)

        assert isinstance(result, np.ndarray)
        assert result.shape == img.shape

    def test_draw_boxes_empty_detections(self):
        """Test draw_boxes with no detections."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.empty((0, 4), dtype=np.float32)
        confs = np.array([])
        class_ids = np.array([])

        result = draw_boxes(img, boxes, confs, class_ids)

        # Should return image unchanged
        assert np.array_equal(result, img)

    def test_draw_boxes_out_of_range_class_id(self):
        """Test draw_boxes handles out-of-range class IDs."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confs = np.array([0.95])
        class_ids = np.array([999])  # Out of range
        names = ['person', 'bicycle', 'car']

        # Should not crash, falls back to numeric label
        result = draw_boxes(img, boxes, confs, class_ids, names)
        assert isinstance(result, np.ndarray)
