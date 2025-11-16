#!/usr/bin/env python3
"""Unit tests for YOLO post-processing utilities."""
import pytest
import numpy as np
from apps.utils.yolo_post import (
    sigmoid,
    letterbox,
    make_anchors,
    dfl_decode,
    nms,
    postprocess_yolov8
)


class TestSigmoid:
    """Test suite for sigmoid activation function."""

    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        result = sigmoid(0.0)
        assert abs(result - 0.5) < 1e-6

    def test_sigmoid_positive_large(self):
        """Test sigmoid of large positive value approaches 1."""
        result = sigmoid(10.0)
        assert result > 0.99

    def test_sigmoid_negative_large(self):
        """Test sigmoid of large negative value approaches 0."""
        result = sigmoid(-10.0)
        assert result < 0.01

    def test_sigmoid_array(self):
        """Test sigmoid works with numpy arrays."""
        arr = np.array([0.0, 1.0, -1.0])
        result = sigmoid(arr)
        assert result.shape == arr.shape
        assert abs(result[0] - 0.5) < 1e-6
        assert result[1] > 0.7
        assert result[2] < 0.3

    def test_sigmoid_numerical_stability_extreme_positive(self):
        """Test sigmoid numerical stability with extreme positive values."""
        # These values would cause overflow in naive implementation
        x = np.array([100, 500, 1000, 10000])
        y = sigmoid(x)
        # Should not produce inf/nan
        assert np.all(np.isfinite(y))
        # Should be very close to 1.0 but not exceed it
        assert np.all(y > 0.9999)
        assert np.all(y <= 1.0)

    def test_sigmoid_numerical_stability_extreme_negative(self):
        """Test sigmoid numerical stability with extreme negative values."""
        # These values would cause overflow in naive implementation
        x = np.array([-100, -500, -1000, -10000])
        y = sigmoid(x)
        # Should not produce inf/nan
        assert np.all(np.isfinite(y))
        # Should be very close to 0.0 but not go negative
        assert np.all(y < 0.0001)
        assert np.all(y >= 0.0)

    def test_sigmoid_range_always_valid(self):
        """Test sigmoid always produces values in [0, 1]."""
        # Test with 10000 random values across extreme range
        x = np.random.randn(10000) * 1000
        y = sigmoid(x)
        assert np.all((y >= 0) & (y <= 1))
        assert np.all(np.isfinite(y))

    def test_sigmoid_no_overflow_underflow(self):
        """Test sigmoid handles overflow and underflow gracefully."""
        x = np.array([-1e10, -1e5, -100, -10, 0, 10, 100, 1e5, 1e10])
        y = sigmoid(x)
        # No inf, no nan
        assert np.all(np.isfinite(y))
        # Correct range
        assert np.all((y >= 0) & (y <= 1))


class TestLetterbox:
    """Test suite for letterbox image resizing."""

    def test_letterbox_square_image(self):
        """Test letterbox with square input image."""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result, ratio, (dw, dh) = letterbox(img, 640)
        assert result.shape == (640, 640, 3)
        assert abs(ratio - 1.0) < 1e-6
        assert dw == 0.0 and dh == 0.0

    def test_letterbox_aspect_ratio_preserved(self):
        """Test letterbox preserves aspect ratio."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result, ratio, (dw, dh) = letterbox(img, 640)
        assert result.shape == (640, 640, 3)
        # Aspect ratio should be preserved (width > height, so scale by height)
        assert ratio > 0

    def test_letterbox_padding_applied(self):
        """Test letterbox applies padding correctly."""
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result, ratio, (dw, dh) = letterbox(img, 640)
        assert result.shape == (640, 640, 3)
        # Should have padding since original is not square
        assert dw > 0 or dh > 0

    def test_letterbox_different_target_size(self):
        """Test letterbox with different target sizes."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for size in [320, 416, 640]:
            result, ratio, _ = letterbox(img, size)
            assert result.shape == (size, size, 3)

    def test_letterbox_very_small_image(self):
        """Test letterbox with very small input image."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result, ratio, (dw, dh) = letterbox(img, 640)
        assert result.shape == (640, 640, 3)
        assert ratio > 1.0  # Image was upscaled (ratio = new_size/orig_size)


class TestMakeAnchors:
    """Test suite for anchor generation."""

    def test_make_anchors_single_stride(self):
        """Test anchor generation with single stride."""
        strides = [8]
        img_size = 640
        anchors = make_anchors(strides, img_size)
        expected_count = (img_size // 8) ** 2
        assert anchors.shape == (expected_count, 2)

    def test_make_anchors_multiple_strides(self):
        """Test anchor generation with multiple strides."""
        strides = [8, 16, 32]
        img_size = 640
        anchors = make_anchors(strides, img_size)
        expected_count = (640//8)**2 + (640//16)**2 + (640//32)**2
        assert anchors.shape == (expected_count, 2)

    def test_make_anchors_coordinate_range(self):
        """Test anchors are within valid coordinate range."""
        strides = [8, 16, 32]
        img_size = 640
        anchors = make_anchors(strides, img_size)
        # All anchors should be within image bounds
        assert np.all(anchors[:, 0] >= 0) and np.all(anchors[:, 0] <= img_size)
        assert np.all(anchors[:, 1] >= 0) and np.all(anchors[:, 1] <= img_size)


class TestDflDecode:
    """Test suite for DFL (Distribution Focal Loss) decoding."""

    def test_dfl_decode_shape(self):
        """Test DFL decode output shape."""
        N = 8400
        reg_max = 16
        d = np.random.randn(N, 4 * reg_max).astype(np.float32)
        result = dfl_decode(d, reg_max=reg_max)
        assert result.shape == (N, 4)

    def test_dfl_decode_output_range(self):
        """Test DFL decode output is in valid range."""
        N = 100
        reg_max = 16
        d = np.random.randn(N, 4 * reg_max).astype(np.float32)
        result = dfl_decode(d, reg_max=reg_max)
        # Output should be in range [0, reg_max-1]
        assert np.all(result >= 0)
        assert np.all(result < reg_max)

    def test_dfl_decode_different_reg_max(self):
        """Test DFL decode with different reg_max values."""
        N = 100
        for reg_max in [8, 16, 24]:
            d = np.random.randn(N, 4 * reg_max).astype(np.float32)
            result = dfl_decode(d, reg_max=reg_max)
            assert result.shape == (N, 4)


class TestNMS:
    """Test suite for Non-Maximum Suppression."""

    def test_nms_removes_overlapping_boxes(self):
        """Test NMS removes highly overlapping boxes."""
        boxes = np.array([
            [0, 0, 100, 100],
            [5, 5, 105, 105],  # High overlap with first
            [200, 200, 300, 300]  # No overlap
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.85])
        keep = nms(boxes, scores, iou_thres=0.5)
        # Should keep box 0 (highest score) and box 2 (no overlap)
        assert len(keep) == 2
        assert 0 in keep
        assert 2 in keep

    def test_nms_keeps_all_non_overlapping(self):
        """Test NMS keeps all boxes when no overlap."""
        boxes = np.array([
            [0, 0, 50, 50],
            [100, 100, 150, 150],
            [200, 200, 250, 250]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_thres=0.5)
        assert len(keep) == 3

    def test_nms_respects_confidence_order(self):
        """Test NMS keeps highest confidence box."""
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110],  # High overlap, higher confidence
        ], dtype=np.float32)
        scores = np.array([0.7, 0.9])
        keep = nms(boxes, scores, iou_thres=0.5)
        assert len(keep) == 1
        assert keep[0] == 1  # Should keep second box (higher confidence)

    def test_nms_topk_limit(self):
        """Test NMS respects topk parameter."""
        boxes = np.random.rand(100, 4).astype(np.float32) * 100
        scores = np.random.rand(100).astype(np.float32)
        keep = nms(boxes, scores, iou_thres=0.5, topk=10)
        assert len(keep) <= 10

    def test_nms_empty_input(self):
        """Test NMS with empty input."""
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.array([])
        keep = nms(boxes, scores, iou_thres=0.5)
        assert len(keep) == 0

    def test_nms_iou_calculation_accuracy(self):
        """Test NMS IoU calculation is accurate for floating-point coordinates."""
        # Two boxes with exact 50% overlap
        # Box 1: [0, 0, 100, 100] - area = 10000
        # Box 2: [50, 0, 150, 100] - area = 10000
        # Intersection: [50, 0, 100, 100] - area = 5000
        # Union: 10000 + 10000 - 5000 = 15000
        # IoU = 5000 / 15000 = 0.3333
        boxes = np.array([
            [0.0, 0.0, 100.0, 100.0],
            [50.0, 0.0, 150.0, 100.0]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])

        # With IoU threshold 0.3, should suppress second box
        keep_low = nms(boxes, scores, iou_thres=0.3)
        assert len(keep_low) == 1

        # With IoU threshold 0.4, should keep both boxes
        keep_high = nms(boxes, scores, iou_thres=0.4)
        assert len(keep_high) == 2

    def test_nms_floating_point_precision(self):
        """Test NMS handles floating-point coordinates correctly."""
        # Boxes with fractional coordinates
        boxes = np.array([
            [10.5, 20.3, 50.7, 60.2],
            [15.2, 25.8, 55.1, 65.9]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])

        keep = nms(boxes, scores, iou_thres=0.5)
        # Should work without errors
        assert len(keep) >= 1

    def test_nms_small_boxes_not_over_suppressed(self):
        """Test NMS doesn't over-suppress small boxes due to area calculation."""
        # Small boxes where +1 correction would have significant impact
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],      # Area = 100
            [2.0, 2.0, 12.0, 12.0],      # Area = 100
            # Intersection: [2, 2, 10, 10] = 64
            # IoU = 64 / (100 + 100 - 64) = 64/136 = 0.47
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])

        # With correct calculation, IoU ~ 0.47, should keep both at threshold 0.5
        keep = nms(boxes, scores, iou_thres=0.5)
        assert len(keep) == 2


class TestPostprocessYolov8:
    """Test suite for YOLOv8 post-processing."""

    def test_postprocess_output_types(self):
        """Test postprocess returns correct output types."""
        # Create mock prediction (1, N, 64+nc)
        # N must match anchor grid for 640: (640/8)^2 + (640/16)^2 + (640/32)^2 = 8400
        N = 8400
        nc = 80  # COCO classes
        preds = np.random.randn(1, N, 64 + nc).astype(np.float32)
        boxes, confs, class_ids = postprocess_yolov8(
            preds, 640, (480, 640), (1.0, (0, 0)),
            conf_thres=0.5, iou_thres=0.45
        )
        assert isinstance(boxes, np.ndarray)
        assert isinstance(confs, np.ndarray)
        assert isinstance(class_ids, np.ndarray)

    def test_postprocess_output_shapes(self):
        """Test postprocess output shapes are consistent."""
        N = 8400  # Match anchor grid for 640
        nc = 80
        preds = np.random.randn(1, N, 64 + nc).astype(np.float32)
        boxes, confs, class_ids = postprocess_yolov8(
            preds, 640, (480, 640), (1.0, (0, 0)),
            conf_thres=0.5, iou_thres=0.45
        )
        # All outputs should have same length
        assert len(boxes) == len(confs) == len(class_ids)
        # Boxes should be (M, 4)
        if len(boxes) > 0:
            assert boxes.shape[1] == 4

    def test_postprocess_confidence_filtering(self):
        """Test postprocess filters by confidence threshold."""
        N = 8400  # Match anchor grid for 640
        nc = 80
        # Create predictions with known low confidences
        preds = np.random.randn(1, N, 64 + nc).astype(np.float32) - 10  # Very negative = low conf
        boxes, confs, class_ids = postprocess_yolov8(
            preds, 640, (480, 640), (1.0, (0, 0)),
            conf_thres=0.9, iou_thres=0.45
        )
        # Should filter out most/all detections
        assert len(boxes) == 0 or np.all(confs >= 0.9)

    def test_postprocess_invalid_input_shape(self):
        """Test postprocess raises error for invalid input."""
        # Wrong number of dimensions
        preds = np.random.randn(100, 144).astype(np.float32)
        with pytest.raises(AssertionError):
            postprocess_yolov8(
                preds, 640, (480, 640), (1.0, (0, 0)),
                conf_thres=0.25, iou_thres=0.45
            )

    def test_postprocess_box_coordinates_valid(self):
        """Test postprocess produces valid box coordinates."""
        N = 8400  # Match anchor grid for 640
        nc = 80
        preds = np.random.randn(1, N, 64 + nc).astype(np.float32)
        orig_h, orig_w = 480, 640
        boxes, confs, class_ids = postprocess_yolov8(
            preds, 640, (orig_h, orig_w), (1.0, (0, 0)),
            conf_thres=0.25, iou_thres=0.45
        )
        if len(boxes) > 0:
            # x1, y1, x2, y2 should be within original image bounds
            assert np.all(boxes[:, 0] >= 0) and np.all(boxes[:, 0] < orig_w)
            assert np.all(boxes[:, 1] >= 0) and np.all(boxes[:, 1] < orig_h)
            assert np.all(boxes[:, 2] >= 0) and np.all(boxes[:, 2] < orig_w)
            assert np.all(boxes[:, 3] >= 0) and np.all(boxes[:, 3] < orig_h)
            # x2 > x1, y2 > y1
            assert np.all(boxes[:, 2] >= boxes[:, 0])
            assert np.all(boxes[:, 3] >= boxes[:, 1])
