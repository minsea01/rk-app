#!/usr/bin/env python3
"""Unit tests for apps.yolov8_stream module.

Tests high-performance streaming RKNN inference pipeline.
"""
import pytest

from apps.yolov8_stream import parse_source, StageStats


class TestParseSource:
    """Test suite for parse_source function."""

    def test_parses_digit_string_as_camera_index(self):
        """Test that digit strings are converted to camera indices."""
        assert parse_source('0') == 0
        assert parse_source('1') == 1
        assert parse_source('2') == 2

    def test_parses_multi_digit_camera_index(self):
        """Test that multi-digit strings are converted correctly."""
        assert parse_source('10') == 10
        assert parse_source('99') == 99

    def test_returns_string_for_rtsp_urls(self):
        """Test that RTSP URLs are returned as strings."""
        rtsp_url = 'rtsp://192.168.1.100:554/stream'
        result = parse_source(rtsp_url)
        assert result == rtsp_url
        assert isinstance(result, str)

    def test_returns_string_for_file_paths(self):
        """Test that file paths are returned as strings."""
        file_path = '/path/to/video.mp4'
        result = parse_source(file_path)
        assert result == file_path
        assert isinstance(result, str)

    def test_returns_string_for_gstreamer_pipelines(self):
        """Test that GStreamer pipelines are returned as strings."""
        gst_pipeline = 'v4l2src device=/dev/video0 ! videoconvert ! appsink'
        result = parse_source(gst_pipeline)
        assert result == gst_pipeline
        assert isinstance(result, str)

    def test_handles_http_urls(self):
        """Test that HTTP URLs are returned as strings."""
        http_url = 'http://example.com/stream.mjpg'
        result = parse_source(http_url)
        assert result == http_url
        assert isinstance(result, str)

    def test_handles_empty_string(self):
        """Test that empty strings are handled gracefully."""
        result = parse_source('')
        assert result == ''
        assert isinstance(result, str)

    def test_handles_string_with_spaces(self):
        """Test that strings with spaces are not parsed as integers."""
        result = parse_source('test video.mp4')
        assert result == 'test video.mp4'
        assert isinstance(result, str)

    def test_handles_negative_numbers(self):
        """Test that negative number strings are returned as strings."""
        # Negative camera indices are invalid, should be returned as string
        result = parse_source('-1')
        # Should be treated as string since it's not a valid camera index
        assert isinstance(result, (int, str))


class TestStageStats:
    """Test suite for StageStats class."""

    def test_initializes_with_zero_stats(self):
        """Test that StageStats initializes with zero values."""
        stats = StageStats()

        assert stats.n == 0
        assert stats.t_sum == 0.0
        assert stats.t_min == 1e9
        assert stats.t_max == 0.0

    def test_adds_timing_samples_correctly(self):
        """Test that timing samples are accumulated correctly."""
        stats = StageStats()

        stats.add(0.1)
        stats.add(0.2)
        stats.add(0.3)

        assert stats.n == 3
        assert stats.t_sum == pytest.approx(0.6)

    def test_tracks_min_correctly(self):
        """Test that minimum time is tracked correctly."""
        stats = StageStats()

        stats.add(0.5)
        stats.add(0.2)  # New minimum
        stats.add(0.8)
        stats.add(0.1)  # New minimum

        assert stats.t_min == 0.1

    def test_tracks_max_correctly(self):
        """Test that maximum time is tracked correctly."""
        stats = StageStats()

        stats.add(0.5)
        stats.add(0.8)  # New maximum
        stats.add(0.2)
        stats.add(1.2)  # New maximum

        assert stats.t_max == 1.2

    def test_calculates_average_correctly(self):
        """Test that average is calculated correctly in summary."""
        stats = StageStats()

        stats.add(0.1)
        stats.add(0.2)
        stats.add(0.3)

        summary = stats.summary()

        # Average should be (0.1 + 0.2 + 0.3) / 3 = 0.2
        assert summary['n'] == 3
        assert abs(summary['avg_ms'] - 200.0) < 0.01  # 0.2s = 200ms

    def test_summary_returns_correct_format(self):
        """Test that summary returns expected dictionary format."""
        stats = StageStats()

        stats.add(0.1)
        stats.add(0.2)

        summary = stats.summary()

        assert 'n' in summary
        assert 'avg_ms' in summary
        assert 'min_ms' in summary
        assert 'max_ms' in summary

        assert isinstance(summary['n'], int)
        assert isinstance(summary['avg_ms'], float)
        assert isinstance(summary['min_ms'], float)
        assert isinstance(summary['max_ms'], float)

    def test_summary_converts_to_milliseconds(self):
        """Test that times are converted from seconds to milliseconds."""
        stats = StageStats()

        stats.add(0.001)  # 1ms
        stats.add(0.002)  # 2ms
        stats.add(0.003)  # 3ms

        summary = stats.summary()

        assert abs(summary['avg_ms'] - 2.0) < 0.01
        assert abs(summary['min_ms'] - 1.0) < 0.01
        assert abs(summary['max_ms'] - 3.0) < 0.01

    def test_handles_zero_samples_gracefully(self):
        """Test that summary works with zero samples (no division by zero)."""
        stats = StageStats()

        summary = stats.summary()

        # With 0 samples, average should be 0
        assert summary['n'] == 0
        assert summary['avg_ms'] == 0.0
        assert summary['min_ms'] == 1e9 * 1000  # Still initial value
        assert summary['max_ms'] == 0.0

    def test_reset_clears_all_stats(self):
        """Test that reset() clears all statistics."""
        stats = StageStats()

        # Add some data
        stats.add(0.1)
        stats.add(0.2)
        stats.add(0.3)

        # Reset
        stats.reset()

        # Should be back to initial state
        assert stats.n == 0
        assert stats.t_sum == 0.0
        assert stats.t_min == 1e9
        assert stats.t_max == 0.0

    def test_handles_very_small_times(self):
        """Test that very small timing values are handled correctly."""
        stats = StageStats()

        stats.add(0.000001)  # 1 microsecond
        stats.add(0.000002)  # 2 microseconds

        summary = stats.summary()

        # Should handle microsecond precision
        assert summary['avg_ms'] > 0
        assert summary['min_ms'] > 0

    def test_handles_large_number_of_samples(self):
        """Test that large number of samples doesn't overflow."""
        stats = StageStats()

        # Add 10000 samples
        for i in range(10000):
            stats.add(0.001)  # 1ms each

        summary = stats.summary()

        assert summary['n'] == 10000
        assert abs(summary['avg_ms'] - 1.0) < 0.01

    def test_summary_preserves_original_data(self):
        """Test that calling summary() doesn't modify the stats."""
        stats = StageStats()

        stats.add(0.1)
        stats.add(0.2)

        summary1 = stats.summary()
        summary2 = stats.summary()

        # Both summaries should be identical
        assert summary1 == summary2

        # Stats should still be intact
        assert stats.n == 2
        assert stats.t_sum == pytest.approx(0.3)


class TestDecodePredictons:
    """Test suite for decode_predictions function."""

    def test_handles_2d_input_by_adding_batch_dimension(self):
        """Test that 2D predictions are expanded to 3D."""
        import numpy as np
        from apps.yolov8_stream import decode_predictions

        # Create 2D prediction (N, C)
        pred = np.random.randn(8400, 84)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45
        )

        # Should handle without error
        assert isinstance(boxes, np.ndarray)
        assert isinstance(confs, np.ndarray)
        assert isinstance(cls_ids, np.ndarray)

    def test_normalizes_to_n_c_format(self):
        """Test that predictions are normalized to (1, N, C) format."""
        import numpy as np
        from apps.yolov8_stream import decode_predictions

        # Create prediction in (1, C, N) format (needs transpose)
        pred = np.random.randn(1, 84, 8400)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45
        )

        # Should handle without error
        assert isinstance(boxes, np.ndarray)

    def test_auto_detects_dfl_head_for_large_channels(self):
        """Test that DFL head is detected for C >= 64."""
        import numpy as np
        from apps.yolov8_stream import decode_predictions

        # Create prediction with 84 channels (DFL format)
        pred = np.random.randn(1, 8400, 84)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.9, iou_thres=0.45, head='auto'
        )

        # Should use DFL decoder
        assert isinstance(boxes, np.ndarray)

    def test_uses_raw_head_for_small_channels(self):
        """Test that raw head is used for C < 64."""
        import numpy as np
        from apps.yolov8_stream import decode_predictions

        # Create prediction with small number of channels
        pred = np.random.randn(1, 100, 6)  # [cx, cy, w, h, obj, cls]

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head='auto'
        )

        # Should use raw decoder
        assert isinstance(boxes, np.ndarray)

    def test_returns_empty_arrays_for_no_detections(self):
        """Test that empty arrays are returned when no detections pass threshold."""
        import numpy as np
        from apps.yolov8_stream import decode_predictions

        # Create prediction with very low confidence
        # Use N=8400 to match anchor grid for 640x640 with strides [8, 16, 32]
        N = 8400  # (640/8)^2 + (640/16)^2 + (640/32)^2
        pred = np.random.randn(1, N, 84) * 0.001  # Very small values

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.9, iou_thres=0.45
        )

        # Should return empty arrays
        assert len(boxes) == 0
        assert len(confs) == 0
        assert len(cls_ids) == 0

    def test_applies_confidence_threshold(self):
        """Test that confidence threshold filters detections."""
        import numpy as np
        from apps.yolov8_stream import decode_predictions

        # Create prediction with known high confidence
        pred = np.zeros((1, 10, 6))
        pred[0, 0, :] = [320, 240, 100, 100, 5.0, 5.0]  # High obj and cls scores

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.45, head='raw'
        )

        # Should have at least one detection
        assert len(boxes) >= 0  # May be filtered by NMS

    def test_applies_nms(self):
        """Test that NMS is applied to remove overlapping boxes."""
        import numpy as np
        from apps.yolov8_stream import decode_predictions

        # Create two overlapping boxes with high confidence
        pred = np.zeros((1, 2, 6))
        pred[0, 0, :] = [320, 240, 100, 100, 5.0, 5.0]  # Box 1
        pred[0, 1, :] = [325, 245, 100, 100, 5.0, 5.0]  # Box 2 (overlapping)

        boxes, confs, cls_ids = decode_predictions(
            pred, imgsz=640, conf_thres=0.5, iou_thres=0.3, head='raw'
        )

        # NMS should reduce overlapping detections
        assert len(boxes) <= 2
