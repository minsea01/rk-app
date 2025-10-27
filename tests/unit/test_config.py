#!/usr/bin/env python3
"""Unit tests for configuration module."""
import pytest
from apps.config import (
    ModelConfig,
    RKNNConfig,
    PreprocessConfig,
    get_detection_config,
    get_rknn_config
)


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_default_size(self):
        """Test default image size."""
        assert ModelConfig.DEFAULT_SIZE == 416

    def test_confidence_threshold(self):
        """Test confidence threshold values."""
        assert 0.0 <= ModelConfig.CONF_THRESHOLD_DEFAULT <= 1.0

    def test_iou_threshold(self):
        """Test IOU threshold values."""
        assert 0.0 <= ModelConfig.IOU_THRESHOLD_DEFAULT <= 1.0

    def test_pixel_range(self):
        """Test pixel value ranges."""
        assert ModelConfig.PIXEL_MIN == 0
        assert ModelConfig.PIXEL_MAX == 255
        assert ModelConfig.PIXEL_SCALE == 255.0

    def test_max_detections(self):
        """Test max detections for different sizes."""
        assert ModelConfig.MAX_DETECTIONS_416 > 0
        assert ModelConfig.MAX_DETECTIONS_640 > 0
        assert ModelConfig.MAX_DETECTIONS_640 > ModelConfig.MAX_DETECTIONS_416


class TestRKNNConfig:
    """Test suite for RKNNConfig."""

    def test_target_platform(self):
        """Test target platform setting."""
        assert RKNNConfig.TARGET_PLATFORM == 'rk3588'

    def test_optimization_level(self):
        """Test optimization level range."""
        assert 0 <= RKNNConfig.OPTIMIZATION_LEVEL <= 3

    def test_core_masks(self):
        """Test NPU core mask values."""
        assert RKNNConfig.CORE_MASK_ALL == 0x7
        assert RKNNConfig.CORE_MASK_CORE0 == 0x1


class TestPreprocessConfig:
    """Test suite for PreprocessConfig."""

    def test_mean_values(self):
        """Test mean values configuration."""
        assert len(PreprocessConfig.MEAN_BGR) == 3
        assert len(PreprocessConfig.MEAN_RGB) == 3

    def test_std_values(self):
        """Test std values configuration."""
        assert len(PreprocessConfig.STD_BGR) == 3
        assert len(PreprocessConfig.STD_RGB) == 3


class TestConfigHelpers:
    """Test suite for configuration helper functions."""

    def test_get_detection_config_default(self):
        """Test get_detection_config with default size."""
        config = get_detection_config()
        assert config['size'] == 416
        assert 'conf_threshold' in config
        assert 'iou_threshold' in config

    def test_get_detection_config_custom_size(self):
        """Test get_detection_config with custom size."""
        config = get_detection_config(size=640)
        assert config['size'] == 640
        assert config['max_detections'] == ModelConfig.MAX_DETECTIONS_640

    def test_get_detection_config_invalid_size(self):
        """Test get_detection_config with unsupported size."""
        with pytest.raises(ValueError):
            get_detection_config(size=320)

    def test_get_rknn_config(self):
        """Test get_rknn_config."""
        config = get_rknn_config()
        assert config['target_platform'] == RKNNConfig.TARGET_PLATFORM
        assert config['optimization_level'] == RKNNConfig.OPTIMIZATION_LEVEL
