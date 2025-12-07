#!/usr/bin/env python3
"""Unit tests for configuration module."""
import pytest
from apps.config import (
    ModelConfig,
    RKNNConfig,
    PreprocessConfig,
    VideoConfig,
    NetworkConfig,
    PathConfig,
    CalibrationConfig,
    PerformanceConfig,
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


class TestVideoConfig:
    """Test suite for VideoConfig."""

    def test_default_camera(self):
        """Test default camera device index."""
        assert VideoConfig.DEFAULT_CAMERA == 0

    def test_default_fps(self):
        """Test default FPS value."""
        assert VideoConfig.DEFAULT_FPS == 30
        assert VideoConfig.DEFAULT_FPS > 0

    def test_default_resolution(self):
        """Test default video resolution."""
        assert VideoConfig.DEFAULT_WIDTH == 1920
        assert VideoConfig.DEFAULT_HEIGHT == 1080
        assert VideoConfig.DEFAULT_WIDTH > 0
        assert VideoConfig.DEFAULT_HEIGHT > 0

    def test_frame_buffer_size(self):
        """Test frame buffer size is positive."""
        assert VideoConfig.FRAME_BUFFER_SIZE > 0


class TestNetworkConfig:
    """Test suite for NetworkConfig."""

    def test_tcp_port_valid(self):
        """Test TCP port is in valid range."""
        assert 1 <= NetworkConfig.DEFAULT_TCP_PORT <= 65535

    def test_udp_port_valid(self):
        """Test UDP port is in valid range."""
        assert 1 <= NetworkConfig.DEFAULT_UDP_PORT <= 65535

    def test_http_port_valid(self):
        """Test HTTP port is in valid range."""
        assert 1 <= NetworkConfig.HTTP_DEFAULT_PORT <= 65535

    def test_max_content_length(self):
        """Test max content length is reasonable."""
        assert NetworkConfig.HTTP_MAX_CONTENT_LENGTH > 0
        assert NetworkConfig.HTTP_MAX_CONTENT_LENGTH == 10 * 1024 * 1024  # 10MB


class TestPathConfig:
    """Test suite for PathConfig."""

    def test_project_root_defined(self):
        """Test project root is defined."""
        assert PathConfig.PROJECT_ROOT is not None

    def test_models_dir_path(self):
        """Test models directory path."""
        assert 'models' in PathConfig.MODELS_DIR.lower() or 'artifacts' in PathConfig.MODELS_DIR.lower()

    def test_default_model_paths(self):
        """Test default model paths are defined."""
        assert PathConfig.DEFAULT_ONNX_MODEL is not None
        assert PathConfig.DEFAULT_RKNN_MODEL is not None
        assert '.onnx' in PathConfig.DEFAULT_ONNX_MODEL
        assert '.rknn' in PathConfig.DEFAULT_RKNN_MODEL

    def test_dataset_paths(self):
        """Test dataset paths are defined."""
        assert PathConfig.DATASETS_DIR is not None
        assert PathConfig.COCO_DIR is not None

    def test_output_paths(self):
        """Test output paths are defined."""
        assert PathConfig.ARTIFACTS_DIR is not None
        assert PathConfig.LOGS_DIR is not None


class TestCalibrationConfig:
    """Test suite for CalibrationConfig."""

    def test_min_calib_images(self):
        """Test minimum calibration images."""
        assert CalibrationConfig.MIN_CALIB_IMAGES > 0
        assert CalibrationConfig.MIN_CALIB_IMAGES == 100

    def test_recommended_calib_images(self):
        """Test recommended calibration images."""
        assert CalibrationConfig.RECOMMENDED_CALIB_IMAGES > CalibrationConfig.MIN_CALIB_IMAGES
        assert CalibrationConfig.RECOMMENDED_CALIB_IMAGES == 300

    def test_max_calib_images(self):
        """Test maximum calibration images."""
        assert CalibrationConfig.MAX_CALIB_IMAGES > CalibrationConfig.RECOMMENDED_CALIB_IMAGES
        assert CalibrationConfig.MAX_CALIB_IMAGES == 500

    def test_calib_image_order(self):
        """Test calibration image counts are in proper order."""
        assert CalibrationConfig.MIN_CALIB_IMAGES < CalibrationConfig.RECOMMENDED_CALIB_IMAGES
        assert CalibrationConfig.RECOMMENDED_CALIB_IMAGES < CalibrationConfig.MAX_CALIB_IMAGES


class TestPerformanceConfig:
    """Test suite for PerformanceConfig."""

    def test_target_latency(self):
        """Test target latency is reasonable."""
        assert PerformanceConfig.TARGET_LATENCY_MS > 0
        assert PerformanceConfig.TARGET_LATENCY_MS == 45

    def test_target_fps(self):
        """Test target FPS."""
        assert PerformanceConfig.TARGET_FPS > 0
        assert PerformanceConfig.TARGET_FPS == 30

    def test_batch_size_defaults(self):
        """Test batch size configuration."""
        assert PerformanceConfig.DEFAULT_BATCH_SIZE >= 1
        assert PerformanceConfig.MAX_BATCH_SIZE >= PerformanceConfig.DEFAULT_BATCH_SIZE

    def test_latency_fps_consistency(self):
        """Test that latency and FPS targets are consistent."""
        # At 30 FPS, each frame should take ~33ms
        # 45ms target allows some headroom
        max_latency_for_target_fps = 1000 / PerformanceConfig.TARGET_FPS
        assert PerformanceConfig.TARGET_LATENCY_MS <= max_latency_for_target_fps * 1.5


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
