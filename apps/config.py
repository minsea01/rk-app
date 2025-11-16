#!/usr/bin/env python3
"""Unified configuration management for rk-app.

This module centralizes all configuration parameters and magic numbers
used throughout the application, making it easier to tune and maintain.
"""


class ModelConfig:
    """Model inference and preprocessing configuration."""

    # Image sizes
    TARGET_SIZE_416 = 416
    TARGET_SIZE_640 = 640
    DEFAULT_SIZE = 416  # Preferred for RK3588 to avoid Transpose CPU fallback

    # Inference thresholds
    # Confidence threshold: 0.5 recommended for production
    # - conf=0.25: 3135ms postprocessing (NMS bottleneck) → 0.3 FPS
    # - conf=0.5:  5.2ms postprocessing → 60+ FPS
    CONF_THRESHOLD_DEFAULT = 0.5   # Confidence threshold for object detection
    IOU_THRESHOLD_DEFAULT = 0.45   # IOU threshold for NMS
    NMS_IOU = 0.45                 # NMS IOU threshold

    # Detection limits
    MAX_DETECTIONS_640 = 8400      # Max detections for 640×640 input
    MAX_DETECTIONS_416 = 3549      # Max detections for 416×416 input

    # Pixel value ranges
    PIXEL_MIN = 0
    PIXEL_MAX = 255
    PIXEL_SCALE = 255.0

    # YOLO head parameters
    HEAD_AUTO = 'auto'
    HEAD_DFL = 'dfl'
    HEAD_RAW = 'raw'
    DFL_THRESHOLD = 64  # If C >= 64, use DFL decoder


class RKNNConfig:
    """RKNN runtime configuration."""

    # Target platform
    TARGET_PLATFORM = 'rk3588'

    # Optimization level
    OPTIMIZATION_LEVEL = 3  # 0-3, higher means more aggressive optimization

    # NPU core masks
    CORE_MASK_ALL = 0x7      # Use all 3 NPU cores
    CORE_MASK_CORE0 = 0x1    # Use only core 0
    CORE_MASK_CORE1 = 0x2    # Use only core 1
    CORE_MASK_CORE2 = 0x4    # Use only core 2

    # Quantization
    QUANTIZED_DTYPE_W8A8 = 'w8a8'  # INT8 weights and activations (rknn-toolkit2 >=2.x)
    QUANTIZED_DTYPE_U8 = 'asymmetric_quantized-u8'  # rknn-toolkit2 1.x


class PreprocessConfig:
    """Image preprocessing configuration."""

    # Mean and std values for normalization
    MEAN_BGR = [0, 0, 0]       # Default: no mean subtraction
    STD_BGR = [255, 255, 255]  # Default: scale by 1/255

    MEAN_RGB = [0, 0, 0]
    STD_RGB = [255, 255, 255]

    # Color channel reordering
    REORDER_BGR_TO_RGB = '2 1 0'  # BGR -> RGB channel reordering


class VideoConfig:
    """Video capture and processing configuration."""

    # Default camera device
    DEFAULT_CAMERA = 0  # /dev/video0

    # Video formats
    DEFAULT_FPS = 30
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080

    # Buffer settings
    FRAME_BUFFER_SIZE = 10


class NetworkConfig:
    """Network streaming configuration."""

    # TCP/UDP ports
    DEFAULT_TCP_PORT = 8080
    DEFAULT_UDP_PORT = 8081

    # HTTP server
    HTTP_DEFAULT_PORT = 8081
    HTTP_MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB


class PathConfig:
    """File path configuration."""

    # Model paths
    DEFAULT_MODEL_DIR = 'artifacts/models'
    DEFAULT_ONNX_MODEL = 'artifacts/models/yolo11n_416.onnx'
    DEFAULT_RKNN_MODEL = 'artifacts/models/yolo11n.rknn'

    # Dataset paths
    DEFAULT_DATASET_DIR = 'datasets'
    COCO_CALIB_DIR = 'datasets/coco/calib_images'
    COCO_CALIB_FILE = 'datasets/coco/calib_images/calib.txt'

    # Output paths
    ARTIFACTS_DIR = 'artifacts'
    LOGS_DIR = 'logs'


class CalibrationConfig:
    """Model calibration configuration."""

    # Number of calibration images
    MIN_CALIB_IMAGES = 100
    RECOMMENDED_CALIB_IMAGES = 300
    MAX_CALIB_IMAGES = 500


class PerformanceConfig:
    """Performance tuning configuration."""

    # Latency targets (milliseconds)
    TARGET_LATENCY_MS = 45  # End-to-end target for RK3588

    # Throughput
    TARGET_FPS = 30

    # Batch processing
    DEFAULT_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 4


# Convenience function to get common configurations
def get_detection_config(size=416):
    """Get detection configuration for a specific input size.

    Args:
        size: Input image size (currently supports 416 or 640)

    Returns:
        dict: Configuration dictionary
    """
    supported_sizes = {
        ModelConfig.TARGET_SIZE_416: ModelConfig.MAX_DETECTIONS_416,
        ModelConfig.TARGET_SIZE_640: ModelConfig.MAX_DETECTIONS_640,
    }
    if size not in supported_sizes:
        raise ValueError(f"Unsupported detection size: {size}. Supported sizes: {list(supported_sizes.keys())}")

    return {
        'size': size,
        'conf_threshold': ModelConfig.CONF_THRESHOLD_DEFAULT,
        'iou_threshold': ModelConfig.IOU_THRESHOLD_DEFAULT,
        'max_detections': supported_sizes[size],
    }


def get_rknn_config():
    """Get default RKNN configuration.

    Returns:
        dict: RKNN configuration dictionary
    """
    return {
        'target_platform': RKNNConfig.TARGET_PLATFORM,
        'optimization_level': RKNNConfig.OPTIMIZATION_LEVEL,
        'core_mask': RKNNConfig.CORE_MASK_ALL,
        'quantized_dtype': RKNNConfig.QUANTIZED_DTYPE_W8A8,
    }
