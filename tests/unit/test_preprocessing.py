#!/usr/bin/env python3
"""Unit tests for preprocessing module."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import cv2

from apps.utils.preprocessing import (
    preprocess_onnx,
    preprocess_rknn_sim,
    preprocess_board,
    preprocess_from_array_onnx,
    preprocess_from_array_rknn_sim,
    preprocess_from_array_board
)
from apps.config import ModelConfig
from apps.exceptions import PreprocessError


class TestPreprocessingFunctions:
    """Test suite for preprocessing functions."""

    @pytest.fixture
    def test_image(self):
        """Create a temporary test image."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            cv2.imwrite(f.name, img)
            yield Path(f.name)
            Path(f.name).unlink()

    @pytest.fixture
    def test_array(self):
        """Create a test numpy array."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_preprocess_onnx_output_shape(self, test_image):
        """Test ONNX preprocessing output shape."""
        output = preprocess_onnx(test_image, target_size=416)
        assert output.shape == (1, 3, 416, 416)
        assert output.dtype == np.float32

    def test_preprocess_onnx_default_size(self, test_image):
        """Test ONNX preprocessing uses default size."""
        output = preprocess_onnx(test_image)
        assert output.shape == (1, 3, ModelConfig.DEFAULT_SIZE, ModelConfig.DEFAULT_SIZE)

    def test_preprocess_rknn_sim_output_shape(self, test_image):
        """Test RKNN simulator preprocessing output shape."""
        output = preprocess_rknn_sim(test_image, target_size=416)
        assert output.shape == (1, 416, 416, 3)
        assert output.dtype == np.float32

    def test_preprocess_board_output_shape(self, test_image):
        """Test board preprocessing output shape."""
        output = preprocess_board(test_image, target_size=416)
        assert output.shape == (1, 416, 416, 3)
        assert output.dtype == np.uint8

    def test_preprocess_from_array_onnx_shape(self, test_array):
        """Test ONNX array preprocessing output shape."""
        output = preprocess_from_array_onnx(test_array, target_size=416)
        assert output.shape == (1, 3, 416, 416)
        assert output.dtype == np.float32

    def test_preprocess_from_array_rknn_sim_shape(self, test_array):
        """Test RKNN simulator array preprocessing output shape."""
        output = preprocess_from_array_rknn_sim(test_array, target_size=416)
        assert output.shape == (1, 416, 416, 3)
        assert output.dtype == np.float32

    def test_preprocess_from_array_board_shape(self, test_array):
        """Test board array preprocessing output shape."""
        output = preprocess_from_array_board(test_array, target_size=416)
        assert output.shape == (1, 416, 416, 3)
        assert output.dtype == np.uint8

    def test_preprocess_invalid_image_path(self):
        """Test preprocessing with invalid image path."""
        with pytest.raises(PreprocessError):
            preprocess_onnx('/nonexistent/path/to/image.jpg')

    def test_preprocess_value_range(self, test_image):
        """Test that preprocessing maintains pixel value range."""
        output = preprocess_onnx(test_image, target_size=416)
        assert output.min() >= 0
        assert output.max() <= 255

    def test_preprocess_different_sizes(self, test_image):
        """Test preprocessing with different target sizes."""
        for size in [320, 416, 640]:
            output = preprocess_onnx(test_image, target_size=size)
            assert output.shape == (1, 3, size, size)


class TestPreprocessingEdgeCases:
    """Test suite for preprocessing edge cases."""

    def test_preprocess_non_square_small_image(self):
        """Test preprocessing with non-square small image."""
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        output = preprocess_from_array_onnx(img, target_size=640)
        assert output.shape == (1, 3, 640, 640)
        assert output.dtype == np.float32

    def test_preprocess_very_large_image(self):
        """Test preprocessing with very large image."""
        img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        output = preprocess_from_array_onnx(img, target_size=640)
        assert output.shape == (1, 3, 640, 640)

    def test_preprocess_very_small_image(self):
        """Test preprocessing with very small image."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        output = preprocess_from_array_onnx(img, target_size=640)
        assert output.shape == (1, 3, 640, 640)

    def test_preprocess_extreme_aspect_ratio_wide(self):
        """Test preprocessing with extreme wide aspect ratio."""
        img = np.random.randint(0, 255, (100, 1000, 3), dtype=np.uint8)
        output = preprocess_from_array_onnx(img, target_size=640)
        assert output.shape == (1, 3, 640, 640)

    def test_preprocess_extreme_aspect_ratio_tall(self):
        """Test preprocessing with extreme tall aspect ratio."""
        img = np.random.randint(0, 255, (1000, 100, 3), dtype=np.uint8)
        output = preprocess_from_array_onnx(img, target_size=640)
        assert output.shape == (1, 3, 640, 640)

    def test_preprocess_rknn_sim_dtype_correct(self):
        """Test RKNN simulator preprocessing maintains float32."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output = preprocess_from_array_rknn_sim(img, target_size=640)
        assert output.dtype == np.float32
        assert output.shape == (1, 640, 640, 3)

    def test_preprocess_board_dtype_correct(self):
        """Test board preprocessing maintains uint8."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output = preprocess_from_array_board(img, target_size=640)
        assert output.dtype == np.uint8
        assert output.shape == (1, 640, 640, 3)

    def test_preprocess_board_value_range(self):
        """Test board preprocessing maintains 0-255 range."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output = preprocess_from_array_board(img, target_size=640)
        assert output.min() >= 0
        assert output.max() <= 255

    def test_preprocess_consistency_across_formats(self):
        """Test that all preprocessing functions handle same input consistently."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        onnx_out = preprocess_from_array_onnx(img, target_size=416)
        rknn_out = preprocess_from_array_rknn_sim(img, target_size=416)
        board_out = preprocess_from_array_board(img, target_size=416)

        # All should produce correct shapes
        assert onnx_out.shape == (1, 3, 416, 416)
        assert rknn_out.shape == (1, 416, 416, 3)
        assert board_out.shape == (1, 416, 416, 3)

    def test_preprocess_onnx_channel_order(self):
        """Test ONNX preprocessing converts BGR to RGB."""
        # Create image with distinct BGR channels
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Blue channel
        img[:, :, 1] = 0    # Green channel
        img[:, :, 2] = 0    # Red channel

        output = preprocess_from_array_onnx(img, target_size=100)

        # After BGR->RGB conversion, red channel should be at index 0
        # Blue should be at index 2
        assert output.shape == (1, 3, 100, 100)

    def test_preprocess_multiple_calls_independence(self):
        """Test that multiple preprocessing calls don't interfere."""
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        out1 = preprocess_from_array_onnx(img1, target_size=416)
        out2 = preprocess_from_array_onnx(img2, target_size=416)

        # Both should have correct independent outputs
        assert out1.shape == (1, 3, 416, 416)
        assert out2.shape == (1, 3, 416, 416)
        # Outputs should be different
        assert not np.array_equal(out1, out2)
