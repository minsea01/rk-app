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
