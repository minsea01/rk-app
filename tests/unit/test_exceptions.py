#!/usr/bin/env python3
"""Unit tests for custom exceptions."""

import pytest
from apps.exceptions import (
    RKAppException,
    RKNNError,
    PreprocessError,
    InferenceError,
    ValidationError,
    ModelLoadError,
    ConfigurationError,
)


class TestExceptions:
    """Test suite for custom exception classes."""

    def test_base_exception(self):
        """Test base exception class."""
        with pytest.raises(RKAppException):
            raise RKAppException("Test error")

    def test_rknn_error(self):
        """Test RKNN error exception."""
        with pytest.raises(RKNNError):
            raise RKNNError("RKNN runtime failed")

    def test_preprocess_error(self):
        """Test preprocess error exception."""
        with pytest.raises(PreprocessError):
            raise PreprocessError("Image loading failed")

    def test_inference_error(self):
        """Test inference error exception."""
        with pytest.raises(InferenceError):
            raise InferenceError("Inference failed")

    def test_validation_error(self):
        """Test validation error exception."""
        with pytest.raises(ValidationError):
            raise ValidationError("Invalid input")

    def test_model_load_error(self):
        """Test model load error exception."""
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("Model not found")

    def test_configuration_error(self):
        """Test configuration error exception."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Config invalid")

    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        assert issubclass(RKNNError, RKAppException)
        assert issubclass(PreprocessError, RKAppException)
        assert issubclass(InferenceError, RKAppException)
        assert issubclass(ValidationError, RKAppException)

    def test_exception_message(self):
        """Test exception message handling."""
        msg = "Custom error message"
        try:
            raise RKNNError(msg)
        except RKNNError as e:
            assert str(e) == msg
