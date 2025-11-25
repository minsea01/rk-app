#!/usr/bin/env python3
"""Custom exception classes for rk-app.

This module provides specific exception types for different error scenarios,
enabling better error handling and debugging throughout the application.
"""


class RKAppException(Exception):
    """Base exception class for all rk-app exceptions."""

    pass


class RKNNError(RKAppException):
    """RKNN runtime or inference related errors.

    Raised when:
    - RKNN model loading fails
    - Runtime initialization fails
    - Inference execution fails
    """

    pass


class PreprocessError(RKAppException):
    """Image preprocessing related errors.

    Raised when:
    - Image loading fails
    - Image format is invalid
    - Preprocessing operations fail
    """

    pass


class InferenceError(RKAppException):
    """General inference related errors.

    Raised when:
    - Model inference fails
    - Output format is unexpected
    - Post-processing fails
    """

    pass


class ValidationError(RKAppException):
    """Input validation errors.

    Raised when:
    - Input parameters are invalid
    - Configuration is malformed
    - Data validation fails
    """

    pass


class ModelLoadError(RKAppException):
    """Model loading related errors.

    Raised when:
    - Model file not found
    - Model format is invalid
    - Model loading fails
    """

    pass


class ConfigurationError(RKAppException):
    """Configuration related errors.

    Raised when:
    - Config file not found
    - Config format is invalid
    - Required config parameters missing
    """

    pass
