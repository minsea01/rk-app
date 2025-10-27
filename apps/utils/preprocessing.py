#!/usr/bin/env python3
"""Shared preprocessing utilities for YOLO models.

This module provides consistent preprocessing functions for different
inference backends (ONNX, RKNN simulator, on-device RKNN).
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union


def preprocess_onnx(img_path: Union[str, Path], target_size: int = 416) -> np.ndarray:
    """Preprocess image for ONNX Runtime inference.

    Args:
        img_path: Path to input image
        target_size: Target image size (square)

    Returns:
        Preprocessed image in NCHW format, float32, 0-255 range
        Shape: (1, 3, target_size, target_size)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    inp = cv2.resize(img, (target_size, target_size))
    inp = inp[..., ::-1]  # BGR -> RGB
    inp = inp.transpose(2, 0, 1)  # HWC -> CHW
    inp = np.expand_dims(inp, axis=0)  # Add batch dimension
    return inp.astype(np.float32)


def preprocess_rknn_sim(img_path: Union[str, Path], target_size: int = 416) -> np.ndarray:
    """Preprocess image for RKNN PC simulator inference.

    Args:
        img_path: Path to input image
        target_size: Target image size (square)

    Returns:
        Preprocessed image in NHWC format, float32, 0-255 range
        Shape: (1, target_size, target_size, 3)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    inp = cv2.resize(img, (target_size, target_size))
    inp = inp[..., ::-1]  # BGR -> RGB
    inp = np.expand_dims(inp, axis=0)  # Add batch dimension: (H,W,C) -> (1,H,W,C)
    return np.ascontiguousarray(inp).astype(np.float32)


def preprocess_board(img_path: Union[str, Path], target_size: int = 416) -> np.ndarray:
    """Preprocess image for on-device RKNN inference.

    Args:
        img_path: Path to input image
        target_size: Target image size (square)

    Returns:
        Preprocessed image in NHWC format, uint8, 0-255 range
        Shape: (1, target_size, target_size, 3)

    Note:
        The RKNN model should have mean/std preprocessing configured
        during conversion to handle BGR->RGB and normalization.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    inp = cv2.resize(img, (target_size, target_size))
    # Note: Keep as BGR uint8 if model has reorder configured
    # Or apply BGR->RGB here: inp = inp[..., ::-1]
    inp = np.expand_dims(inp, axis=0)
    return np.ascontiguousarray(inp).astype(np.uint8)


def preprocess_from_array_onnx(img: np.ndarray, target_size: int = 416) -> np.ndarray:
    """Preprocess numpy array for ONNX Runtime inference.

    Args:
        img: Input image as numpy array (HWC, BGR, uint8)
        target_size: Target image size (square)

    Returns:
        Preprocessed image in NCHW format, float32
        Shape: (1, 3, target_size, target_size)
    """
    inp = cv2.resize(img, (target_size, target_size))
    inp = inp[..., ::-1]  # BGR -> RGB
    inp = inp.transpose(2, 0, 1)  # HWC -> CHW
    inp = np.expand_dims(inp, axis=0)
    return inp.astype(np.float32)


def preprocess_from_array_rknn_sim(img: np.ndarray, target_size: int = 416) -> np.ndarray:
    """Preprocess numpy array for RKNN PC simulator inference.

    Args:
        img: Input image as numpy array (HWC, BGR, uint8)
        target_size: Target image size (square)

    Returns:
        Preprocessed image in NHWC format, float32
        Shape: (1, target_size, target_size, 3)
    """
    inp = cv2.resize(img, (target_size, target_size))
    inp = inp[..., ::-1]  # BGR -> RGB
    inp = np.expand_dims(inp, axis=0)
    return np.ascontiguousarray(inp).astype(np.float32)


def preprocess_from_array_board(img: np.ndarray, target_size: int = 416) -> np.ndarray:
    """Preprocess numpy array for on-device RKNN inference.

    Args:
        img: Input image as numpy array (HWC, BGR, uint8)
        target_size: Target image size (square)

    Returns:
        Preprocessed image in NHWC format, uint8
        Shape: (1, target_size, target_size, 3)
    """
    inp = cv2.resize(img, (target_size, target_size))
    inp = np.expand_dims(inp, axis=0)
    return np.ascontiguousarray(inp).astype(np.uint8)
