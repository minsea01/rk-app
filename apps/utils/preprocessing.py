#!/usr/bin/env python3
"""Compatibility preprocessing API built on preprocess_pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from apps.config import ModelConfig
from apps.exceptions import PreprocessError
from apps.utils.preprocess_pipeline import PreprocessConfig, PreprocessState, run_preprocess

# Shared default pipeline settings to keep legacy API behavior deterministic.
_DEFAULT_CONFIG = PreprocessConfig(
    profile="speed",
    undistort_enable=False,
    roi_enable=False,
    gamma_enable=False,
    white_balance_enable=False,
    denoise_enable=False,
    input_format="bgr",
)
_DEFAULT_STATE = PreprocessState()


def _load_image(img_path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(img_path))
    if image is None:
        raise PreprocessError(
            f"Failed to load image: {img_path}\n"
            f"  Possible causes: file doesn't exist, unsupported format, or corrupted file"
        )
    return image


def _prepare_common(img: np.ndarray, target_size: Optional[int]) -> np.ndarray:
    if target_size is None:
        target_size = ModelConfig.DEFAULT_SIZE
    try:
        processed, _meta, _coord = run_preprocess(
            img,
            int(target_size),
            _DEFAULT_CONFIG,
            _DEFAULT_STATE,
            logger=None,
        )
        return processed
    except (ValueError, TypeError, cv2.error) as exc:
        raise PreprocessError(f"Preprocess pipeline failed: {exc}") from exc


def preprocess_onnx(img_path: Union[str, Path], target_size: Optional[int] = None) -> np.ndarray:
    image = _load_image(img_path)
    return preprocess_from_array_onnx(image, target_size=target_size)


def preprocess_rknn_sim(
    img_path: Union[str, Path], target_size: Optional[int] = None
) -> np.ndarray:
    image = _load_image(img_path)
    return preprocess_from_array_rknn_sim(image, target_size=target_size)


def preprocess_board(img_path: Union[str, Path], target_size: Optional[int] = None) -> np.ndarray:
    image = _load_image(img_path)
    return preprocess_from_array_board(image, target_size=target_size)


def preprocess_from_array_onnx(img: np.ndarray, target_size: Optional[int] = None) -> np.ndarray:
    prepared = _prepare_common(img, target_size)
    prepared = prepared[..., ::-1]  # BGR -> RGB
    prepared = prepared.transpose(2, 0, 1)  # HWC -> CHW
    prepared = np.expand_dims(prepared, axis=0)  # NCHW
    return prepared.astype(np.float32)


def preprocess_from_array_rknn_sim(
    img: np.ndarray, target_size: Optional[int] = None
) -> np.ndarray:
    prepared = _prepare_common(img, target_size)
    prepared = prepared[..., ::-1]  # BGR -> RGB
    prepared = np.expand_dims(prepared, axis=0)  # NHWC
    return np.ascontiguousarray(prepared).astype(np.float32)


def preprocess_from_array_board(img: np.ndarray, target_size: Optional[int] = None) -> np.ndarray:
    prepared = _prepare_common(img, target_size)
    prepared = np.expand_dims(prepared, axis=0)  # NHWC
    return np.ascontiguousarray(prepared).astype(np.uint8)
