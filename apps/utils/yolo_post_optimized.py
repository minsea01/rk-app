#!/usr/bin/env python3
"""Compatibility layer for optimized postprocess entrypoints."""

from __future__ import annotations

from apps.deprecation import warn_deprecated
from apps.utils.yolo_post import letterbox, nms, postprocess_yolov8, sigmoid

try:
    import numba as _numba  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def postprocess_yolov8_optimized(*args, **kwargs):
    """Compatibility wrapper around canonical postprocess implementation."""
    warn_deprecated(
        "apps.utils.yolo_post_optimized.postprocess_yolov8_optimized",
        "apps.utils.yolo_post.postprocess_yolov8",
        once=True,
    )
    return postprocess_yolov8(*args, **kwargs)


__all__ = [
    "NUMBA_AVAILABLE",
    "letterbox",
    "nms",
    "postprocess_yolov8_optimized",
    "sigmoid",
]

