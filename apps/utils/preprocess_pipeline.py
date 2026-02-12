#!/usr/bin/env python3
"""Shared preprocessing pipeline for C++/Python behavior parity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
import logging

import cv2
import numpy as np
import yaml

from apps.utils.yolo_post import letterbox


def _lower(value: str) -> str:
    return value.strip().lower()


def parse_xywh(text: str, as_int: bool = False) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 comma-separated values, got: {text}")
    if as_int:
        parsed = tuple(float(int(round(float(p)))) for p in parts)
    else:
        parsed = tuple(float(p) for p in parts)
    return parsed  # type: ignore[return-value]


@dataclass
class PreprocessConfig:
    profile: str = "speed"
    undistort_enable: bool = False
    calibration_file: str = ""

    roi_enable: bool = False
    roi_mode: str = "normalized"  # normalized|pixel
    roi_normalized_xywh: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    roi_pixel_xywh: Tuple[int, int, int, int] = (0, 0, 0, 0)
    roi_clamp: bool = True
    roi_min_size: int = 8

    gamma_enable: Optional[bool] = None
    gamma_value: float = 1.0
    white_balance_enable: Optional[bool] = None
    white_balance_clip_percent: float = 0.0
    denoise_enable: Optional[bool] = None
    denoise_method: str = "bilateral"
    denoise_d: int = 5
    denoise_sigma_color: float = 35.0
    denoise_sigma_space: float = 35.0

    input_format: str = "auto"  # auto|bgr|rgb|gray|bayer_rg|bayer_bg|bayer_gr|bayer_gb


@dataclass
class PreprocessState:
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    undistort_map1: Optional[np.ndarray] = None
    undistort_map2: Optional[np.ndarray] = None
    undistort_size: Tuple[int, int] = (0, 0)
    calibration_loaded: bool = False
    calibration_attempted: bool = False
    gamma_lut: Optional[np.ndarray] = None
    gamma_lut_value: Optional[float] = None
    warned_invalid_roi: bool = False


@dataclass
class FrameMeta:
    ratio: float
    pad: Tuple[float, float]
    roi_rect: Tuple[int, int, int, int]
    roi_applied: bool
    coord_shape: Tuple[int, int]


def _profile_defaults(profile: str) -> Tuple[bool, bool, bool]:
    lowered = _lower(profile)
    if lowered == "balanced":
        return True, True, False
    if lowered == "quality":
        return True, True, True
    return False, False, False


def _effective_flags(config: PreprocessConfig) -> Tuple[bool, bool, bool]:
    wb_default, gamma_default, denoise_default = _profile_defaults(config.profile)
    wb = wb_default if config.white_balance_enable is None else config.white_balance_enable
    gamma = gamma_default if config.gamma_enable is None else config.gamma_enable
    denoise = denoise_default if config.denoise_enable is None else config.denoise_enable
    return bool(wb), bool(gamma), bool(denoise)


def _get_nested(node: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = node
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _load_yaml_config(cfg_path: Optional[Path]) -> Dict[str, Any]:
    if cfg_path is None:
        return {}
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level YAML must be a mapping")
    return loaded


def build_preprocess_config(
    cfg_path: Optional[Path],
    cli_overrides: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> PreprocessConfig:
    config = PreprocessConfig()
    raw = _load_yaml_config(cfg_path) if cfg_path is not None else {}
    preprocess = raw.get("preprocess", {}) if isinstance(raw, dict) else {}
    if isinstance(preprocess, Mapping):
        profile = preprocess.get("profile")
        if isinstance(profile, str) and profile.strip():
            config.profile = profile.strip()

        undistort = preprocess.get("undistort", {})
        if isinstance(undistort, Mapping):
            if "enable" in undistort:
                config.undistort_enable = bool(undistort.get("enable"))
            calib = undistort.get("calibration_file")
            if isinstance(calib, str):
                config.calibration_file = calib

        roi = preprocess.get("roi", {})
        if isinstance(roi, Mapping):
            if "enable" in roi:
                config.roi_enable = bool(roi.get("enable"))
            if isinstance(roi.get("mode"), str):
                config.roi_mode = str(roi.get("mode"))
            norm = roi.get("normalized_xywh")
            if isinstance(norm, (list, tuple)) and len(norm) == 4:
                config.roi_normalized_xywh = tuple(float(v) for v in norm)  # type: ignore[assignment]
            px = roi.get("pixel_xywh")
            if isinstance(px, (list, tuple)) and len(px) == 4:
                config.roi_pixel_xywh = tuple(int(v) for v in px)  # type: ignore[assignment]
            if "clamp" in roi:
                config.roi_clamp = bool(roi.get("clamp"))
            if "min_size" in roi:
                config.roi_min_size = int(roi.get("min_size"))

        gamma = preprocess.get("gamma", {})
        if isinstance(gamma, Mapping):
            if "enable" in gamma:
                config.gamma_enable = bool(gamma.get("enable"))
            if "value" in gamma:
                config.gamma_value = float(gamma.get("value"))

        white_balance = preprocess.get("white_balance", {})
        if isinstance(white_balance, Mapping):
            if "enable" in white_balance:
                config.white_balance_enable = bool(white_balance.get("enable"))
            if "clip_percent" in white_balance:
                config.white_balance_clip_percent = float(white_balance.get("clip_percent"))

        denoise = preprocess.get("denoise", {})
        if isinstance(denoise, Mapping):
            if "enable" in denoise:
                config.denoise_enable = bool(denoise.get("enable"))
            if "method" in denoise and isinstance(denoise.get("method"), str):
                config.denoise_method = str(denoise.get("method"))
            if "d" in denoise:
                config.denoise_d = int(denoise.get("d"))
            if "sigma_color" in denoise:
                config.denoise_sigma_color = float(denoise.get("sigma_color"))
            if "sigma_space" in denoise:
                config.denoise_sigma_space = float(denoise.get("sigma_space"))

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None:
                continue
            if key == "roi_normalized_xywh" and isinstance(value, str):
                config.roi_normalized_xywh = parse_xywh(value)
                continue
            if key == "roi_pixel_xywh" and isinstance(value, str):
                parsed = parse_xywh(value, as_int=True)
                config.roi_pixel_xywh = tuple(int(v) for v in parsed)  # type: ignore[assignment]
                continue
            if hasattr(config, key):
                setattr(config, key, value)

    config.profile = _lower(config.profile)
    if config.profile not in {"speed", "balanced", "quality"}:
        if logger is not None:
            logger.warning("Unknown preprocess profile '%s', fallback to speed", config.profile)
        config.profile = "speed"
    config.roi_mode = _lower(config.roi_mode)
    if config.roi_mode not in {"normalized", "pixel"}:
        config.roi_mode = "normalized"
    config.input_format = _lower(config.input_format)
    config.denoise_method = _lower(config.denoise_method)
    config.roi_min_size = max(1, int(config.roi_min_size))
    config.gamma_value = max(1e-6, float(config.gamma_value))
    config.white_balance_clip_percent = float(np.clip(config.white_balance_clip_percent, 0.0, 49.0))
    return config


def _load_calibration_once(
    state: PreprocessState, config: PreprocessConfig, logger: Optional[logging.Logger]
) -> None:
    if state.calibration_attempted:
        return
    state.calibration_attempted = True
    if not config.undistort_enable or not config.calibration_file:
        return

    fs = cv2.FileStorage(config.calibration_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        if logger is not None:
            logger.warning("Failed to open calibration file: %s", config.calibration_file)
        return

    def read_first(keys: Tuple[str, ...]) -> Optional[np.ndarray]:
        for key in keys:
            node = fs.getNode(key)
            if not node.empty():
                mat = node.mat()
                if mat is not None and mat.size > 0:
                    return mat
        return None

    camera = read_first(("camera_matrix", "cameraMatrix", "K", "intrinsic_matrix"))
    dist = read_first(("dist_coeffs", "distCoeffs", "distortion_coefficients", "D"))
    fs.release()

    if camera is None or dist is None:
        if logger is not None:
            logger.warning("Calibration missing camera_matrix/dist_coeffs")
        return

    camera = camera.astype(np.float64)
    dist = dist.astype(np.float64)
    if camera.shape == (1, 9):
        camera = camera.reshape(3, 3)
    if camera.shape != (3, 3):
        if logger is not None:
            logger.warning("Calibration camera_matrix must be 3x3")
        return
    dist = dist.reshape(1, -1)
    if dist.shape[1] < 4:
        if logger is not None:
            logger.warning("Calibration dist_coeffs length must be >= 4")
        return

    state.camera_matrix = camera
    state.dist_coeffs = dist
    state.calibration_loaded = True


def _ensure_bgr8(frame: np.ndarray, input_format: str) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    if input_format in {"auto", "bgr"}:
        if frame.shape[2] == 3:
            return frame
        if frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if input_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if input_format == "gray":
        gray = frame if frame.shape[2] == 1 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if input_format == "bayer_rg":
        return cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
    if input_format == "bayer_bg":
        return cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
    if input_format == "bayer_gr":
        return cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
    if input_format == "bayer_gb":
        return cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
    raise ValueError(f"Unsupported input format: {input_format}")


def _resolve_roi(
    config: PreprocessConfig, shape: Tuple[int, int]
) -> Tuple[int, int, int, int, bool]:
    h, w = shape
    full = (0, 0, w, h, False)
    if not config.roi_enable:
        return full

    if config.roi_mode == "pixel":
        x, y, rw, rh = config.roi_pixel_xywh
    else:
        nx, ny, nw, nh = config.roi_normalized_xywh
        x = int(round(nx * w))
        y = int(round(ny * h))
        rw = int(round(nw * w))
        rh = int(round(nh * h))

    min_size = max(1, config.roi_min_size)
    if config.roi_clamp:
        x = int(np.clip(x, 0, max(0, w - 1)))
        y = int(np.clip(y, 0, max(0, h - 1)))
        rw = max(rw, min_size)
        rh = max(rh, min_size)
        rw = min(rw, w - x)
        rh = min(rh, h - y)

    if rw < min_size or rh < min_size:
        return full
    if x < 0 or y < 0 or x + rw > w or y + rh > h:
        return full

    applied = not (x == 0 and y == 0 and rw == w and rh == h)
    return (x, y, rw, rh, applied)


def _white_balance_gray_world(image: np.ndarray, clip_percent: float) -> np.ndarray:
    working = image.astype(np.float32)
    channels = cv2.split(working)

    clipped = float(np.clip(clip_percent, 0.0, 49.0))
    if clipped > 0.0:
        each_side = clipped * 0.5
        for index, channel in enumerate(channels):
            lo = float(np.percentile(channel, each_side))
            hi = float(np.percentile(channel, 100.0 - each_side))
            channels[index] = np.clip(channel, lo, hi)

    means = np.array([float(np.mean(channel)) for channel in channels], dtype=np.float32)
    gray = float(np.mean(means))
    gains = gray / (means + 1e-6)
    gains = np.clip(gains, 0.2, 5.0)
    balanced = [channels[i] * gains[i] for i in range(3)]
    merged = cv2.merge(balanced)
    return np.clip(merged, 0, 255).astype(np.uint8)


def _gamma_lut(state: PreprocessState, gamma: float) -> np.ndarray:
    if state.gamma_lut is not None and state.gamma_lut_value is not None:
        if abs(state.gamma_lut_value - gamma) < 1e-9:
            return state.gamma_lut
    values = np.arange(256, dtype=np.float32) / 255.0
    table = np.power(values, gamma) * 255.0
    lut = np.clip(np.round(table), 0, 255).astype(np.uint8)
    state.gamma_lut = lut
    state.gamma_lut_value = gamma
    return lut


def run_preprocess(
    frame: np.ndarray,
    imgsz: int,
    config: PreprocessConfig,
    state: PreprocessState,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, FrameMeta, np.ndarray]:
    bgr = _ensure_bgr8(frame, config.input_format)

    _load_calibration_once(state, config, logger)
    coord_frame = bgr
    if config.undistort_enable and state.calibration_loaded:
        h, w = bgr.shape[:2]
        if (
            state.undistort_size != (w, h)
            or state.undistort_map1 is None
            or state.undistort_map2 is None
        ):
            new_camera, _ = cv2.getOptimalNewCameraMatrix(
                state.camera_matrix, state.dist_coeffs, (w, h), 0.0, (w, h)
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                state.camera_matrix,
                state.dist_coeffs,
                None,
                new_camera,
                (w, h),
                cv2.CV_16SC2,
            )
            state.undistort_map1 = map1
            state.undistort_map2 = map2
            state.undistort_size = (w, h)
        coord_frame = cv2.remap(
            bgr,
            state.undistort_map1,
            state.undistort_map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    roi_x, roi_y, roi_w, roi_h, roi_applied = _resolve_roi(config, coord_frame.shape[:2])
    if config.roi_enable and not roi_applied and not state.warned_invalid_roi:
        if (roi_x, roi_y, roi_w, roi_h) == (0, 0, coord_frame.shape[1], coord_frame.shape[0]):
            if logger is not None and (
                config.roi_mode == "pixel" or config.roi_normalized_xywh != (0.0, 0.0, 1.0, 1.0)
            ):
                logger.warning("ROI config invalid or degenerate, fallback to full frame")
                state.warned_invalid_roi = True

    stage = (
        coord_frame
        if not roi_applied
        else coord_frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
    )

    wb_enabled, gamma_enabled, denoise_enabled = _effective_flags(config)
    if denoise_enabled:
        stage = cv2.bilateralFilter(
            stage,
            int(config.denoise_d),
            float(max(config.denoise_sigma_color, 0.1)),
            float(max(config.denoise_sigma_space, 0.1)),
        )
    if wb_enabled:
        stage = _white_balance_gray_world(stage, config.white_balance_clip_percent)
    if gamma_enabled:
        lut = _gamma_lut(state, config.gamma_value)
        stage = cv2.LUT(stage, lut)

    img, ratio, pad = letterbox(stage, imgsz)
    meta = FrameMeta(
        ratio=float(ratio),
        pad=(float(pad[0]), float(pad[1])),
        roi_rect=(int(roi_x), int(roi_y), int(roi_w), int(roi_h)),
        roi_applied=bool(roi_applied),
        coord_shape=(int(coord_frame.shape[0]), int(coord_frame.shape[1])),
    )
    return img, meta, coord_frame


def map_boxes_back(boxes: np.ndarray, meta: FrameMeta) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    mapped = boxes.astype(np.float32).copy()

    mapped[:, [0, 2]] -= meta.pad[0]
    mapped[:, [1, 3]] -= meta.pad[1]
    mapped /= max(meta.ratio, 1e-6)

    roi_x, roi_y, roi_w, roi_h = meta.roi_rect
    mapped[:, 0::2] = mapped[:, 0::2].clip(0, max(0, roi_w - 1))
    mapped[:, 1::2] = mapped[:, 1::2].clip(0, max(0, roi_h - 1))

    if meta.roi_applied:
        mapped[:, [0, 2]] += float(roi_x)
        mapped[:, [1, 3]] += float(roi_y)

    h, w = meta.coord_shape
    mapped[:, 0::2] = mapped[:, 0::2].clip(0, max(0, w - 1))
    mapped[:, 1::2] = mapped[:, 1::2].clip(0, max(0, h - 1))
    return mapped
