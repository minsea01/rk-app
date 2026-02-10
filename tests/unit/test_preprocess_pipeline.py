#!/usr/bin/env python3
"""Unit tests for shared preprocess pipeline."""

from pathlib import Path

import cv2
import numpy as np

from apps.utils.preprocess_pipeline import (
    PreprocessConfig,
    PreprocessState,
    build_preprocess_config,
    map_boxes_back,
    run_preprocess,
)


def test_build_preprocess_config_yaml_and_cli_override(tmp_path: Path):
    cfg_file = tmp_path / "pp.yaml"
    cfg_file.write_text(
        """
preprocess:
  profile: quality
  roi:
    enable: true
    mode: pixel
    pixel_xywh: [10, 20, 100, 80]
  gamma:
    enable: true
    value: 0.8
""",
        encoding="utf-8",
    )

    cfg = build_preprocess_config(
        cfg_file,
        {
            "profile": "balanced",
            "gamma_value": 0.6,
            "roi_pixel_xywh": "1,2,30,40",
        },
    )
    assert cfg.profile == "balanced"
    assert cfg.gamma_value == 0.6
    assert cfg.roi_pixel_xywh == (1, 2, 30, 40)
    assert cfg.roi_enable is True
    assert cfg.roi_mode == "pixel"


def test_run_preprocess_roi_and_box_mapping_round_trip():
    image = np.full((120, 200, 3), 128, dtype=np.uint8)
    config = PreprocessConfig(
        profile="speed",
        roi_enable=True,
        roi_mode="pixel",
        roi_pixel_xywh=(40, 30, 100, 60),
    )
    state = PreprocessState()

    preprocessed, meta, _ = run_preprocess(image, 64, config, state)
    assert preprocessed.shape == (64, 64, 3)
    assert meta.roi_applied is True
    assert meta.roi_rect == (40, 30, 100, 60)

    stage_box = np.array([[10.0, 8.0, 50.0, 40.0]], dtype=np.float32)
    letterbox_box = stage_box.copy()
    letterbox_box[:, [0, 2]] = letterbox_box[:, [0, 2]] * meta.ratio + meta.pad[0]
    letterbox_box[:, [1, 3]] = letterbox_box[:, [1, 3]] * meta.ratio + meta.pad[1]

    mapped = map_boxes_back(letterbox_box, meta)
    expected = stage_box.copy()
    expected[:, [0, 2]] += 40
    expected[:, [1, 3]] += 30
    assert np.allclose(mapped, expected, atol=1.0)


def test_run_preprocess_auto_gray_to_bgr():
    image = np.full((48, 64), 127, dtype=np.uint8)
    config = PreprocessConfig(profile="speed", input_format="auto")
    state = PreprocessState()
    preprocessed, meta, coord = run_preprocess(image, 32, config, state)
    assert preprocessed.shape == (32, 32, 3)
    assert coord.ndim == 3 and coord.shape[2] == 3
    assert meta.coord_shape == (48, 64)


def test_gamma_lut_cache_reused_between_frames():
    image = np.full((80, 80, 3), 120, dtype=np.uint8)
    config = PreprocessConfig(profile="speed", gamma_enable=True, gamma_value=0.7)
    state = PreprocessState()

    _ = run_preprocess(image, 64, config, state)
    first_lut = state.gamma_lut
    _ = run_preprocess(image, 64, config, state)
    second_lut = state.gamma_lut

    assert first_lut is not None
    assert second_lut is not None
    assert first_lut is second_lut
    assert state.gamma_lut_value == 0.7


def test_balanced_profile_applies_wb_and_gamma_defaults():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = 220
    image[..., 1] = 100
    image[..., 2] = 50

    speed_cfg = PreprocessConfig(profile="speed")
    balanced_cfg = PreprocessConfig(profile="balanced")
    speed_state = PreprocessState()
    balanced_state = PreprocessState()

    speed_img, _, _ = run_preprocess(image, 64, speed_cfg, speed_state)
    balanced_img, _, _ = run_preprocess(image, 64, balanced_cfg, balanced_state)

    assert speed_img.shape == balanced_img.shape
    assert not np.array_equal(speed_img, balanced_img)


def test_build_preprocess_config_clamps_negative_gamma_value():
    cfg = build_preprocess_config(
        None,
        {
            "gamma_enable": True,
            "gamma_value": -2.0,
        },
    )
    assert cfg.gamma_enable is True
    assert cfg.gamma_value == 1e-6


def test_run_preprocess_executes_undistort_and_quality_chain(tmp_path: Path):
    calib_path = tmp_path / "camera.yaml"
    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", np.array([[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]]))
    fs.write("dist_coeffs", np.array([[0.05, -0.01, 0.0, 0.0, 0.0]]))
    fs.release()

    image = np.full((64, 64, 3), (180, 90, 40), dtype=np.uint8)
    config = PreprocessConfig(
        profile="quality",
        undistort_enable=True,
        calibration_file=str(calib_path),
    )
    state = PreprocessState()

    output, meta, coord = run_preprocess(image, 64, config, state)
    assert output.shape == (64, 64, 3)
    assert meta.coord_shape == (64, 64)
    assert coord.shape == (64, 64, 3)
    assert state.calibration_loaded is True
    assert state.undistort_map1 is not None
    assert state.undistort_map2 is not None
    assert state.gamma_lut is not None
