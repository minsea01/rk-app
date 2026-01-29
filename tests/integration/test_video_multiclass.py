#!/usr/bin/env python3
"""RK3588(rknnlite) 视频多类别检测集成测试.

说明:
- 仅在 RK3588 板端 (安装 rknnlite) 上可运行。
- CI/PC 环境会自动 skip，避免 pytest 收集阶段崩溃。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_hardware,
    pytest.mark.requires_model,
    pytest.mark.slow,
]

rknnlite = pytest.importorskip("rknnlite.api", reason="requires RK3588 rknnlite runtime")
RKNNLite = rknnlite.RKNNLite  # type: ignore[attr-defined]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from apps.utils.yolo_post import letterbox, postprocess_yolov8  # noqa: E402

MODEL_PATH = REPO_ROOT / "artifacts/models/yolo11n_416.rknn"
VIDEO_PATH = REPO_ROOT / "assets/street.mp4"


def _require_inputs() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"RKNN model not found: {MODEL_PATH}")
    if not VIDEO_PATH.exists():
        pytest.skip(f"Video not found: {VIDEO_PATH}")


def _load_runtime(rknn: "RKNNLite") -> None:
    ret = rknn.load_rknn(str(MODEL_PATH))
    if ret != 0:
        pytest.fail(f"Failed to load RKNN model (ret={ret}): {MODEL_PATH}")

    ret = rknn.init_runtime(core_mask=0x7)
    if ret != 0:
        pytest.fail(f"Failed to init RKNN runtime (ret={ret})")


def _infer_frame(rknn: "RKNNLite", frame: np.ndarray, conf_thres: float = 0.3) -> None:
    img_input, ratio, (dw, dh) = letterbox(frame, 416)
    img_input = img_input[np.newaxis, ...]

    outputs = rknn.inference(inputs=[img_input])
    assert outputs and outputs[0] is not None, "RKNN inference returned empty outputs"

    pred = outputs[0]
    if pred.ndim == 2:
        pred = pred[None, ...]
    if pred.shape[1] < pred.shape[2]:
        pred = pred.transpose(0, 2, 1)

    boxes, confs, cls_ids = postprocess_yolov8(
        pred, 416, (416, 416), (ratio, (dw, dh)), conf_thres, 0.45
    )
    assert boxes.shape[0] == confs.shape[0] == cls_ids.shape[0]


def test_multiclass_video_smoke() -> None:
    _require_inputs()

    rknn = RKNNLite()
    cap = None
    try:
        _load_runtime(rknn)

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        assert cap.isOpened(), f"Failed to open video: {VIDEO_PATH}"

        frames = 0
        while frames < 5:
            ok, frame = cap.read()
            if not ok:
                break
            _infer_frame(rknn, frame)
            frames += 1

        assert frames > 0, "No frames decoded from video"
    finally:
        if cap is not None:
            cap.release()
        rknn.release()

