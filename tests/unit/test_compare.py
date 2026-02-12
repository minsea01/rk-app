#!/usr/bin/env python3
"""Unit tests for unified comparison utilities."""

import json
from pathlib import Path

import numpy as np

from tools.compare import build_report, compute_tensor_metrics, parse_metric_set
from scripts.compare_onnx_rknn import build_default_args


def test_compute_tensor_metrics_basic():
    onnx_out = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rknn_out = np.array([1.5, 1.5, 2.0], dtype=np.float32)
    metrics = compute_tensor_metrics(onnx_out, rknn_out)

    assert "max_abs" in metrics
    assert "mean_abs" in metrics
    assert "median_abs" in metrics
    assert "mse" in metrics
    assert "mae" in metrics
    assert metrics["max_abs"] == 1.0


def test_build_report_metrics_switch():
    onnx_out = np.ones((1, 6, 5), dtype=np.float32)
    rknn_out = np.ones((1, 6, 5), dtype=np.float32)

    report_tensor = build_report(
        onnx_out,
        rknn_out,
        metrics={"tensor"},
        num_classes=1,
        conf=0.5,
        iou=0.45,
    )
    assert "tensor" in report_tensor["results"]
    assert "post" not in report_tensor["results"]

    report_both = build_report(
        onnx_out,
        rknn_out,
        metrics={"tensor", "post"},
        num_classes=1,
        conf=0.5,
        iou=0.45,
    )
    assert "tensor" in report_both["results"]
    assert "post" in report_both["results"]


def test_report_json_schema_roundtrip(tmp_path: Path):
    onnx_out = np.ones((1, 6, 5), dtype=np.float32)
    rknn_out = np.ones((1, 6, 5), dtype=np.float32)

    report = build_report(
        onnx_out,
        rknn_out,
        metrics=parse_metric_set("tensor"),
        num_classes=1,
        conf=0.5,
        iou=0.45,
    )
    out = tmp_path / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "1.0"
    assert "meta" in loaded
    assert "results" in loaded
    assert "onnx_shape" in loaded["meta"]
    assert "tensor" in loaded["results"]


def test_compare_onnx_rknn_default_path_adaptation():
    class DummyPathConfig:
        YOLO11N_ONNX_416 = "artifacts/models/dummy.onnx"
        COCO_CALIB_DIR = "datasets/dummy_calib"

    def fake_resolve(path_value: str) -> Path:
        return Path("/repo") / path_value

    defaults = build_default_args(path_config_cls=DummyPathConfig, resolver=fake_resolve)
    assert defaults["onnx"] == Path("/repo/artifacts/models/dummy.onnx")
    assert defaults["calib_dir"] == Path("/repo/datasets/dummy_calib")
