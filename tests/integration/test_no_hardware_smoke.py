#!/usr/bin/env python3
"""No-hardware smoke test for video -> ONNX -> JSON/vis pipeline."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper

from apps.utils.decode import decode_predictions
from apps.utils.decode_meta import load_decode_meta
from apps.utils.preprocessing import preprocess_from_array_onnx


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_has_onnx(build_dir: Path) -> bool:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return False
    cache_text = cache_path.read_text(encoding="utf-8", errors="ignore")
    if "RKAPP_BUILD_WITH_ONNX:BOOL=ON" in cache_text:
        return True
    if "RKAPP_BUILD_WITH_ONNX:BOOL=OFF" in cache_text:
        return False
    return "ENABLE_ONNX:BOOL=ON" in cache_text


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    existing = env.get(key, "")
    env[key] = value if not existing else f"{value}:{existing}"


def _ensure_asan_detect_leaks_disabled(env: dict[str, str]) -> None:
    existing = env.get("ASAN_OPTIONS", "")
    if "detect_leaks=" in existing:
        return
    env["ASAN_OPTIONS"] = "detect_leaks=0" if not existing else f"{existing}:detect_leaks=0"


def _create_smoke_model(model_path: Path) -> None:
    predictions = np.zeros((1, 10, 5), dtype=np.float32)
    predictions[0, 0] = [0.50, 0.48, 0.40, 0.52, 10.0]
    predictions[0, 1] = [0.28, 0.40, 0.22, 0.30, 9.0]
    predictions[0, 2] = [0.76, 0.58, 0.18, 0.28, 8.0]
    predictions[0, 3:] = [0.10, 0.10, 0.05, 0.05, -10.0]

    input_info = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output_info = helper.make_tensor_value_info(
        "output0",
        TensorProto.FLOAT,
        list(predictions.shape),
    )
    const_tensor = helper.make_tensor(
        "predictions_value",
        TensorProto.FLOAT,
        predictions.shape,
        predictions.reshape(-1).tolist(),
    )
    const_node = helper.make_node("Constant", inputs=[], outputs=["output0"], value=const_tensor)
    graph = helper.make_graph([const_node], "rkapp_no_hardware_smoke", [input_info], [output_info])
    model = helper.make_model(
        graph,
        producer_name="rkapp_no_hardware_smoke",
        opset_imports=[helper.make_operatorsetid("", 12)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save_model(model, model_path)

    Path(f"{model_path}.json").write_text(
        json.dumps(
            {
                "head": "raw",
                "num_classes": 1,
                "has_objectness": 0,
                "output_index": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _create_smoke_video(source_image: Path, video_path: Path) -> int:
    image = cv2.imread(str(source_image))
    assert image is not None, f"failed to load source image: {source_image}"
    frame = cv2.resize(image, (640, 640))

    frame_count = 8
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        4.0,
        (640, 640),
    )
    assert writer.isOpened(), f"failed to open video writer: {video_path}"

    try:
        for index in range(frame_count):
            annotated = frame.copy()
            cv2.putText(
                annotated,
                f"smoke-{index}",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                annotated,
                (80 + index * 12, 96 + index * 6),
                16,
                (255, 128, 0),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
    finally:
        writer.release()

    return frame_count


def _draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    out = image.copy()
    for (x1, y1, x2, y2), score, class_id in zip(boxes.astype(int), confs, class_ids):
        label = f"person:{float(score):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"id={int(class_id)}",
            (x1, min(out.shape[0] - 8, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def _run_smoke_pipeline(video_path: Path, model_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    decode_meta = load_decode_meta(model_path)
    assert decode_meta["head"] == "raw"
    assert decode_meta["num_classes"] == 1
    assert decode_meta["has_objectness"] == 0

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    capture = cv2.VideoCapture(str(video_path))
    assert capture.isOpened(), f"failed to open video: {video_path}"

    frames = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            tensor = preprocess_from_array_onnx(frame, target_size=640)
            prediction = session.run(None, {input_name: tensor})[0]
            boxes, confs, class_ids = decode_predictions(
                prediction,
                imgsz=640,
                conf_thres=0.25,
                iou_thres=0.45,
                head="auto",
                decode_meta=decode_meta,
            )

            vis_path = vis_dir / f"frame_{index:03d}.jpg"
            cv2.imwrite(str(vis_path), _draw_boxes(frame, boxes, confs, class_ids))

            frames.append(
                {
                    "frame_index": index,
                    "num_detections": int(len(boxes)),
                    "vis_path": str(vis_path),
                    "detections": [
                        {
                            "bbox": [float(value) for value in box],
                            "score": float(score),
                            "class_id": int(class_id),
                            "class_name": "person",
                        }
                        for box, score, class_id in zip(boxes, confs, class_ids)
                    ],
                }
            )
            index += 1
    finally:
        capture.release()

    summary = {
        "source": str(video_path),
        "model": str(model_path),
        "frame_count": len(frames),
        "frames": frames,
    }
    (output_dir / "detections.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _write_detect_cli_config(config_path: Path, model_path: Path, video_path: Path) -> None:
    config_path.write_text(
        (
            "source:\n"
            "  type: video\n"
            f"  uri: \"{video_path}\"\n"
            "engine:\n"
            "  type: onnx\n"
            f"  model: \"{model_path}\"\n"
            "  input_size: [640, 640]\n"
            "postprocess:\n"
            "  conf_threshold: 0.25\n"
            "  nms_threshold: 0.45\n"
            "  max_detections: 20\n"
            "classes:\n"
            "  names: [person]\n"
            "output:\n"
            "  type: tcp\n"
            "  tcp:\n"
            "    host: \"127.0.0.1\"\n"
            "    port: 9000\n"
            "    queue_size: 8\n"
            "    include_image: true\n"
            "    image_quality: 70\n"
            "    image_interval: 1\n"
            "    draw_detections: true\n"
            "logging:\n"
            "  level: info\n"
            "runtime:\n"
            "  warmup: 1\n"
            "  async: false\n"
        ),
        encoding="utf-8",
    )


@pytest.mark.integration
def test_no_hardware_video_to_onnx_smoke(tmp_path: Path) -> None:
    source_image = _repo_root() / "assets" / "bus.jpg"
    assert source_image.exists(), f"missing tracked fixture: {source_image}"

    model_path = tmp_path / "smoke.onnx"
    video_path = tmp_path / "smoke.avi"
    output_dir = tmp_path / "out"

    _create_smoke_model(model_path)
    expected_frames = _create_smoke_video(source_image, video_path)
    summary = _run_smoke_pipeline(video_path, model_path, output_dir)

    json_path = output_dir / "detections.json"
    vis_files = sorted((output_dir / "vis").glob("frame_*.jpg"))

    assert json_path.exists()
    assert summary["frame_count"] == expected_frames
    assert len(vis_files) == expected_frames
    assert summary["frames"], "smoke pipeline produced no frame results"
    assert all(frame["num_detections"] >= 1 for frame in summary["frames"])

    first_frame = summary["frames"][0]
    assert set(first_frame) == {"frame_index", "num_detections", "vis_path", "detections"}
    assert first_frame["detections"], "expected deterministic smoke model to produce detections"
    assert set(first_frame["detections"][0]) == {"bbox", "score", "class_id", "class_name"}


@pytest.mark.integration
def test_detect_cli_no_hardware_smoke(tmp_path: Path) -> None:
    repo_root = _repo_root()
    build_dir = repo_root / "build" / "x86-debug"
    detect_cli = build_dir / "detect_cli"
    if not detect_cli.exists():
        pytest.skip("detect_cli not built")
    if not _build_has_onnx(build_dir):
        pytest.skip("detect_cli build does not enable ONNX")

    source_image = repo_root / "assets" / "bus.jpg"
    model_path = tmp_path / "smoke_cli.onnx"
    video_path = tmp_path / "smoke_cli.avi"
    config_path = tmp_path / "detect_cli.yaml"
    output_dir = tmp_path / "detect_cli_output"
    vis_dir = output_dir / "vis"
    json_path = output_dir / "detections.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    _create_smoke_model(model_path)
    expected_frames = _create_smoke_video(source_image, video_path)
    _write_detect_cli_config(config_path, model_path, video_path)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root))
    ort_home = env.get("ORT_HOME")
    if ort_home:
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(Path(ort_home) / "lib"))
    _ensure_asan_detect_leaks_disabled(env)
    completed = subprocess.run(
        [
            str(detect_cli),
            "--cfg",
            str(config_path),
            "--json",
            str(json_path),
            "--save_vis",
            str(vis_dir),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert json_path.exists()
    saved_frames = sorted(vis_dir.glob("frame_*.jpg"))
    assert len(saved_frames) == expected_frames

    payload = json.loads(json_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["source_uri"] == str(video_path)
    assert payload["detections"]
    assert payload["image"]["encoding"] == "jpeg"
    assert payload["image"]["contains_overlays"] is True
    assert payload["image"]["data_base64"]
