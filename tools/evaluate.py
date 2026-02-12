#!/usr/bin/env python3
"""Unified evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

# Allow direct script execution from repository root context.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.logger import setup_logger
from apps.utils.decode import decode_predictions
from apps.utils.preprocessing import preprocess_from_array_onnx

logger = setup_logger(__name__)


def run_onnx_stats(
    onnx_path: Path,
    dataset_path: Path,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
) -> Dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(f"onnxruntime not installed: {exc}") from exc

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    image_paths = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No images found in dataset path: {dataset_path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    total_detections = 0
    images_with_detections = 0

    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        tensor = preprocess_from_array_onnx(image, target_size=imgsz)
        outputs = session.run(None, {input_name: tensor})
        boxes, confs, _ = decode_predictions(
            outputs[0],
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            head="auto",
            orig_shape=None,
        )

        det_count = len(boxes)
        total_detections += det_count
        if det_count > 0:
            images_with_detections += 1

        if idx % 50 == 0:
            logger.info("Processed %d/%d images", idx, len(image_paths))

    total_images = len(image_paths)
    return {
        "total_images": float(total_images),
        "images_with_detections": float(images_with_detections),
        "detection_rate": float(images_with_detections / total_images if total_images else 0.0),
        "total_detections": float(total_detections),
        "avg_detections_per_image": float(total_detections / total_images if total_images else 0.0),
        "conf_threshold": float(conf_thres),
        "iou_threshold": float(iou_thres),
        "imgsz": float(imgsz),
        "note": "Detection statistics only. Full mAP requires GT annotations and matching pipeline.",
    }


def write_markdown_report(metrics: Dict[str, float], onnx_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# ONNX Detection Statistics Report",
                "",
                f"**Model:** {onnx_path.name}",
                f"**Input Size:** {int(metrics['imgsz'])}x{int(metrics['imgsz'])}",
                f"**Confidence Threshold:** {metrics['conf_threshold']}",
                f"**IoU Threshold:** {metrics['iou_threshold']}",
                "",
                "## Summary",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total Images | {int(metrics['total_images'])} |",
                f"| Images with Detections | {int(metrics['images_with_detections'])} |",
                f"| Detection Rate | {metrics['detection_rate']:.2%} |",
                f"| Total Detections | {int(metrics['total_detections'])} |",
                f"| Avg Detections/Image | {metrics['avg_detections_per_image']:.2f} |",
                "",
                "## Notes",
                "",
                f"- {metrics.get('note', '')}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_yolo_full(
    model_path: Path,
    data_yaml: Path,
    conf: float,
    iou: float,
    output_dir: Path,
) -> int:
    from tools.model_evaluation import ModelEvaluator

    output_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        evaluator = ModelEvaluator(str(model_path), str(data_yaml), conf, iou)
        evaluator.run_full_evaluation()
    finally:
        os.chdir(cwd)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified evaluation tool")
    sub = parser.add_subparsers(dest="command", required=True)

    onnx = sub.add_parser(
        "onnx-stats", help="Evaluate ONNX model detection statistics on image folder"
    )
    onnx.add_argument("--onnx", type=Path, required=True)
    onnx.add_argument("--dataset", type=Path, required=True)
    onnx.add_argument("--imgsz", type=int, default=416)
    onnx.add_argument("--conf", type=float, default=0.5)
    onnx.add_argument("--iou", type=float, default=0.45)
    onnx.add_argument("--output", type=Path, default=Path("artifacts/map_evaluation.md"))
    onnx.add_argument("--json", type=Path, default=Path("artifacts/map_metrics.json"))

    yolo = sub.add_parser("yolo-full", help="Run full YOLO evaluation suite")
    yolo.add_argument("--model", type=Path, required=True)
    yolo.add_argument("--data", type=Path, required=True)
    yolo.add_argument("--conf", type=float, default=0.25)
    yolo.add_argument("--iou", type=float, default=0.6)
    yolo.add_argument("--output-dir", type=Path, default=Path("."))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "onnx-stats":
        metrics = run_onnx_stats(args.onnx, args.dataset, args.imgsz, args.conf, args.iou)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_markdown_report(metrics, args.onnx, args.output)
        print(f"Report: {args.output}")
        print(f"Metrics: {args.json}")
        return 0

    if args.command == "yolo-full":
        return run_yolo_full(args.model, args.data, args.conf, args.iou, args.output_dir)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
