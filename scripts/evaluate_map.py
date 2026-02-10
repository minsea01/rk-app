#!/usr/bin/env python3
"""Deprecated wrapper for tools/evaluate.py onnx-stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.deprecation import warn_deprecated
from tools.evaluate import main as evaluate_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deprecated: use tools/evaluate.py onnx-stats")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--dataset", required=True, help="Path to test dataset directory")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--output", default="artifacts/map_evaluation.md", help="Output report path")
    parser.add_argument("--json", default="artifacts/map_metrics.json", help="Output metrics JSON path")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated("scripts/evaluate_map.py", "tools/evaluate.py onnx-stats", once=True)
    return evaluate_main(
        [
            "onnx-stats",
            "--onnx",
            args.onnx,
            "--dataset",
            args.dataset,
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--output",
            args.output,
            "--json",
            args.json,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

