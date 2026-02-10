#!/usr/bin/env python3
"""Deprecated wrapper for unified model comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.deprecation import warn_deprecated
from tools.compare import main as compare_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deprecated: use tools/compare.py for ONNX vs RKNN validation"
    )
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model")
    parser.add_argument("--rknn", type=Path, required=True, help="Path to RKNN model")
    parser.add_argument("--image", type=Path, required=True, help="Path to test image")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output JSON path (defaults to artifacts/onnx_rknn_validation.json)",
    )
    return parser


def _build_compare_argv(args: argparse.Namespace) -> List[str]:
    json_out = args.json_out or Path("artifacts/onnx_rknn_validation.json")
    return [
        "--onnx",
        str(args.onnx),
        "--rknn",
        str(args.rknn),
        "--img",
        str(args.image),
        "--imgsz",
        str(args.imgsz),
        "--metrics",
        "tensor",
        "--json-out",
        str(json_out),
    ]


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated("scripts/validate_models.py", "tools/compare.py", once=True)
    return compare_main(_build_compare_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
