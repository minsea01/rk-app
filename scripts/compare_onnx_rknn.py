#!/usr/bin/env python3
"""Deprecated compatibility wrapper for legacy compare workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.deprecation import warn_deprecated
from tools.compare import main as compare_main


def build_default_args(
    path_config_cls=None,
    resolver: Optional[Callable[[str], Path]] = None,
) -> Dict[str, Path]:
    """Build legacy defaults from PathConfig (old script behavior)."""
    if path_config_cls is None:
        from apps.config import PathConfig

        path_config_cls = PathConfig
    if resolver is None:
        from apps.utils.paths import resolve_path

        resolver = resolve_path

    return {
        "onnx": resolver(path_config_cls.YOLO11N_ONNX_416),
        "calib_dir": resolver(path_config_cls.COCO_CALIB_DIR),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = build_default_args()
    parser = argparse.ArgumentParser(description="Deprecated: use tools/compare.py")
    parser.add_argument("--onnx", type=Path, default=defaults["onnx"], help="ONNX model path")
    parser.add_argument("--rknn", type=Path, default=None, help="Optional RKNN model path")
    parser.add_argument("--img", type=Path, default=None, help="Optional test image path")
    parser.add_argument("--imgsz", type=int, default=416, help="Input size (legacy default: 416)")
    parser.add_argument("--metrics", type=str, default="tensor", help="Metric set for tools/compare.py")
    parser.add_argument("--quant", action="store_true", help="Enable quantized RKNN simulator build")
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=defaults["calib_dir"],
        help="Calibration directory used with --quant",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/onnx_rknn_comparison.json"),
        help="JSON report output path",
    )
    return parser


def _build_compare_argv(args: argparse.Namespace) -> List[str]:
    argv = [
        "--onnx",
        str(args.onnx),
        "--imgsz",
        str(args.imgsz),
        "--metrics",
        args.metrics,
        "--json-out",
        str(args.json_out),
    ]
    if args.rknn is not None:
        argv.extend(["--rknn", str(args.rknn)])
    if args.img is not None:
        argv.extend(["--img", str(args.img)])
    if args.quant:
        argv.append("--quant")
    if args.calib_dir is not None:
        argv.extend(["--calib-dir", str(args.calib_dir)])
    return argv


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated("scripts/compare_onnx_rknn.py", "tools/compare.py", once=True)
    return compare_main(_build_compare_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
