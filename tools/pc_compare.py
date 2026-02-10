#!/usr/bin/env python3
"""Deprecated wrapper for tools.compare."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.deprecation import warn_deprecated
from tools.compare import main as compare_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deprecated: use tools/compare.py")
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX model path")
    parser.add_argument("--img", type=Path, default=None, help="Optional test image")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Deprecated output directory; mapped to <outdir>/pc_compare_report.json",
    )
    return parser


def _build_compare_argv(args: argparse.Namespace) -> List[str]:
    argv = [
        "--onnx",
        str(args.onnx),
        "--imgsz",
        str(args.imgsz),
        "--num-classes",
        str(args.num_classes),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--metrics",
        "tensor,post",
    ]
    if args.img is not None:
        argv.extend(["--img", str(args.img)])
    if args.quant:
        argv.append("--quant")
    if args.calib_dir is not None:
        argv.extend(["--calib-dir", str(args.calib_dir)])
    if args.outdir is not None:
        argv.extend(["--json-out", str(args.outdir / "pc_compare_report.json")])
    return argv


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated("tools/pc_compare.py", "tools/compare.py", once=True)
    return compare_main(_build_compare_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
