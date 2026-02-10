#!/usr/bin/env python3
"""Deprecated wrapper for tools/dataset_prepare.py yolo-labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.deprecation import warn_deprecated
from tools.dataset_prepare import main as dataset_prepare_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deprecated: use tools/dataset_prepare.py yolo-labels")
    parser.add_argument("--coco-root", type=Path, default=Path("/root/autodl-tmp/coco"))
    parser.add_argument("--output", type=Path, default=Path("datasets/coco_person"))
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated("cloud_training/filter_coco_person.py", "tools/dataset_prepare.py yolo-labels", once=True)
    argv = [
        "yolo-labels",
        "--coco-root",
        str(args.coco_root),
        "--output",
        str(args.output),
    ]
    if args.copy:
        argv.append("--copy")
    return dataset_prepare_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

