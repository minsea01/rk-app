#!/usr/bin/env python3
"""Deprecated wrapper for tools/dataset_prepare.py coco-json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.deprecation import warn_deprecated
from tools.dataset_prepare import main as dataset_prepare_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deprecated: use tools/dataset_prepare.py coco-json"
    )
    parser.add_argument(
        "--coco-dir", type=Path, required=True, help="Path to COCO dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/coco_person"),
        help="Output directory for processed dataset",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images into output dir")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated("tools/prepare_coco_person.py", "tools/dataset_prepare.py coco-json", once=True)
    return dataset_prepare_main(
        [
            "coco-json",
            "--coco-dir",
            str(args.coco_dir),
            "--output-dir",
            str(args.output_dir),
            *([] if not args.copy_images else ["--copy-images"]),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
