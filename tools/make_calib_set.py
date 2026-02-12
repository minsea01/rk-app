#!/usr/bin/env python3
"""Deprecated wrapper for tools/dataset_prepare.py calib-from-data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.deprecation import warn_deprecated
from tools.dataset_prepare import main as dataset_prepare_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deprecated: use tools/dataset_prepare.py calib-from-data"
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated(
        "tools/make_calib_set.py", "tools/dataset_prepare.py calib-from-data", once=True
    )
    return dataset_prepare_main(
        [
            "calib-from-data",
            "--data",
            str(args.data),
            "--output",
            str(args.output),
            "--num",
            str(args.num),
            "--seed",
            str(args.seed),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
