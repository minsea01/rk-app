#!/usr/bin/env python3
"""Deprecated wrapper for tools/dataset_prepare.py calib-from-dir."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.deprecation import warn_deprecated
from tools.dataset_prepare import main as dataset_prepare_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deprecated: use tools/dataset_prepare.py calib-from-dir"
    )
    parser.add_argument("data_dir", type=Path, help="Image directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("config/quant_dataset.txt"))
    parser.add_argument("-n", "--num-samples", type=int, default=300)
    parser.add_argument("--min-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated(
        "tools/prepare_quant_dataset.py", "tools/dataset_prepare.py calib-from-dir", once=True
    )
    return dataset_prepare_main(
        [
            "calib-from-dir",
            str(args.data_dir),
            "--output",
            str(args.output),
            "--num-samples",
            str(args.num_samples),
            "--min-size",
            str(args.min_size),
            "--seed",
            str(args.seed),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
