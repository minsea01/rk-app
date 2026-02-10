#!/usr/bin/env python3
"""Deprecated wrapper for scripts/train.sh."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.deprecation import warn_deprecated


def parse_batch(value: Union[str, int, float]) -> Union[int, float]:
    """Parse Ultralytics batch argument (int/float/auto)."""
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return -1
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid --batch value '{value}', expected int/float/auto"
            ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deprecated: use scripts/train.sh")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=parse_batch, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _build_train_cmd(args: argparse.Namespace) -> List[str]:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        "bash",
        str(repo_root / "scripts" / "train.sh"),
        "--profile",
        "none",
        "--model",
        args.model,
        "--data",
        args.data,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--project",
        args.project,
        "--name",
        args.name,
        "--patience",
        str(args.patience),
        "--extra",
        f"seed={args.seed}",
        "--no-export",
        "--no-summary",
    ]
    if args.lr0 is not None:
        cmd.extend(["--lr0", str(args.lr0)])
    if args.lrf is not None:
        cmd.extend(["--lrf", str(args.lrf)])
    return cmd


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated("tools/train_yolov8.py", "scripts/train.sh", once=True)

    cmd = _build_train_cmd(args)
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
