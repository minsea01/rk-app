#!/usr/bin/env python3
"""Build a deterministic video loop for board-side no-camera acceptance tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Image folder to loop")
    parser.add_argument("--output", required=True, type=Path, help="Output AVI path")
    parser.add_argument("--fps", default=30.0, type=float, help="Video FPS")
    parser.add_argument("--frames", default=1800, type=int, help="Total frames to write")
    parser.add_argument("--width", default=416, type=int, help="Output width")
    parser.add_argument("--height", default=416, type=int, help="Output height")
    parser.add_argument("--codec", default="MJPG", help="FourCC codec, default MJPG")
    return parser.parse_args()


def list_images(source: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if len(args.codec) != 4:
        raise SystemExit("--codec must be exactly four characters")
    if not args.source.is_dir():
        raise SystemExit(f"source folder not found: {args.source}")

    images = list_images(args.source)
    if not images:
        raise SystemExit(f"no images found under: {args.source}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit(f"failed to open VideoWriter: {args.output}")

    for index in range(args.frames):
        image_path = images[index % len(images)]
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            writer.release()
            raise SystemExit(f"failed to read image: {image_path}")
        if frame.shape[1] != args.width or frame.shape[0] != args.height:
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)

    writer.release()
    print(f"wrote {args.frames} frames to {args.output}")


if __name__ == "__main__":
    main()
