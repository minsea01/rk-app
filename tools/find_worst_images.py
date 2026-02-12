#!/usr/bin/env python3
"""
Rank images by over-prediction using YOLO val predictions.json.

This script is intentionally robust to variations in Ultralytics output:
- If predictions.json is COCO-style (list of dicts with image_id), it groups by image_id
- If predictions.json has richer structure with file_name, it uses that

Usage:
  python tools/find_worst_images.py --pred /path/to/runs/detect/<run>/predictions.json

Or auto-discover latest predictions.json under runs/detect/:
  python tools/find_worst_images.py --auto --top 50
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def find_latest_predictions(root: Path) -> Optional[Path]:
    files = list(root.rglob("runs/detect/**/predictions.json"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_predictions(pred_path: Path) -> Any:
    return json.loads(pred_path.read_text())


def group_counts(pred: Any) -> Counter:
    counts: Counter = Counter()
    if isinstance(pred, list):
        # COCO-style list of dicts
        for d in pred:
            if not isinstance(d, dict):
                continue
            img_key = str(d.get("image_id", "unknown"))
            counts[img_key] += 1
        return counts
    if isinstance(pred, dict):
        # Try nested formats
        # 1) {"images": [{"file_name":..., "instances":[{"tp":0/1,...}]}]}
        if "images" in pred and isinstance(pred["images"], list):
            for img in pred["images"]:
                key = img.get("file_name") or str(img.get("id", "unknown"))
                inst = img.get("instances") or img.get("predictions") or []
                # Prefer counting predicted boxes; some formats include TP flags
                for _ in inst:
                    counts[str(key)] += 1
            return counts
        # 2) {"predictions": {<img_key>: [..]}}
        if "predictions" in pred and isinstance(pred["predictions"], dict):
            for k, arr in pred["predictions"].items():
                counts[str(k)] += len(arr) if isinstance(arr, list) else 0
            return counts
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank images by number of predictions")
    parser.add_argument("--pred", type=str, default=None, help="Path to predictions.json")
    parser.add_argument("--auto", action="store_true", help="Auto-find the latest predictions.json")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    pred_path: Optional[Path] = Path(args.pred).resolve() if args.pred else None
    if args.auto or pred_path is None:
        pred_path = find_latest_predictions(project_root)
    if pred_path is None or not pred_path.exists():
        print("[ERROR] predictions.json not found. Run val with save_json=True first.")
        return 2

    data = load_predictions(pred_path)
    counts = group_counts(data)
    if not counts:
        print("[WARN] Could not parse predictions format or it contains no entries.")
        return 1

    print(f"Top {args.top} images by number of predictions in: {pred_path}")
    for key, num in counts.most_common(args.top):
        print(f"{num}\t{key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
