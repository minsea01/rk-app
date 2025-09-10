#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run YOLO val with save_json=True")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default=None)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        print("[ERROR] ultralytics not installed. pip install ultralytics")
        return 2

    model = YOLO(args.model)
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
        verbose=True,
        project=args.project,
        name=args.name,
    )

    save_dir = None
    try:
        save_dir = getattr(results, "save_dir", None)
    except Exception:
        save_dir = None
    if save_dir:
        print(f"SAVE_DIR={save_dir}")
        pred_path = Path(save_dir) / "predictions.json"
        print(f"PRED_JSON={'FOUND' if pred_path.exists() else 'MISSING'}:{pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


