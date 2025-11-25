#!/usr/bin/env python3
"""
One-click YOLO dataset audit and optional validation run.

Features:
- Parse dataset YAML to resolve train/val image and label directories
- Check label files for:
  - empty files, malformed lines, non-integer class ids
  - out-of-range class ids vs names[], negative/NaN coords, coords outside [0,1]
  - zero-area boxes
  - image/label mismatches (images without labels, labels without images)
- Generate overlay previews for a few samples per split
- Optional: run a YOLO val with plots and save_json

Usage examples:
  python tools/yolo_data_audit.py \
    --data /path/to/data.yaml --overlay-samples 24 --out diagnosis_results/audit_$(date +%y%m%d_%H%M)

  python tools/yolo_data_audit.py \
    --auto --overlay-samples 24 --run-val --imgsz 960 --conf 0.001 --iou 0.65
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError as exc:  # pragma: no cover - defensive
        print("[WARN] PyYAML not found. Install with: pip install pyyaml", file=sys.stderr)
        raise exc


def _now_tag() -> str:
    return time.strftime("%y%m%d_%H%M%S")


def _resolve(path: Optional[str]) -> Optional[Path]:
    return None if path is None else Path(path).expanduser().resolve()


def _strip_ext(p: Path) -> str:
    return p.name[: -len(p.suffix)] if p.suffix else p.name


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_latest_match(root: Path, pattern: str) -> Optional[Path]:
    candidates = list(root.rglob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@dataclass
class DatasetPaths:
    yaml_path: Path
    root: Path
    train_images: Path
    val_images: Path
    train_labels: Path
    val_labels: Path
    names: List[str]


def load_dataset_paths(dataset_yaml: Path) -> DatasetPaths:
    yaml = _import_yaml()
    cfg = yaml.safe_load(dataset_yaml.read_text())

    # names can be list or dict
    names: List[str]
    if isinstance(cfg.get("names"), dict):
        names = [v for k, v in sorted(cfg["names"].items(), key=lambda kv: int(kv[0]))]
    else:
        names = list(cfg.get("names", []))

    path_root = cfg.get("path", None)
    root = (_resolve(path_root) if path_root else dataset_yaml.parent).resolve()

    def resolve_child(key: str) -> Path:
        v = cfg.get(key)
        if v is None:
            raise ValueError(f"Dataset YAML missing '{key}' entry: {dataset_yaml}")
        p = Path(v)
        if not p.is_absolute():
            p = (root / p).resolve()
        return p

    train_images = resolve_child("train")
    val_images = resolve_child("val")

    def images_to_labels_dir(images_dir: Path) -> Path:
        # Common convention: replace /images with /labels at same depth
        parts = list(images_dir.parts)
        try:
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts)
        except ValueError:
            # Fallback: sibling directory named labels
            candidate = images_dir.parent / "labels"
            return candidate

    train_labels = images_to_labels_dir(train_images)
    val_labels = images_to_labels_dir(val_images)

    return DatasetPaths(
        yaml_path=dataset_yaml,
        root=root,
        train_images=train_images,
        val_images=val_images,
        train_labels=train_labels,
        val_labels=val_labels,
        names=names,
    )


@dataclass
class LabelAudit:
    files_total: int
    files_empty: List[Path]
    images_without_labels: List[Path]
    labels_without_images: List[Path]
    id_counts: Counter
    format_errors: List[str]
    invalid_ids: List[Tuple[Path, int, str]]
    out_of_range_ids: List[Tuple[Path, int, int]]
    coords_outside_unit: List[Tuple[Path, int, List[float]]]
    zero_area_boxes: List[Tuple[Path, int, List[float]]]


def _read_label_file(path: Path) -> List[str]:
    try:
        return path.read_text().splitlines()
    except (IOError, OSError, UnicodeDecodeError):
        return []


def audit_split(images_dir: Path, labels_dir: Path, names: Sequence[str]) -> LabelAudit:
    images = [p for p in images_dir.rglob("*") if p.is_file() and _is_image(p)]
    labels = [p for p in labels_dir.rglob("*.txt") if p.is_file()]

    images_basenames = {_strip_ext(p): p for p in images}
    labels_basenames = {_strip_ext(p): p for p in labels}

    images_wo_labels = [images_basenames[b] for b in sorted(images_basenames.keys() - labels_basenames.keys())]
    labels_wo_images = [labels_basenames[b] for b in sorted(labels_basenames.keys() - images_basenames.keys())]

    files_empty: List[Path] = []
    id_counts: Counter = Counter()
    format_errors: List[str] = []
    invalid_ids: List[Tuple[Path, int, str]] = []
    out_of_range_ids: List[Tuple[Path, int, int]] = []
    coords_outside_unit: List[Tuple[Path, int, List[float]]] = []
    zero_area_boxes: List[Tuple[Path, int, List[float]]] = []

    num_classes = len(names) if names else None

    for label_path in labels:
        lines = _read_label_file(label_path)
        if len(lines) == 0:
            files_empty.append(label_path)
            continue
        for li, line in enumerate(lines, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                format_errors.append(f"{label_path}:{li}: expected 'cls cx cy w h [..]', got: '{line}'")
                continue
            cls_token = parts[0]
            if not re.fullmatch(r"-?\d+", cls_token):
                invalid_ids.append((label_path, li, cls_token))
                continue
            cls_id = int(cls_token)
            if num_classes is not None and (cls_id < 0 or cls_id >= num_classes):
                out_of_range_ids.append((label_path, li, cls_id))
            id_counts[cls_id] += 1

            try:
                cx, cy, w, h = map(float, parts[1:5])
            except (ValueError, TypeError):
                format_errors.append(f"{label_path}:{li}: non-float box coords in: '{line}'")
                continue

            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                coords_outside_unit.append((label_path, li, [cx, cy, w, h]))
            if w <= 0.0 or h <= 0.0:
                zero_area_boxes.append((label_path, li, [cx, cy, w, h]))

    return LabelAudit(
        files_total=len(labels),
        files_empty=files_empty,
        images_without_labels=images_wo_labels,
        labels_without_images=labels_wo_images,
        id_counts=id_counts,
        format_errors=format_errors,
        invalid_ids=invalid_ids,
        out_of_range_ids=out_of_range_ids,
        coords_outside_unit=coords_outside_unit,
        zero_area_boxes=zero_area_boxes,
    )


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _xywhn_to_xyxy(px: int, py: int, cx: float, cy: float, bw: float, bh: float) -> Tuple[int, int, int, int]:
    x = (cx - bw / 2.0) * px
    y = (cy - bh / 2.0) * py
    x2 = (cx + bw / 2.0) * px
    y2 = (cy + bh / 2.0) * py
    return int(round(x)), int(round(y)), int(round(x2)), int(round(y2))


def draw_overlays(images_dir: Path, labels_dir: Path, names: Sequence[str], out_dir: Path, max_images: int = 24) -> None:
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        print("[WARN] OpenCV not available, skipping overlays. Install opencv-python-headless.", file=sys.stderr)
        return

    out_dir = _ensure_dir(out_dir)
    images = [p for p in images_dir.rglob("*") if p.is_file() and _is_image(p)]
    random.shuffle(images)
    images = images[: max_images]
    for img_path in images:
        label_path = (labels_dir / (_strip_ext(img_path) + ".txt"))
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for li, line in enumerate(_read_label_file(label_path), start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
            except (ValueError, TypeError, IndexError):
                continue
            x1, y1, x2, y2 = _xywhn_to_xyxy(w, h, cx, cy, bw, bh)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        out_path = out_dir / f"{_strip_ext(img_path)}_overlay.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2  # re-import to help static checkers
        cv2.imwrite(str(out_path), img)


def run_yolo_val(model_path: Path, data_yaml: Path, imgsz: int, device: str, conf: float, iou: float, project: Optional[Path], name: Optional[str]) -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:  # pragma: no cover
        print("[WARN] ultralytics not available. Skipping val. Install with: pip install ultralytics", file=sys.stderr)
        return
    model = YOLO(str(model_path))
    model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
        verbose=True,
        project=str(project) if project else None,
        name=name,
    )


def autodetect_defaults(project_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (best_or_last_weight, dataset_yaml) if can be inferred from recent runs."""
    latest_args = find_latest_match(project_root / "runs" / "detect", "**/args.yaml")
    data_yaml = None
    if latest_args and latest_args.exists():
        try:
            yaml = _import_yaml()
            cfg = yaml.safe_load(latest_args.read_text())
            if isinstance(cfg.get("data"), str):
                data_yaml = _resolve(cfg["data"])  # type: ignore
        except (IOError, OSError, ValueError, ImportError):
            pass  # Ignore failures to parse args.yaml
    best = find_latest_match(project_root / "runs" / "detect", "**/weights/best.pt")
    last = find_latest_match(project_root / "runs" / "detect", "**/weights/last.pt")
    weight = best or last
    return weight, data_yaml


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="YOLO dataset audit and optional validation")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset YAML")
    parser.add_argument("--auto", action="store_true", help="Auto-detect dataset+weights from latest run")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (best.pt/last.pt)")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--overlay-samples", type=int, default=24)
    parser.add_argument("--out", type=str, default=f"diagnosis_results/audit_{_now_tag()}")
    parser.add_argument("--run-val", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]

    model_path: Optional[Path] = _resolve(args.model)
    data_yaml: Optional[Path] = _resolve(args.data)
    if args.auto:
        auto_weight, auto_data = autodetect_defaults(project_root)
        if model_path is None:
            model_path = auto_weight
        if data_yaml is None:
            data_yaml = auto_data

    if data_yaml is None:
        print("[ERROR] Dataset YAML not provided and auto-detect failed. Use --data /path/to/data.yaml", file=sys.stderr)
        return 2

    ds = load_dataset_paths(data_yaml)
    out_dir = _ensure_dir(project_root / args.out)

    # Per-split audits
    print(f"[INFO] Auditing train labels in: {ds.train_labels}")
    train_audit = audit_split(ds.train_images, ds.train_labels, ds.names)
    print(f"[INFO] Auditing val labels in: {ds.val_labels}")
    val_audit = audit_split(ds.val_images, ds.val_labels, ds.names)

    # Aggregate report
    summary = {
        "dataset_yaml": str(ds.yaml_path),
        "names": ds.names,
        "num_classes": len(ds.names),
        "train": {
            "images": str(ds.train_images),
            "labels": str(ds.train_labels),
            "files_total": train_audit.files_total,
            "files_empty": [str(p) for p in train_audit.files_empty],
            "images_without_labels": [str(p) for p in train_audit.images_without_labels],
            "labels_without_images": [str(p) for p in train_audit.labels_without_images],
            "id_counts": dict(train_audit.id_counts),
            "format_errors": train_audit.format_errors,
            "invalid_ids": [(str(p), ln, tok) for p, ln, tok in train_audit.invalid_ids],
            "out_of_range_ids": [(str(p), ln, cid) for p, ln, cid in train_audit.out_of_range_ids],
            "coords_outside_unit": [(str(p), ln, xywh) for p, ln, xywh in train_audit.coords_outside_unit],
            "zero_area_boxes": [(str(p), ln, xywh) for p, ln, xywh in train_audit.zero_area_boxes],
        },
        "val": {
            "images": str(ds.val_images),
            "labels": str(ds.val_labels),
            "files_total": val_audit.files_total,
            "files_empty": [str(p) for p in val_audit.files_empty],
            "images_without_labels": [str(p) for p in val_audit.images_without_labels],
            "labels_without_images": [str(p) for p in val_audit.labels_without_images],
            "id_counts": dict(val_audit.id_counts),
            "format_errors": val_audit.format_errors,
            "invalid_ids": [(str(p), ln, tok) for p, ln, tok in val_audit.invalid_ids],
            "out_of_range_ids": [(str(p), ln, cid) for p, ln, cid in val_audit.out_of_range_ids],
            "coords_outside_unit": [(str(p), ln, xywh) for p, ln, xywh in val_audit.coords_outside_unit],
            "zero_area_boxes": [(str(p), ln, xywh) for p, ln, xywh in val_audit.zero_area_boxes],
        },
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OK] Wrote audit summary: {(out_dir / 'audit_summary.json').resolve()}")

    # Overlays
    if not args.no_overlays:
        print("[INFO] Rendering overlays (train)...")
        draw_overlays(ds.train_images, ds.train_labels, ds.names, out_dir / "overlays_train", max_images=args.overlay_samples)
        print("[INFO] Rendering overlays (val)...")
        draw_overlays(ds.val_images, ds.val_labels, ds.names, out_dir / "overlays_val", max_images=args.overlay_samples)

    # Optional validation
    if args.run_val:
        if model_path is None:
            # Try to auto-detect if not provided explicitly
            model_path, _ = autodetect_defaults(project_root)
        if model_path is None:
            print("[WARN] No model weights found to run validation.")
        else:
            print(f"[INFO] Running YOLO val with model: {model_path}")
            run_yolo_val(
                model_path=model_path,
                data_yaml=ds.yaml_path,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                project=project_root / "runs" / "detect",
                name=f"audit_val_{_now_tag()}",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


