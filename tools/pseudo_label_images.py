#!/usr/bin/env python3
"""Generate a YOLO-format pseudo-labeled dataset from unlabeled images.

The script is designed for the common workflow in this repository:
1. Start from an existing ``best.pt`` model.
2. Run inference on an unlabeled image folder.
3. Materialize a train/val dataset with YOLO txt labels.
4. Manually review and correct labels before fine-tuning.

Images are split into train/val in a prefix-aware way. For filenames such as
``back-side-view-1001.jpg`` and ``pexels-123.jpg`` this keeps each scene type
represented in both splits instead of shuffling the whole folder blindly.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.logger import setup_logger

logger = setup_logger(__name__, level="INFO")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitPlan:
    split_by_source: Mapping[Path, str]
    groups: Mapping[str, List[Path]]


def _iter_images(root: Path, recursive: bool) -> List[Path]:
    iterator: Iterable[Path]
    iterator = root.rglob("*") if recursive else root.iterdir()
    images = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def infer_group_key(path: Path) -> str:
    """Infer a stable group key from a filename.

    Examples:
        back-side-view-1001.jpg -> back-side-view
        pexels-12345.png -> pexels
        IMG_0012.jpg -> img
    """

    stem = path.stem.lower()
    stem = re.sub(r"[-_]*\d+$", "", stem)
    stem = re.sub(r"[-_]+$", "", stem)
    if not stem:
        return "ungrouped"
    return stem


def build_split_plan(images: Sequence[Path], val_ratio: float, seed: int) -> SplitPlan:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for image in images:
        groups[infer_group_key(image)].append(image)

    rng = random.Random(seed)
    split_by_source: Dict[Path, str] = {}

    for group_images in groups.values():
        ordered = list(group_images)
        rng.shuffle(ordered)

        if len(ordered) <= 1:
            val_count = 0
        else:
            val_count = int(round(len(ordered) * val_ratio))
            if val_ratio > 0.0:
                val_count = max(1, val_count)
            val_count = min(val_count, len(ordered) - 1)

        val_images = set(ordered[:val_count])
        for image in ordered:
            split_by_source[image] = "val" if image in val_images else "train"

    return SplitPlan(split_by_source=split_by_source, groups=groups)


def parse_class_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    parsed = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        parsed.append(int(token))
    return parsed or None


def resolve_names(
    model_names: Mapping[int, str] | Sequence[str], class_ids: Optional[List[int]]
) -> Dict[int, str]:
    if isinstance(model_names, Mapping):
        normalized = {int(k): str(v) for k, v in model_names.items()}
    else:
        normalized = {idx: str(name) for idx, name in enumerate(model_names)}

    if class_ids is None:
        return normalized

    missing = [class_id for class_id in class_ids if class_id not in normalized]
    if missing:
        raise ValueError(f"class ids not found in model names: {missing}")

    return {new_id: normalized[old_id] for new_id, old_id in enumerate(class_ids)}


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def load_yolo_model(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required for pseudo-label inference; install the train "
            "extras or run `pip install ultralytics`"
        ) from exc
    return YOLO(str(weights))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize_image(src: Path, dst: Path, copy_images: bool) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        logger.warning("symlink failed for %s, falling back to copy", src)
        shutil.copy2(src, dst)


def select_overlay_sources(images: Sequence[Path], count: int, seed: int) -> Set[Path]:
    if count <= 0 or not images:
        return set()
    rng = random.Random(seed)
    chosen = list(images)
    rng.shuffle(chosen)
    return set(chosen[: min(count, len(chosen))])


def write_label_file(
    label_path: Path,
    boxes,
    class_remap: Optional[Dict[int, int]],
) -> Dict[str, int]:
    ensure_parent(label_path)
    if boxes is None or len(boxes) == 0:
        label_path.write_text("", encoding="utf-8")
        return {"boxes": 0, "tiny_boxes": 0}

    xywhn = boxes.xywhn.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    lines: List[str] = []
    tiny_boxes = 0
    for cls_id, (cx, cy, w, h) in zip(cls_ids, xywhn):
        out_cls = class_remap[cls_id] if class_remap is not None else cls_id
        lines.append(f"{out_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if w * h < 0.001:
            tiny_boxes += 1

    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"boxes": len(lines), "tiny_boxes": tiny_boxes}


def dump_data_yaml(out_root: Path, names: Mapping[int, str]) -> None:
    payload = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": [names[idx] for idx in sorted(names)],
    }
    (out_root / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a pseudo-labeled YOLO dataset from an unlabeled image folder."
    )
    parser.add_argument("--weights", required=True, type=Path, help="Path to .pt weights")
    parser.add_argument("--source", required=True, type=Path, help="Source image directory")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset root")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Optional comma-separated class ids, e.g. 0 or 0,2,3",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlay-samples", type=int, default=48)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlink")
    parser.add_argument("--recursive", action="store_true", help="Scan source recursively")
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")
    if not args.source.is_dir():
        raise SystemExit(f"source dir not found: {args.source}")
    if not 0.0 <= args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be in [0, 1)")

    source_root = args.source.resolve()
    out_root = args.out.resolve()
    images = _iter_images(source_root, recursive=args.recursive)
    if not images:
        raise SystemExit(f"no images found under: {source_root}")

    class_ids = parse_class_ids(args.classes)
    device = resolve_device(args.device)

    logger.info("loading model: %s", args.weights)
    model = load_yolo_model(args.weights)
    dataset_names = resolve_names(model.names, class_ids)

    split_plan = build_split_plan(images, val_ratio=args.val_ratio, seed=args.seed)
    overlay_sources = select_overlay_sources(
        [image for image in images if split_plan.split_by_source[image] == "val"],
        count=args.overlay_samples,
        seed=args.seed,
    )

    logger.info(
        "pseudo-labeling %d images -> %s (device=%s, classes=%s)",
        len(images),
        out_root,
        device,
        class_ids if class_ids is not None else "all",
    )

    class_remap = None
    if class_ids is not None:
        class_remap = {old_id: new_id for new_id, old_id in enumerate(class_ids)}

    stats = {
        "source": str(source_root),
        "weights": str(args.weights.resolve()),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": device,
        "val_ratio": args.val_ratio,
        "classes": class_ids,
        "model_names": {int(k): v for k, v in dataset_names.items()},
        "total_images": len(images),
        "groups": {group: len(group_images) for group, group_images in split_plan.groups.items()},
        "splits": {
            "train": {"images": 0, "empty": 0, "boxes": 0, "tiny_boxes": 0},
            "val": {"images": 0, "empty": 0, "boxes": 0, "tiny_boxes": 0},
        },
    }
    manifest: List[Dict[str, object]] = []

    predict_kwargs = {
        "source": [str(image) for image in images],
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": device,
        "verbose": False,
        "stream": True,
        "max_det": args.max_det,
    }
    if class_ids is not None:
        predict_kwargs["classes"] = class_ids

    for image_path, result in zip(images, model.predict(**predict_kwargs)):
        rel_path = image_path.relative_to(source_root)
        split = split_plan.split_by_source[image_path]
        group = infer_group_key(image_path)

        dst_image = out_root / "images" / split / rel_path
        dst_label = out_root / "labels" / split / rel_path.with_suffix(".txt")
        materialize_image(image_path, dst_image, copy_images=args.copy_images)
        counts = write_label_file(dst_label, result.boxes, class_remap)

        overlay_path = None
        if image_path in overlay_sources:
            overlay_path = out_root / "overlays" / split / rel_path
            ensure_parent(overlay_path)
            plotted = result.plot()
            import cv2

            cv2.imwrite(str(overlay_path), plotted)

        split_stats = stats["splits"][split]
        split_stats["images"] += 1
        split_stats["boxes"] += counts["boxes"]
        split_stats["tiny_boxes"] += counts["tiny_boxes"]
        if counts["boxes"] == 0:
            split_stats["empty"] += 1

        manifest.append(
            {
                "source": str(image_path),
                "relative_path": str(rel_path),
                "split": split,
                "group": group,
                "boxes": counts["boxes"],
                "tiny_boxes": counts["tiny_boxes"],
                "overlay": str(overlay_path) if overlay_path else None,
            }
        )

    for split_name, split_stats in stats["splits"].items():
        image_count = max(1, split_stats["images"])
        box_count = max(1, split_stats["boxes"])
        split_stats["empty_rate"] = round(split_stats["empty"] / image_count, 4)
        split_stats["avg_boxes_per_image"] = round(split_stats["boxes"] / image_count, 4)
        split_stats["tiny_rate"] = round(split_stats["tiny_boxes"] / box_count, 4)

    empty_by_group = Counter()
    for item in manifest:
        if item["boxes"] == 0:
            empty_by_group[item["group"]] += 1
    stats["empty_by_group"] = dict(empty_by_group)

    out_root.mkdir(parents=True, exist_ok=True)
    dump_data_yaml(out_root, dataset_names)
    (out_root / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n",
        encoding="utf-8",
    )

    logger.info("dataset ready: %s", out_root)
    logger.info("data yaml: %s", out_root / "data.yaml")
    logger.info("stats: %s", out_root / "stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
