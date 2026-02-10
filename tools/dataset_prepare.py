#!/usr/bin/env python3
"""Unified dataset preparation entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import yaml


def _iter_images(directory: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    images: List[Path] = []
    for ext in exts:
        images.extend(directory.rglob(ext))
    return images


def _to_yolo_bbox(bbox: List[float], width: float, height: float) -> str:
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / width
    y_center = (y + h / 2.0) / height
    w_norm = w / width
    h_norm = h / height
    return f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def prepare_coco_person_from_json(coco_dir: Path, output_dir: Path, copy_images: bool) -> int:
    coco_dir = coco_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": coco_dir / "annotations" / "instances_train2017.json",
        "val": coco_dir / "annotations" / "instances_val2017.json",
    }
    total_images = 0

    for split, ann_path in split_map.items():
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        images = {img["id"]: img for img in coco["images"]}
        per_image: Dict[int, List[List[float]]] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("category_id") == 1:
                per_image[ann["image_id"]].append(ann["bbox"])

        labels_dir = output_dir / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_list = []

        for image_id, bboxes in per_image.items():
            image_info = images.get(image_id)
            if not image_info:
                continue
            file_name = image_info["file_name"]
            width = float(image_info["width"])
            height = float(image_info["height"])

            yolo_lines = [_to_yolo_bbox(bbox, width, height) for bbox in bboxes]
            (labels_dir / f"{Path(file_name).stem}.txt").write_text(
                "\n".join(yolo_lines), encoding="utf-8"
            )
            image_list.append(file_name)

        (output_dir / f"{split}_images.txt").write_text("\n".join(image_list), encoding="utf-8")
        total_images += len(image_list)

        if copy_images:
            src_dir = coco_dir / f"{split}2017"
            dst_dir = output_dir / "images" / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            for rel in image_list:
                src = src_dir / rel
                dst = dst_dir / rel
                if src.exists():
                    shutil.copy2(src, dst)

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return total_images


def prepare_coco_person_from_yolo_labels(coco_root: Path, output_dir: Path, copy_mode: bool) -> int:
    coco_root = coco_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ("train2017", "val2017"):
        images_dir = coco_root / "images" / split
        labels_dir = coco_root / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            continue

        out_split = "train" if "train" in split else "val"
        out_images = output_dir / out_split / "images"
        out_labels = output_dir / out_split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for label_path in labels_dir.glob("*.txt"):
            lines = label_path.read_text(encoding="utf-8").splitlines()
            person_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == "0":
                    person_lines.append(line)
            if not person_lines:
                continue

            img_path = images_dir / f"{label_path.stem}.jpg"
            if not img_path.exists():
                continue

            out_img = out_images / img_path.name
            out_lbl = out_labels / label_path.name
            if copy_mode:
                shutil.copy2(img_path, out_img)
            elif not out_img.exists():
                out_img.symlink_to(img_path.resolve())
            out_lbl.write_text("\n".join(person_lines), encoding="utf-8")
            total += 1

    (output_dir / "coco_person.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: train/images",
                "val: val/images",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return total


def make_calib_from_data_yaml(data_yaml: Path, output_dir: Path, num_samples: int, seed: int) -> int:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = Path(cfg["path"])
    train_path = dataset_root / cfg["train"]

    images = _iter_images(train_path)
    random.Random(seed).shuffle(images)
    selected = images[:num_samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "calib.txt"
    output_file.write_text(
        "\n".join(str(p.resolve()) for p in selected),
        encoding="utf-8",
    )
    return len(selected)


def make_calib_from_image_dir(
    data_dir: Path,
    output_file: Path,
    num_samples: int,
    min_size: int,
    seed: int,
) -> int:
    all_images = _iter_images(data_dir)
    valid: List[Path] = []
    for image_path in all_images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        if min(h, w) >= min_size:
            valid.append(image_path)

    if not valid:
        return 0

    grouped: Dict[str, List[Path]] = defaultdict(list)
    for item in valid:
        grouped[str(item.parent)].append(item)

    rng = random.Random(seed)
    selected: List[Path] = []
    per_group = max(1, num_samples // max(1, len(grouped)))
    for group_images in grouped.values():
        picks = min(per_group, len(group_images))
        selected.extend(rng.sample(group_images, picks))

    if len(selected) < num_samples:
        remain = [p for p in valid if p not in selected]
        rng.shuffle(remain)
        selected.extend(remain[: max(0, num_samples - len(selected))])

    selected = selected[:num_samples]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(str(p.resolve()) for p in selected),
        encoding="utf-8",
    )
    return len(selected)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified dataset preparation tool")
    sub = parser.add_subparsers(dest="command", required=True)

    coco_json = sub.add_parser("coco-json", help="Prepare COCO person dataset from COCO JSON annotations")
    coco_json.add_argument("--coco-dir", type=Path, required=True)
    coco_json.add_argument("--output-dir", type=Path, default=Path("datasets/coco_person"))
    coco_json.add_argument("--copy-images", action="store_true")

    coco_yolo = sub.add_parser("yolo-labels", help="Prepare COCO person dataset from YOLO labels")
    coco_yolo.add_argument("--coco-root", type=Path, required=True)
    coco_yolo.add_argument("--output", type=Path, default=Path("datasets/coco_person"))
    coco_yolo.add_argument("--copy", action="store_true")

    calib_yaml = sub.add_parser("calib-from-data", help="Build calib.txt by sampling dataset yaml train split")
    calib_yaml.add_argument("--data", type=Path, required=True)
    calib_yaml.add_argument("--output", type=Path, required=True)
    calib_yaml.add_argument("--num", type=int, default=300)
    calib_yaml.add_argument("--seed", type=int, default=42)

    calib_dir = sub.add_parser("calib-from-dir", help="Build quant dataset list from image directory")
    calib_dir.add_argument("data_dir", type=Path)
    calib_dir.add_argument("-o", "--output", type=Path, default=Path("config/quant_dataset.txt"))
    calib_dir.add_argument("-n", "--num-samples", type=int, default=300)
    calib_dir.add_argument("--min-size", type=int, default=200)
    calib_dir.add_argument("--seed", type=int, default=42)

    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "coco-json":
        count = prepare_coco_person_from_json(args.coco_dir, args.output_dir, args.copy_images)
        print(f"Prepared COCO-person dataset from JSON: {count} images")
        return 0

    if args.command == "yolo-labels":
        count = prepare_coco_person_from_yolo_labels(args.coco_root, args.output, args.copy)
        print(f"Prepared COCO-person dataset from YOLO labels: {count} images")
        return 0

    if args.command == "calib-from-data":
        count = make_calib_from_data_yaml(args.data, args.output, args.num, args.seed)
        print(f"Generated calibration list from data.yaml: {count} images -> {args.output / 'calib.txt'}")
        return 0

    if args.command == "calib-from-dir":
        count = make_calib_from_image_dir(
            args.data_dir,
            args.output,
            args.num_samples,
            args.min_size,
            args.seed,
        )
        if count <= 0:
            print("No valid images found for quantization dataset")
            return 1
        print(f"Generated calibration list from directory: {count} images -> {args.output}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

