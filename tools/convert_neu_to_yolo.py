#!/usr/bin/env python3
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEU_ROOT = REPO_ROOT / "temp_data" / "NEU_repo"


def _import_yaml():
    import yaml  # type: ignore
    return yaml


@dataclass
class DataCfg:
    root: Path
    train_images: Path
    val_images: Path
    names: List[str]


def load_data_cfg(data_yaml: Path) -> DataCfg:
    yaml = _import_yaml()
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent)).expanduser().resolve()
    train = cfg.get("train")
    val = cfg.get("val")
    if train is None or val is None:
        raise ValueError("data.yaml missing train/val entries")
    train_images = (root / train).resolve()
    val_images = (root / val).resolve()
    # names may be list or dict
    names: List[str]
    if isinstance(cfg.get("names"), dict):
        names = [v for k, v in sorted(cfg["names"].items(), key=lambda kv: int(kv[0]))]
    else:
        names = list(cfg.get("names", []))
    return DataCfg(root, train_images, val_images, names)


def build_xml_index(neu_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for sub in ("ANNOTATIONS", "Validation_Annotations"):
        d = neu_root / sub
        if not d.exists():
            continue
        for xml in d.rglob("*.xml"):
            index[xml.stem] = xml
    return index


def ensure_labels_dir(data_root: Path) -> Tuple[Path, Path]:
    labels_train = (data_root / "labels" / "train")
    labels_val = (data_root / "labels" / "val")
    labels_train.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)
    return labels_train, labels_val


def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[str, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size in {xml_path}")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))
    items: List[Tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        if name is None or bnd is None:
            continue
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        # Convert to YOLO normalized cx, cy, w, h
        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        cx = xmin + bw / 2.0
        cy = ymin + bh / 2.0
        if w <= 0 or h <= 0:
            continue
        items.append((name, cx / w, cy / h, bw / w, bh / h))
    return w, h, items


def write_yolo_labels(images_dir: Path, labels_dir: Path, xml_index: Dict[str, Path], name_to_id: Dict[str, int]) -> Tuple[int, int, int]:
    num_images = 0
    num_labeled = 0
    num_empty = 0
    for img in images_dir.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        num_images += 1
        stem = img.stem
        xml = xml_index.get(stem)
        out = labels_dir / f"{stem}.txt"
        lines: List[str] = []
        if xml and xml.exists():
            try:
                _, _, items = parse_voc_xml(xml)
                for (cls_name, cx, cy, bw, bh) in items:
                    if cls_name not in name_to_id:
                        # Skip classes not in the current dataset config
                        continue
                    cls_id = name_to_id[cls_name]
                    # guard
                    if bw <= 0 or bh <= 0:
                        continue
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            except (ValueError, ET.ParseError):
                # If parsing fails, leave as empty label
                lines = []
        if lines:
            num_labeled += 1
            out.write_text("\n".join(lines) + "\n")
        else:
            num_empty += 1
            out.write_text("")
    return num_images, num_labeled, num_empty


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert NEU VOC XML annotations to YOLO labels for an existing dataset")
    ap.add_argument("--data", required=True, help="Path to YOLO data.yaml for the target dataset")
    ap.add_argument("--neu_root", default=str(DEFAULT_NEU_ROOT), help="NEU repository root containing ANNOTATIONS/")
    args = ap.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    neu_root = Path(args.neu_root).expanduser().resolve()

    cfg = load_data_cfg(data_yaml)
    labels_train, labels_val = ensure_labels_dir(cfg.root)

    # Map dataset names to IDs
    name_to_id = {n: i for i, n in enumerate(cfg.names)}

    # Build XML index
    xml_index = build_xml_index(neu_root)
    if not xml_index:
        print(f"[ERROR] No XML annotations found under {neu_root}. Check --neu_root.")
        return 2

    # Train
    n_img_t, n_lab_t, n_empty_t = write_yolo_labels(cfg.train_images, labels_train, xml_index, name_to_id)
    # Val
    n_img_v, n_lab_v, n_empty_v = write_yolo_labels(cfg.val_images, labels_val, xml_index, name_to_id)

    print("[DONE] Labels written.")
    print(f"Train: images={n_img_t}, labeled={n_lab_t}, empty={n_empty_t}")
    print(f"Val  : images={n_img_v}, labeled={n_lab_v}, empty={n_empty_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


