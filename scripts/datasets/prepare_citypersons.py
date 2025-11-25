#!/usr/bin/env python3
"""
Convert CityPersons annotations to YOLO format

CityPersons annotation format (JSON):
{
    "imgHeight": 1024,
    "imgWidth": 2048,
    "objects": [
        {
            "label": "person",
            "bbox": [x, y, width, height],
            "bboxVis": [x_vis, y_vis, w_vis, h_vis]  # visible part
        }
    ]
}

YOLO format:
<class_id> <x_center> <y_center> <width> <height>
(normalized to [0, 1])
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import shutil
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Convert [x, y, w, h] to YOLO format [x_center, y_center, w, h] normalized"""
    x, y, w, h = bbox

    # Calculate center
    x_center = x + w / 2.0
    y_center = y + h / 2.0

    # Normalize
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Clamp to [0, 1]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return [x_center_norm, y_center_norm, w_norm, h_norm]


def process_citypersons_annotation(ann_file: Path, img_width: int = 2048, img_height: int = 1024,
                                   use_visible: bool = True) -> List[List[float]]:
    """
    Process CityPersons JSON annotation

    Args:
        ann_file: Path to annotation JSON
        img_width: Image width (default 2048 for CityScapes)
        img_height: Image height (default 1024 for CityScapes)
        use_visible: Use bboxVis (visible part) instead of full bbox

    Returns:
        List of YOLO format annotations: [class_id, x_center, y_center, w, h]
    """
    try:
        with open(ann_file, 'r') as f:
            data = json.load(f)
    except (IOError, OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load {ann_file}: {e}")
        return []

    yolo_annotations = []

    for obj in data.get('objects', []):
        # Filter for persons only
        if obj.get('label') != 'person':
            continue

        # Use visible bbox if available and requested
        if use_visible and 'bboxVis' in obj:
            bbox = obj['bboxVis']
        else:
            bbox = obj['bbox']

        # Skip invalid boxes
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue

        # Convert to YOLO format
        yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Person class_id = 0
        yolo_annotations.append([0] + yolo_bbox)

    return yolo_annotations


def prepare_citypersons_dataset(
    raw_dir: Path = Path("datasets/citypersons/raw"),
    output_dir: Path = Path("datasets/citypersons/yolo"),
    use_visible: bool = True,
    train_cities: List[str] = None,
    val_cities: List[str] = None
):
    """
    Prepare CityPersons dataset in YOLO format

    Args:
        raw_dir: Directory with raw CityPersons data
        output_dir: Output directory for YOLO format
        use_visible: Use visible bbox (recommended for occluded persons)
        train_cities: List of cities for training (default: most cities)
        val_cities: List of cities for validation (default: 3 cities)
    """
    logger.info("="*60)
    logger.info("CityPersons to YOLO Format Conversion")
    logger.info("="*60)

    # Default city splits (CityScapes standard split)
    if train_cities is None:
        train_cities = ['aachen', 'bremen', 'darmstadt', 'erfurt', 'hanover',
                       'krefeld', 'strasbourg', 'tubingen', 'weimar', 'bochum',
                       'cologne', 'dusseldorf', 'hamburg', 'jena', 'monchengladbach',
                       'stuttgart', 'ulm', 'zurich']

    if val_cities is None:
        val_cities = ['frankfurt', 'lindau', 'munster']

    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Find annotation files
    ann_base = raw_dir / "gtBboxCityPersons"

    train_ann_dir = ann_base / "train"
    val_ann_dir = ann_base / "val"

    if not train_ann_dir.exists() or not val_ann_dir.exists():
        raise FileNotFoundError(
            f"Annotations not found. Expected:\n"
            f"  {train_ann_dir}\n"
            f"  {val_ann_dir}\n"
            f"Run download_citypersons.sh first."
        )

    stats = {
        'train': {'images': 0, 'persons': 0},
        'val': {'images': 0, 'persons': 0}
    }

    # Process training set
    logger.info("\nProcessing training set...")
    for city_dir in train_ann_dir.iterdir():
        if not city_dir.is_dir():
            continue

        city_name = city_dir.name

        for ann_file in tqdm(list(city_dir.glob("*.json")), desc=f"  {city_name}"):
            # Find corresponding image
            img_name = ann_file.stem.replace("_gtBboxCityPersons", "_leftImg8bit")
            img_file = raw_dir / "leftImg8bit" / "train" / city_name / f"{img_name}.png"

            if not img_file.exists():
                logger.warning(f"Image not found: {img_file}")
                continue

            # Convert annotations
            yolo_anns = process_citypersons_annotation(ann_file, use_visible=use_visible)

            if len(yolo_anns) == 0:
                continue  # Skip images with no persons

            # Copy image
            dst_img = output_dir / "train" / "images" / f"{img_name}.png"
            if not dst_img.exists():
                shutil.copy(img_file, dst_img)

            # Write YOLO labels
            dst_label = output_dir / "train" / "labels" / f"{img_name}.txt"
            with open(dst_label, 'w') as f:
                for ann in yolo_anns:
                    f.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

            stats['train']['images'] += 1
            stats['train']['persons'] += len(yolo_anns)

    # Process validation set
    logger.info("\nProcessing validation set...")
    for city_dir in val_ann_dir.iterdir():
        if not city_dir.is_dir():
            continue

        city_name = city_dir.name

        for ann_file in tqdm(list(city_dir.glob("*.json")), desc=f"  {city_name}"):
            img_name = ann_file.stem.replace("_gtBboxCityPersons", "_leftImg8bit")
            img_file = raw_dir / "leftImg8bit" / "val" / city_name / f"{img_name}.png"

            if not img_file.exists():
                logger.warning(f"Image not found: {img_file}")
                continue

            yolo_anns = process_citypersons_annotation(ann_file, use_visible=use_visible)

            if len(yolo_anns) == 0:
                continue

            dst_img = output_dir / "val" / "images" / f"{img_name}.png"
            if not dst_img.exists():
                shutil.copy(img_file, dst_img)

            dst_label = output_dir / "val" / "labels" / f"{img_name}.txt"
            with open(dst_label, 'w') as f:
                for ann in yolo_anns:
                    f.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

            stats['val']['images'] += 1
            stats['val']['persons'] += len(yolo_anns)

    # Create dataset YAML
    yaml_content = f"""# CityPersons Dataset Configuration
# Converted from CityPersons to YOLO format

path: {output_dir.absolute()}
train: train/images
val: val/images

# Classes
nc: 1  # number of classes
names: ['person']

# Dataset statistics
# Train: {stats['train']['images']} images, {stats['train']['persons']} persons
# Val:   {stats['val']['images']} images, {stats['val']['persons']} persons
"""

    yaml_path = output_dir / "citypersons.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    logger.info("\n" + "="*60)
    logger.info("Dataset Preparation Complete!")
    logger.info("="*60)
    logger.info(f"\nStatistics:")
    logger.info(f"  Train: {stats['train']['images']} images, {stats['train']['persons']} persons")
    logger.info(f"  Val:   {stats['val']['images']} images, {stats['val']['persons']} persons")
    logger.info(f"  Avg persons/image: {stats['train']['persons'] / max(1, stats['train']['images']):.1f} (train)")
    logger.info(f"\nDataset YAML: {yaml_path}")
    logger.info(f"\nNext step:")
    logger.info(f"  yolo train model=yolo11n.pt data={yaml_path} epochs=50 imgsz=640")

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert CityPersons to YOLO format')
    parser.add_argument('--raw-dir', type=Path, default=Path('datasets/citypersons/raw'))
    parser.add_argument('--output-dir', type=Path, default=Path('datasets/citypersons/yolo'))
    parser.add_argument('--use-full-bbox', action='store_true',
                       help='Use full bbox instead of visible bbox (not recommended)')
    args = parser.parse_args()

    prepare_citypersons_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        use_visible=not args.use_full_bbox
    )
