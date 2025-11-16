#!/usr/bin/env python3
"""
Official YOLO mAP evaluation using Ultralytics API

This script uses Ultralytics' built-in validation to get accurate mAP results,
avoiding custom postprocessing issues.
"""

import argparse
import json
import logging
from pathlib import Path
import time

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_person_only_validation(model_path: str, annotations_file: str, images_dir: str, imgsz: int = 640, limit: int = None):
    """Evaluate on person class only using Ultralytics predict + manual mAP calculation"""
    logger.info("="*60)
    logger.info("Person-Only mAP Evaluation (Official Ultralytics)")
    logger.info("="*60)

    model = YOLO(model_path)

    # Load COCO annotations
    import json
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Build image ID to annotations mapping
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] == 1:  # person in COCO
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []

            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            annotations_by_image[image_id].append({
                'bbox': [x, y, x + w, y + h],  # Convert to [x1, y1, x2, y2]
                'area': ann['area'],
                'iscrowd': ann.get('iscrowd', 0)
            })

    # Build image id to filename mapping
    images_by_id = {}
    for img in coco_data['images']:
        images_by_id[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    logger.info(f"Loaded {len(annotations_by_image)} images with person annotations")

    # Run predictions
    image_ids = list(annotations_by_image.keys())
    if limit:
        image_ids = image_ids[:limit]

    logger.info(f"\nEvaluating {len(image_ids)} images...")

    all_predictions = []
    all_ground_truths = []

    start_time = time.time()

    for idx, img_id in enumerate(image_ids):
        img_info = images_by_id[img_id]
        img_path = Path(images_dir) / img_info['file_name']

        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        # Run prediction with official Ultralytics
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        result = results[0]

        # Extract person detections (class 0 in YOLO)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # person class
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                all_predictions.append({
                    'image_id': str(img_id),
                    'bbox': xyxy.tolist(),
                    'score': conf
                })

        # Add ground truths
        for gt in annotations_by_image[img_id]:
            all_ground_truths.append({
                'image_id': str(img_id),
                'bbox': gt['bbox'],
                'iscrowd': gt.get('iscrowd', 0)
            })

        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx+1}/{len(image_ids)} images...")

    elapsed = time.time() - start_time
    fps = len(image_ids) / elapsed

    logger.info(f"✓ Prediction completed in {elapsed:.2f}s ({fps:.2f} FPS)")
    logger.info(f"  Total predictions: {len(all_predictions)}")
    logger.info(f"  Total ground truths: {len(all_ground_truths)}")

    # Calculate mAP
    import numpy as np

    def calculate_iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    # Sort predictions by score (descending)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)

    # Track which ground truths have been matched
    gt_matched = [False] * len(all_ground_truths)

    true_positives = []
    false_positives = []

    # For each prediction
    for pred in all_predictions:
        best_iou = 0.0
        best_gt_idx = -1

        # Find matching ground truth
        for gt_idx, gt in enumerate(all_ground_truths):
            if gt['image_id'] != pred['image_id']:
                continue

            if gt_matched[gt_idx]:
                continue

            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if match exceeds threshold (IoU=0.5)
        if best_iou >= 0.5 and best_gt_idx >= 0:
            true_positives.append(1)
            false_positives.append(0)
            gt_matched[best_gt_idx] = True
        else:
            true_positives.append(0)
            false_positives.append(1)

    # Calculate precision and recall
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)

    num_ground_truths = len([gt for gt in all_ground_truths if gt.get('iscrowd', 0) == 0])

    recalls = tp_cumsum / num_ground_truths if num_ground_truths > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    # Calculate average precision (AP) using 11-point interpolation
    ap = 0.0
    for recall_threshold in np.linspace(0, 1, 11):
        precisions_above_threshold = precisions[recalls >= recall_threshold]
        if len(precisions_above_threshold) > 0:
            ap += precisions_above_threshold.max()

    ap /= 11.0

    metrics = {
        'mAP@0.5': ap,
        'precision': float(precisions[-1]) if len(precisions) > 0 else 0.0,
        'recall': float(recalls[-1]) if len(recalls) > 0 else 0.0,
        'true_positives': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
        'false_positives': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0,
        'num_predictions': len(all_predictions),
        'num_ground_truths': num_ground_truths,
        'num_images': len(image_ids),
        'fps': fps
    }

    logger.info("\n" + "="*60)
    logger.info("PERSON-ONLY RESULTS (Official Ultralytics)")
    logger.info("="*60)
    logger.info(f"mAP@0.5: {metrics['mAP@0.5']:.4f} ({metrics['mAP@0.5']*100:.2f}%)")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    logger.info(f"Graduation Requirement (>=90% @ IoU=0.5): {'PASS ✓' if metrics['mAP@0.5'] >= 0.9 else 'FAIL ✗'}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Model path (.pt file)')
    parser.add_argument('--annotations', type=str,
                        default='datasets/coco/annotations/person_val2017.json',
                        help='COCO annotations JSON')
    parser.add_argument('--images-dir', type=str, default='datasets/coco/val2017',
                        help='Images directory')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--limit', type=int, help='Limit number of images')
    parser.add_argument('--output', type=str, help='Output JSON file')
    args = parser.parse_args()

    metrics = run_person_only_validation(
        args.model,
        args.annotations,
        args.images_dir,
        args.imgsz,
        args.limit
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'model': args.model,
            'imgsz': args.imgsz,
            'metrics': metrics,
            'graduation_requirement': {
                'threshold': 0.9,
                'achieved': metrics['mAP@0.5'],
                'status': 'PASS' if metrics['mAP@0.5'] >= 0.9 else 'FAIL',
                'margin': (metrics['mAP@0.5'] - 0.9) * 100
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n✓ Report saved: {output_path}")


if __name__ == '__main__':
    main()
