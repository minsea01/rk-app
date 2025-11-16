#!/usr/bin/env python3
"""
Pedestrian Detection mAP Evaluator

Purpose: Evaluate pedestrian detection performance with mAP@0.5 metric
Supports:
  - COCO person class subset
  - Penn-Fudan Pedestrian Detection Dataset
  - Custom annotated datasets
  - ONNX vs RKNN model comparison

Requirements: >= 90% mAP@0.5 for graduation thesis
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PedestrianEvaluator:
    """Evaluator for pedestrian detection performance"""

    def __init__(self, model_path: Path, model_type: str = 'onnx',
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 imgsz: int = 640):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load detection model"""
        logger.info(f"Loading {self.model_type.upper()} model: {self.model_path}")

        if self.model_type == 'onnx':
            self._load_onnx()
        elif self.model_type == 'rknn':
            self._load_rknn()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_onnx(self):
        """Load ONNX model with ONNXRuntime"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit("onnxruntime not installed. pip install onnxruntime")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = ort.InferenceSession(str(self.model_path), providers=providers)

        # Log provider
        provider = self.model.get_providers()[0]
        logger.info(f"  Using provider: {provider}")

    def _load_rknn(self):
        """Load RKNN model (simulation mode on PC)"""
        try:
            from rknn.api import RKNN
        except ImportError:
            raise SystemExit("rknn-toolkit2 not installed. pip install rknn-toolkit2")

        # For evaluation, we load ONNX and build RKNN for simulation
        logger.info("  Loading RKNN in simulation mode (PC)")

        rknn = RKNN(verbose=False)

        # Need to rebuild from ONNX for simulation
        onnx_path = self.model_path.with_suffix('.onnx')
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX source not found: {onnx_path}")

        logger.info(f"  Building from ONNX: {onnx_path}")

        # Config RKNN (required before load_onnx)
        ret = rknn.config(target_platform='rk3588')
        if ret != 0:
            raise RuntimeError("Failed to config RKNN")

        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError("Failed to load ONNX")

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError("Failed to build RKNN")

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime")

        self.model = rknn
        logger.info("  ✓ RKNN model ready (simulation mode)")

    def preprocess(self, image: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, Tuple]:
        """Preprocess image for inference"""
        from apps.utils.yolo_post import letterbox

        # Letterbox resize
        img, ratio, (dw, dh) = letterbox(image, target_size)

        if self.model_type == 'onnx':
            # NCHW, RGB, normalize to [0,1]
            img = img[:, :, ::-1]  # BGR -> RGB
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)  # Add batch dimension

        elif self.model_type == 'rknn':
            # NHWC, RGB, keep as uint8
            img = img[:, :, ::-1]  # BGR -> RGB
            img = np.expand_dims(img, axis=0)  # Add batch dimension

        return img, ratio, (dw, dh)

    def inference(self, img: np.ndarray) -> np.ndarray:
        """Run model inference"""
        if self.model_type == 'onnx':
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: img})
            return outputs[0]

        elif self.model_type == 'rknn':
            outputs = self.model.inference(inputs=[img], data_format='nhwc')
            return outputs[0]

    def postprocess(self, output: np.ndarray, orig_shape: Tuple,
                   ratio_pad: Tuple, person_class_id: int = 0) -> List[Dict]:
        """Postprocess predictions, filter for person class only"""
        from apps.yolov8_rknn_infer import decode_predictions

        boxes, confs, class_ids = decode_predictions(
            output, self.imgsz, self.conf_threshold, self.iou_threshold,
            ratio_pad=ratio_pad, orig_shape=orig_shape
        )

        # Filter for person class (COCO class 0)
        detections = []
        for box, conf, cls_id in zip(boxes, confs, class_ids):
            if int(cls_id) == person_class_id:
                detections.append({
                    'bbox': box.tolist(),
                    'score': float(conf),
                    'class': 'person'
                })

        return detections

    def detect_image(self, image_path: Path) -> List[Dict]:
        """Run detection on single image"""
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return []

        orig_shape = img.shape[:2]

        # Preprocess
        img_processed, ratio, pad = self.preprocess(img, target_size=self.imgsz)

        # Inference
        output = self.inference(img_processed)

        # Postprocess
        detections = self.postprocess(output, orig_shape, (ratio, pad))

        return detections


def load_coco_annotations(annotation_file: Path, person_class_id: int = 1) -> Dict:
    """Load COCO format annotations, filter for person class"""
    logger.info(f"Loading COCO annotations: {annotation_file}")

    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Filter for person category
    person_cat = None
    for cat in coco_data['categories']:
        if cat['id'] == person_class_id and cat['name'] == 'person':
            person_cat = cat
            break

    if not person_cat:
        logger.warning("Person category not found in annotations")

    # Build image id to annotations mapping
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] == person_class_id:
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

    logger.info(f"  Loaded {len(annotations_by_image)} images with person annotations")

    return {
        'annotations': annotations_by_image,
        'images': images_by_id
    }


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_map(predictions: Dict[str, List[Dict]],
                 ground_truths: Dict[str, List[Dict]],
                 iou_threshold: float = 0.5) -> Tuple[float, Dict]:
    """Calculate mAP@IoU_threshold for pedestrian detection"""
    logger.info(f"Calculating mAP@{iou_threshold}...")

    all_predictions = []
    all_ground_truths = []

    # Flatten predictions and ground truths
    for img_id in predictions:
        for pred in predictions[img_id]:
            all_predictions.append({
                'image_id': img_id,
                'bbox': pred['bbox'],
                'score': pred['score']
            })

    for img_id in ground_truths:
        for gt in ground_truths[img_id]:
            all_ground_truths.append({
                'image_id': img_id,
                'bbox': gt['bbox'],
                'iscrowd': gt.get('iscrowd', 0)
            })

    # Sort predictions by confidence score (descending)
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

        # Check if match exceeds threshold
        if best_iou >= iou_threshold and best_gt_idx >= 0:
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

    # Calculate average precision (AP)
    # Use 11-point interpolation
    ap = 0.0
    for recall_threshold in np.linspace(0, 1, 11):
        precisions_above_threshold = precisions[recalls >= recall_threshold]
        if len(precisions_above_threshold) > 0:
            ap += precisions_above_threshold.max()

    ap /= 11.0

    stats = {
        'ap': ap,
        'num_predictions': len(all_predictions),
        'num_ground_truths': num_ground_truths,
        'true_positives': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
        'false_positives': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0,
        'precision': float(precisions[-1]) if len(precisions) > 0 else 0.0,
        'recall': float(recalls[-1]) if len(recalls) > 0 else 0.0
    }

    logger.info(f"  mAP@{iou_threshold}: {ap:.4f} ({ap*100:.2f}%)")
    logger.info(f"  Precision: {stats['precision']:.4f}")
    logger.info(f"  Recall: {stats['recall']:.4f}")
    logger.info(f"  TP: {stats['true_positives']}, FP: {stats['false_positives']}")

    return ap, stats


def main():
    parser = argparse.ArgumentParser(description='Pedestrian Detection mAP Evaluator')
    parser.add_argument('--model', type=Path, required=True, help='Model file (.onnx or .rknn)')
    parser.add_argument('--model-type', type=str, choices=['onnx', 'rknn'], default='onnx')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset type or path')
    parser.add_argument('--annotations', type=Path, help='COCO format annotations JSON')
    parser.add_argument('--images-dir', type=Path, help='Directory containing images')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--map-iou', type=float, default=0.5, help='mAP IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size (416 or 640)')
    parser.add_argument('--output', type=Path, default=Path('artifacts/pedestrian_map_report.json'))
    parser.add_argument('--limit', type=int, help='Limit number of images to evaluate')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Pedestrian Detection mAP Evaluation")
    logger.info("="*60)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    evaluator = PedestrianEvaluator(
        args.model,
        model_type=args.model_type,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz
    )

    # Load ground truth annotations
    if args.annotations:
        gt_data = load_coco_annotations(args.annotations)
        images_by_id = gt_data['images']
        annotations_by_image = gt_data['annotations']
    else:
        raise ValueError("--annotations is required")

    # Determine images directory
    if args.images_dir:
        images_dir = args.images_dir
    else:
        images_dir = args.annotations.parent / 'images'

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    logger.info(f"Images directory: {images_dir}")

    # Run evaluation
    predictions_by_image = {}
    ground_truths_by_image = {}

    image_ids = list(annotations_by_image.keys())
    if args.limit:
        image_ids = image_ids[:args.limit]

    logger.info(f"\nEvaluating {len(image_ids)} images...")

    start_time = time.time()

    for idx, img_id in enumerate(image_ids):
        img_info = images_by_id[img_id]
        img_filename = img_info['file_name']
        img_path = images_dir / img_filename

        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        # Run detection
        detections = evaluator.detect_image(img_path)

        predictions_by_image[str(img_id)] = detections
        ground_truths_by_image[str(img_id)] = annotations_by_image[img_id]

        if (idx + 1) % 10 == 0:
            logger.info(f"  Processed {idx + 1}/{len(image_ids)} images...")

    elapsed_time = time.time() - start_time
    fps = len(image_ids) / elapsed_time

    logger.info(f"✓ Evaluation completed in {elapsed_time:.2f}s ({fps:.2f} FPS)")

    # Calculate mAP
    map_score, stats = calculate_map(
        predictions_by_image,
        ground_truths_by_image,
        iou_threshold=args.map_iou
    )

    # Generate report
    report = {
        'model': str(args.model),
        'model_type': args.model_type,
        'dataset': args.dataset,
        'num_images': len(image_ids),
        'conf_threshold': args.conf,
        'iou_threshold': args.iou,
        'map_iou_threshold': args.map_iou,
        'map': map_score,
        'map_percentage': map_score * 100,
        'statistics': stats,
        'performance': {
            'total_time_seconds': elapsed_time,
            'fps': fps
        },
        'graduation_requirement': {
            'threshold': 0.9,
            'achieved': map_score,
            'status': 'PASS' if map_score >= 0.9 else 'FAIL',
            'margin': (map_score - 0.9) * 100
        }
    }

    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n✓ Report saved: {args.output}")

    # Display summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Model: {args.model.name}")
    logger.info(f"mAP@{args.map_iou}: {map_score:.4f} ({map_score*100:.2f}%)")
    logger.info(f"Graduation Requirement (>= 90%): {report['graduation_requirement']['status']}")

    if map_score >= 0.9:
        logger.info(f"✓ PASS - Exceeds requirement by {report['graduation_requirement']['margin']:.2f}%")
        return 0
    else:
        logger.warning(f"✗ FAIL - Below requirement by {abs(report['graduation_requirement']['margin']):.2f}%")
        logger.info("\nRecommendations:")
        logger.info("  - Fine-tune model on pedestrian-specific dataset")
        logger.info("  - Adjust confidence threshold (current: {:.2f})".format(args.conf))
        logger.info("  - Increase training data diversity")
        return 1


if __name__ == '__main__':
    sys.exit(main())
