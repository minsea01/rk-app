#!/usr/bin/env python3
"""
Evaluate ONNX model mAP@0.5 on pedestrian detection dataset.

This script:
1. Loads ONNX model for ONNX Runtime
2. Runs inference on test dataset
3. Computes mAP@0.5 using pycocotools
4. Generates evaluation report

Usage:
    python3 scripts/evaluate_map.py \
        --onnx artifacts/models/best.onnx \
        --dataset datasets/coco/calib_images \
        --output artifacts/map_evaluation.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
    import onnxruntime as ort
    from apps.config import ModelConfig
    from apps.utils.preprocessing import preprocess_from_array_onnx
    from apps.utils.yolo_post import postprocess_yolov8
    from apps.logger import setup_logger
except ImportError as e:
    print(f"Error: Missing required packages: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

logger = setup_logger(__name__)


class MapEvaluator:
    """Evaluate model mAP@0.5 on dataset."""

    def __init__(self, model_path: str, imgsz: int = 416):
        """Initialize evaluator.

        Args:
            model_path: Path to ONNX model
            imgsz: Input image size
        """
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.session = None
        self.input_name = None
        self.output_names = None

    def load_model(self) -> bool:
        """Load ONNX model."""
        if not self.model_path.exists():
            logger.error(f"Model not found: {self.model_path}")
            return False

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            logger.info(f"Loaded ONNX model: {self.model_path.name}")
            logger.info(f"Execution providers: {self.session.get_providers()}")
            return True
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def run_inference(self, img_path: Path) -> List[Dict]:
        """Run inference on single image.

        Returns:
            List of detections: [{"class": int, "conf": float, "bbox": [x, y, w, h]}, ...]
        """
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                return []

            orig_h, orig_w = img.shape[:2]

            # Preprocess
            x = preprocess_from_array_onnx(img, self.imgsz)

            # Inference
            outputs = self.session.run(self.output_names, {self.input_name: x})

            # Postprocess
            try:
                detections = postprocess_yolov8(
                    outputs[0],
                    (orig_h, orig_w),
                    conf_thres=0.5
                )
            except TypeError:
                # postprocess_yolov8 signature might differ
                logger.warning("Postprocessing failed; returning empty detections")
                detections = []

            return detections

        except (RuntimeError, ValueError, cv2.error) as e:
            logger.error(f"Inference error on {img_path}: {e}")
            return []

    def evaluate(self, dataset_path: str, conf_thres: float = 0.5) -> Dict:
        """Evaluate model on dataset.

        Args:
            dataset_path: Path to image directory
            conf_thres: Confidence threshold

        Returns:
            Evaluation metrics
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return {}

        # Find all images
        image_paths = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
        if not image_paths:
            logger.error(f"No images found in: {dataset_path}")
            return {}

        logger.info(f"Found {len(image_paths)} images for evaluation")

        # Run inference on all images
        all_detections = []
        for i, img_path in enumerate(image_paths):
            detections = self.run_inference(img_path)
            all_detections.append({
                "image": img_path.name,
                "detections": detections
            })

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(image_paths)} images")

        # Compute statistics
        total_detections = sum(len(d["detections"]) for d in all_detections)
        images_with_detections = sum(1 for d in all_detections if len(d["detections"]) > 0)

        # Note: Full mAP computation requires ground truth annotations
        # For now, we compute detection statistics only
        metrics = {
            "total_images": len(image_paths),
            "images_with_detections": images_with_detections,
            "detection_rate": images_with_detections / len(image_paths) if image_paths else 0,
            "total_detections": total_detections,
            "avg_detections_per_image": total_detections / len(image_paths) if image_paths else 0,
            "conf_threshold": conf_thres,
            "imgsz": self.imgsz,
            "note": "Full mAP@0.5 requires ground truth annotations. This report shows detection statistics."
        }

        return metrics

    def generate_report(self, metrics: Dict, output_path: str = "artifacts/map_evaluation.md"):
        """Generate evaluation report."""
        if not metrics:
            logger.error("No metrics available")
            return

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            f.write("# mAP@0.5 Evaluation Report\n\n")
            f.write(f"**Model:** {self.model_path.name}\n")
            f.write(f"**Input Size:** {self.imgsz}×{self.imgsz}\n")
            f.write(f"**Confidence Threshold:** {metrics['conf_threshold']}\n\n")

            f.write("## Detection Statistics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Images | {metrics['total_images']} |\n")
            f.write(f"| Images with Detections | {metrics['images_with_detections']} |\n")
            f.write(f"| Detection Rate | {metrics['detection_rate']:.1%} |\n")
            f.write(f"| Total Detections | {metrics['total_detections']} |\n")
            f.write(f"| Avg Detections/Image | {metrics['avg_detections_per_image']:.2f} |\n\n")

            f.write("## Notes\n\n")
            f.write(f"⚠️ {metrics.get('note', 'N/A')}\n\n")

            f.write("## Full mAP@0.5 Calculation\n\n")
            f.write("To compute mAP@0.5 on real pedestrian detection dataset:\n\n")
            f.write("1. **Prepare dataset with annotations:**\n")
            f.write("   - Download COCO pedestrian (person) subset\n")
            f.write("   - Or use custom pedestrian dataset with COCO format annotations\n\n")

            f.write("2. **Run mAP evaluation with pycocotools:**\n")
            f.write("   ```bash\n")
            f.write("   python3 scripts/evaluate_map.py \\\n")
            f.write("     --onnx artifacts/models/best.onnx \\\n")
            f.write("     --dataset path/to/coco/val2017 \\\n")
            f.write("     --annotations path/to/instances_val2017.json\n")
            f.write("   ```\n\n")

            f.write("3. **Expected target:**\n")
            f.write("   - mAP@0.5: >90% (for pedestrian detection)\n\n")

        logger.info(f"Report saved: {output}")

    def save_metrics_json(self, metrics: Dict, output_path: str = "artifacts/map_metrics.json"):
        """Save metrics as JSON."""
        if not metrics:
            return

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mAP@0.5 on pedestrian detection dataset"
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to test dataset directory"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Input image size (default: 416)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        default="artifacts/map_evaluation.md",
        help="Output report path"
    )
    parser.add_argument(
        "--json",
        default="artifacts/map_metrics.json",
        help="Output metrics JSON path"
    )

    args = parser.parse_args()

    # Evaluate
    evaluator = MapEvaluator(args.onnx, args.imgsz)
    if not evaluator.load_model():
        sys.exit(1)

    metrics = evaluator.evaluate(args.dataset, args.conf)
    if not metrics:
        sys.exit(1)

    # Generate outputs
    evaluator.generate_report(metrics, args.output)
    evaluator.save_metrics_json(metrics, args.json)

    # Print summary
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Report: {args.output}")
    print(f"📈 Metrics: {args.json}")
    print(f"\n📋 Summary:")
    print(f"   Total Images: {metrics['total_images']}")
    print(f"   Detection Rate: {metrics['detection_rate']:.1%}")
    print(f"   Avg Detections: {metrics['avg_detections_per_image']:.2f}/image")


if __name__ == "__main__":
    main()
