#!/usr/bin/env python3
"""
Performance Profiling Suite for RK3588 Pedestrian Detection System

Features:
  - End-to-end latency measurement
  - Component-level profiling (preprocess, inference, postprocess)
  - Memory usage tracking
  - FPS benchmarking
  - Comparison across different configurations
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import tracemalloc

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiling for detection pipeline"""

    def __init__(self, model_path: Path, model_type: str = 'onnx', warmup_runs: int = 5):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.warmup_runs = warmup_runs
        self.model = None
        self.timings = {
            'preprocess': [],
            'inference': [],
            'postprocess': [],
            'end_to_end': []
        }
        self.memory_usage = []

        self._load_model()
        self._warmup()

    def _load_model(self):
        """Load detection model"""
        logger.info(f"Loading {self.model_type.upper()} model: {self.model_path}")

        if self.model_type == 'onnx':
            import onnxruntime as ort
            providers = []

            # Check for GPU providers
            available_providers = ort.get_available_providers()
            if 'TensorrtExecutionProvider' in available_providers:
                providers.append('TensorrtExecutionProvider')
                logger.info("  Using TensorRT provider")
            elif 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                logger.info("  Using CUDA provider")
            else:
                providers.append('CPUExecutionProvider')
                logger.info("  Using CPU provider")

            self.model = ort.InferenceSession(str(self.model_path), providers=providers)

        elif self.model_type == 'rknn':
            from rknn.api import RKNN

            rknn = RKNN(verbose=False)

            # Load ONNX for simulation
            onnx_path = self.model_path.with_suffix('.onnx')
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX source required: {onnx_path}")

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

    def _warmup(self):
        """Warmup model with dummy inputs"""
        logger.info(f"Warming up model ({self.warmup_runs} runs)...")

        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        for i in range(self.warmup_runs):
            self.profile_single_image(dummy, record=False)

        logger.info("✓ Warmup completed")

    def profile_single_image(self, image: np.ndarray, record: bool = True) -> Dict:
        """Profile single image detection"""
        from apps.utils.yolo_post import letterbox
        from apps.yolov8_rknn_infer import decode_predictions

        timings = {}

        # Start memory tracking
        if record:
            tracemalloc.start()

        # Preprocess
        t0 = time.perf_counter()
        img, ratio, pad = letterbox(image, 640)

        if self.model_type == 'onnx':
            img_processed = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            img_processed = np.expand_dims(img_processed, axis=0)
        elif self.model_type == 'rknn':
            img_processed = img[:, :, ::-1]
            img_processed = np.expand_dims(img_processed, axis=0)

        t1 = time.perf_counter()
        timings['preprocess'] = (t1 - t0) * 1000

        # Inference
        t2 = time.perf_counter()
        if self.model_type == 'onnx':
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: img_processed})
            pred = outputs[0]
        elif self.model_type == 'rknn':
            outputs = self.model.inference(inputs=[img_processed], data_format='nhwc')
            pred = outputs[0]
        t3 = time.perf_counter()
        timings['inference'] = (t3 - t2) * 1000

        # Postprocess
        t4 = time.perf_counter()
        boxes, confs, class_ids = decode_predictions(
            pred, 640, 0.25, 0.45,
            ratio_pad=(ratio, pad),
            orig_shape=image.shape[:2]
        )
        t5 = time.perf_counter()
        timings['postprocess'] = (t5 - t4) * 1000

        timings['end_to_end'] = timings['preprocess'] + timings['inference'] + timings['postprocess']

        # Record timings
        if record:
            for key in ['preprocess', 'inference', 'postprocess', 'end_to_end']:
                self.timings[key].append(timings[key])

            # Memory usage
            current, peak = tracemalloc.get_traced_memory()
            self.memory_usage.append(peak / 1024 / 1024)  # MB
            tracemalloc.stop()

        return {
            'timings': timings,
            'num_detections': len(boxes)
        }

    def profile_batch(self, images: List[np.ndarray]) -> Dict:
        """Profile batch of images"""
        logger.info(f"Profiling {len(images)} images...")

        for idx, img in enumerate(images):
            self.profile_single_image(img, record=True)

            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(images)}")

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate performance report"""
        report = {
            'model': str(self.model_path),
            'model_type': self.model_type,
            'num_samples': len(self.timings['end_to_end']),
            'performance': {}
        }

        # Calculate statistics for each component
        for component in ['preprocess', 'inference', 'postprocess', 'end_to_end']:
            timings = self.timings[component]

            if len(timings) > 0:
                report['performance'][component] = {
                    'mean_ms': float(np.mean(timings)),
                    'median_ms': float(np.median(timings)),
                    'min_ms': float(np.min(timings)),
                    'max_ms': float(np.max(timings)),
                    'std_ms': float(np.std(timings)),
                    'p95_ms': float(np.percentile(timings, 95)),
                    'p99_ms': float(np.percentile(timings, 99))
                }

        # FPS calculation
        if len(self.timings['end_to_end']) > 0:
            mean_latency = np.mean(self.timings['end_to_end'])
            report['fps'] = {
                'mean': 1000.0 / mean_latency if mean_latency > 0 else 0,
                'p95': 1000.0 / np.percentile(self.timings['end_to_end'], 95)
            }

        # Memory usage
        if len(self.memory_usage) > 0:
            report['memory'] = {
                'mean_mb': float(np.mean(self.memory_usage)),
                'peak_mb': float(np.max(self.memory_usage))
            }

        # Graduation requirements check
        report['graduation_requirements'] = {
            'latency_requirement_ms': 45,
            'fps_requirement': 30,
            'achieved_latency_ms': report['performance']['end_to_end']['mean_ms'],
            'achieved_fps': report['fps']['mean'],
            'latency_status': 'PASS' if report['performance']['end_to_end']['mean_ms'] <= 45 else 'FAIL',
            'fps_status': 'PASS' if report['fps']['mean'] >= 30 else 'FAIL'
        }

        return report


def load_test_images(images_dir: Path, limit: Optional[int] = None) -> List[np.ndarray]:
    """Load test images from directory"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(images_dir.glob(ext))

    if limit:
        image_files = image_files[:limit]

    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)

    return images


def main():
    parser = argparse.ArgumentParser(description='Performance Profiling Suite')
    parser.add_argument('--model', type=Path, required=True, help='Model file (.onnx or .rknn)')
    parser.add_argument('--model-type', type=str, choices=['onnx', 'rknn'], default='onnx')
    parser.add_argument('--images-dir', type=Path, required=True, help='Test images directory')
    parser.add_argument('--limit', type=int, default=100, help='Number of images to profile')
    parser.add_argument('--output', type=Path, default=Path('artifacts/performance_profile.json'))
    parser.add_argument('--warmup', type=int, default=5, help='Warmup runs')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Performance Profiling Suite")
    logger.info("="*60)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load test images
    logger.info(f"Loading test images from: {args.images_dir}")
    images = load_test_images(args.images_dir, limit=args.limit)
    logger.info(f"✓ Loaded {len(images)} images")

    if len(images) == 0:
        logger.error("No images found")
        return 1

    # Create profiler
    profiler = PerformanceProfiler(
        args.model,
        model_type=args.model_type,
        warmup_runs=args.warmup
    )

    # Run profiling
    report = profiler.profile_batch(images)

    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n✓ Report saved: {args.output}")

    # Display summary
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*60)
    logger.info(f"Model: {args.model.name}")
    logger.info(f"Samples: {report['num_samples']}")
    logger.info(f"\nLatency Breakdown:")
    logger.info(f"  Preprocess:  {report['performance']['preprocess']['mean_ms']:.2f} ms (± {report['performance']['preprocess']['std_ms']:.2f})")
    logger.info(f"  Inference:   {report['performance']['inference']['mean_ms']:.2f} ms (± {report['performance']['inference']['std_ms']:.2f})")
    logger.info(f"  Postprocess: {report['performance']['postprocess']['mean_ms']:.2f} ms (± {report['performance']['postprocess']['std_ms']:.2f})")
    logger.info(f"  End-to-End:  {report['performance']['end_to_end']['mean_ms']:.2f} ms (± {report['performance']['end_to_end']['std_ms']:.2f})")
    logger.info(f"\nThroughput:")
    logger.info(f"  Mean FPS: {report['fps']['mean']:.2f}")
    logger.info(f"  P95 FPS:  {report['fps']['p95']:.2f}")
    logger.info(f"\nMemory:")
    logger.info(f"  Mean: {report['memory']['mean_mb']:.2f} MB")
    logger.info(f"  Peak: {report['memory']['peak_mb']:.2f} MB")
    logger.info(f"\nGraduation Requirements:")
    logger.info(f"  Latency (≤45ms): {report['graduation_requirements']['latency_status']}")
    logger.info(f"  FPS (≥30):       {report['graduation_requirements']['fps_status']}")

    # Return exit code based on requirements
    if (report['graduation_requirements']['latency_status'] == 'PASS' and
        report['graduation_requirements']['fps_status'] == 'PASS'):
        logger.info("\n✓ All performance requirements met!")
        return 0
    else:
        logger.warning("\n⚠ Some performance requirements not met")
        return 1


if __name__ == '__main__':
    sys.exit(main())
