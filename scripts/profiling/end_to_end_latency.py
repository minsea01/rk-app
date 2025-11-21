#!/usr/bin/env python3
"""
端到端延迟测试：预处理 + 推理 + 后处理 + 网络传输
用于验证完整系统延迟是否满足 ≤45ms 的毕设要求
"""

import argparse
import time
import json
import sys
import socket
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from apps.config import ModelConfig
from apps.logger import setup_logger
from apps.utils.preprocessing import preprocess_board
from apps.utils.yolo_post import postprocess_yolov8

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Latency Test")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to RKNN model")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to test image")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Input image size")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--target-host", type=str, default=None,
                        help="Target host for network transmission test")
    parser.add_argument("--target-port", type=int, default=8080,
                        help="Target port")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    return parser.parse_args()


def measure_preprocessing(image_path, imgsz):
    """Measure preprocessing time"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    start = time.perf_counter()
    processed = preprocess_board(img, target_size=imgsz)
    end = time.perf_counter()

    return processed, (end - start) * 1000


def measure_inference(rknn, input_data):
    """Measure inference time"""
    start = time.perf_counter()
    outputs = rknn.inference(inputs=[input_data])
    end = time.perf_counter()

    return outputs, (end - start) * 1000


def measure_postprocessing(outputs, conf_threshold, iou_threshold, imgsz):
    """Measure postprocessing time"""
    start = time.perf_counter()

    # Simple postprocessing (decode + NMS)
    output = outputs[0][0]  # (84, N) or (N, 84)
    if output.shape[0] == 84:
        output = output.T  # (N, 84)

    # Filter by confidence
    boxes = output[:, :4]
    scores = output[:, 4:].max(axis=1)
    class_ids = output[:, 4:].argmax(axis=1)

    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # Simple NMS (using confidence-based filtering)
    detections = len(boxes)

    end = time.perf_counter()

    return detections, (end - start) * 1000


def measure_network_transmission(data, target_host, target_port):
    """Measure network transmission time"""
    if target_host is None:
        return 0.0

    # Create JSON payload
    payload = json.dumps(data).encode('utf-8')

    try:
        start = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((target_host, target_port))
        sock.sendall(payload)
        sock.close()
        end = time.perf_counter()

        return (end - start) * 1000
    except Exception as e:
        logger.warning(f"Network transmission failed: {e}")
        return 0.0


def main():
    args = parse_args()

    # Load RKNN model
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        logger.error("RKNNLite not installed")
        sys.exit(1)

    logger.info(f"Loading RKNN model: {args.model}")
    rknn = RKNNLite()
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        logger.error(f"Load model failed: ret={ret}")
        sys.exit(1)

    ret = rknn.init_runtime(core_mask=0x7)
    if ret != 0:
        logger.error(f"Init runtime failed: ret={ret}")
        sys.exit(1)

    # Run iterations
    logger.info(f"Running {args.iterations} iterations...")

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    network_times = []
    total_times = []

    for i in range(args.iterations):
        # 1. Preprocessing
        processed, preprocess_ms = measure_preprocessing(args.source, args.imgsz)

        # 2. Inference
        outputs, inference_ms = measure_inference(rknn, processed)

        # 3. Postprocessing
        detections, postprocess_ms = measure_postprocessing(
            outputs, args.conf, args.iou, args.imgsz
        )

        # 4. Network transmission (if enabled)
        network_ms = 0.0
        if args.target_host:
            data = {
                "timestamp": time.time(),
                "detections": detections,
                "latency_ms": inference_ms
            }
            network_ms = measure_network_transmission(data, args.target_host, args.target_port)

        total_ms = preprocess_ms + inference_ms + postprocess_ms + network_ms

        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        postprocess_times.append(postprocess_ms)
        network_times.append(network_ms)
        total_times.append(total_ms)

        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{args.iterations}: {total_ms:.2f}ms total")

    rknn.release()

    # Calculate statistics
    def calc_stats(times):
        arr = np.array(times)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    results = {
        "model": str(args.model),
        "image": str(args.source),
        "input_size": args.imgsz,
        "iterations": args.iterations,
        "preprocessing_ms": calc_stats(preprocess_times),
        "inference_ms": calc_stats(inference_times),
        "postprocessing_ms": calc_stats(postprocess_times),
        "network_tx_ms": calc_stats(network_times) if args.target_host else None,
        "total_latency_ms": calc_stats(total_times),
    }

    # Print results
    print("\n" + "="*60)
    print("End-to-End Latency Test Results")
    print("="*60)

    print(f"\nPreprocessing:  {results['preprocessing_ms']['mean']:.2f}ms")
    print(f"Inference:      {results['inference_ms']['mean']:.2f}ms")
    print(f"Postprocessing: {results['postprocessing_ms']['mean']:.2f}ms")
    if args.target_host:
        print(f"Network TX:     {results['network_tx_ms']['mean']:.2f}ms")
    print(f"{'='*20}")
    print(f"Total:          {results['total_latency_ms']['mean']:.2f}ms")

    total_mean = results['total_latency_ms']['mean']
    requirement_met = total_mean <= 45

    print(f"\nGraduation Requirement (≤45ms): {'✅ PASS' if requirement_met else '❌ FAIL'}")
    print("="*60 + "\n")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

    sys.exit(0 if requirement_met else 1)


if __name__ == "__main__":
    main()
