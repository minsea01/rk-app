#!/usr/bin/env python3
"""
RK3588板上性能基准测试脚本
测试推理延迟、FPS、NPU利用率等关键指标
"""

import argparse
import time
import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from apps.config import ModelConfig
from apps.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RK3588 Board Benchmark")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to RKNN model file")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Input image size (default: 416)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--core-mask", type=int, default=0x7,
                        help="NPU core mask (default: 0x7 for 3 cores)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed progress")
    return parser.parse_args()


def benchmark_rknn(model_path, imgsz, iterations, warmup, core_mask, verbose):
    """Run RKNN inference benchmark on board"""

    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        logger.error("RKNNLite not installed. Run: pip3 install rknn-toolkit-lite2")
        sys.exit(1)

    logger.info(f"Loading RKNN model: {model_path}")
    rknn = RKNNLite()

    # Load model
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        logger.error(f"Load RKNN model failed! ret={ret}")
        sys.exit(1)

    # Initialize runtime
    logger.info(f"Initializing runtime with core_mask=0x{core_mask:x}")
    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        logger.error(f"Init runtime failed! ret={ret}")
        sys.exit(1)

    # Create dummy input
    logger.info(f"Creating dummy input: ({imgsz}x{imgsz}x3)")
    dummy_input = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    logger.info(f"Warmup: {warmup} iterations...")
    for i in range(warmup):
        rknn.inference(inputs=[dummy_input])
        if verbose and (i + 1) % 5 == 0:
            logger.info(f"  Warmup {i+1}/{warmup}")

    # Benchmark
    logger.info(f"Benchmarking: {iterations} iterations...")
    latencies = []

    for i in range(iterations):
        start = time.perf_counter()
        outputs = rknn.inference(inputs=[dummy_input])
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        if verbose and (i + 1) % 10 == 0:
            logger.info(f"  Iteration {i+1}/{iterations}: {latency_ms:.2f}ms")

    # Release
    rknn.release()

    # Calculate statistics
    latencies = np.array(latencies)
    results = {
        "model": str(model_path),
        "input_size": imgsz,
        "iterations": iterations,
        "core_mask": f"0x{core_mask:x}",
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "std": float(np.std(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "fps": {
            "mean": float(1000.0 / np.mean(latencies)),
            "median": float(1000.0 / np.median(latencies)),
        },
        "throughput_imgs_per_sec": float(iterations / (np.sum(latencies) / 1000.0)),
    }

    return results


def print_results(results):
    """Print benchmark results in a readable format"""

    print("\n" + "="*60)
    print("RK3588 Board Benchmark Results")
    print("="*60)
    print(f"\nModel: {results['model']}")
    print(f"Input Size: {results['input_size']}x{results['input_size']}")
    print(f"NPU Core Mask: {results['core_mask']}")
    print(f"Iterations: {results['iterations']}")

    print(f"\n{'Latency (ms)':<20} {'Value':>10}")
    print("-"*32)
    for k, v in results['latency_ms'].items():
        print(f"{k:<20} {v:>10.2f}")

    print(f"\n{'FPS':<20} {'Value':>10}")
    print("-"*32)
    for k, v in results['fps'].items():
        print(f"{k:<20} {v:>10.2f}")

    print(f"\nThroughput: {results['throughput_imgs_per_sec']:.2f} images/sec")

    # Graduation requirements check
    print(f"\n{'Graduation Requirements':<40} {'Status':>10}")
    print("-"*52)

    fps_mean = results['fps']['mean']
    latency_mean = results['latency_ms']['mean']

    fps_ok = fps_mean >= 30
    latency_ok = latency_mean <= 45

    print(f"FPS ≥ 30: {fps_mean:.1f} FPS {'✅ PASS' if fps_ok else '❌ FAIL':>20}")
    print(f"Latency ≤ 45ms: {latency_mean:.1f}ms {'✅ PASS' if latency_ok else '❌ FAIL':>15}")

    print("="*60 + "\n")


def main():
    args = parse_args()

    # Check model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    # Run benchmark
    results = benchmark_rknn(
        model_path=args.model,
        imgsz=args.imgsz,
        iterations=args.iterations,
        warmup=args.warmup,
        core_mask=args.core_mask,
        verbose=args.verbose
    )

    # Print results
    print_results(results)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

    # Return exit code based on graduation requirements
    fps_mean = results['fps']['mean']
    latency_mean = results['latency_ms']['mean']

    if fps_mean >= 30 and latency_mean <= 45:
        logger.info("✅ All graduation requirements met!")
        sys.exit(0)
    else:
        logger.warning("⚠️  Some graduation requirements not met")
        sys.exit(1)


if __name__ == "__main__":
    main()
