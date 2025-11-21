#!/usr/bin/env python3
"""
Lightweight ONNX latency/throughput micro-bench.
- Loads a single image, resizes to imgsz, runs warmup then timed runs.
- Supports provider selection (CPU/CUDA) without altering runtime.

Usage:
  python tools/bench_onnx_latency.py --model artifacts/models/best.onnx \
      --image assets/test.jpg --imgsz 640 --runs 50 --warmup 5 \
      --providers cpu,cuda --out artifacts/bench_summary.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort


def parse_providers(raw: str) -> List[str]:
    if not raw:
        return ["CPUExecutionProvider"]
    mapping = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    providers = []
    for token in raw.split(","):
        t = token.strip().lower()
        if t in mapping:
            providers.append(mapping[t])
        else:
            providers.append(token)
    return providers


def load_image(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Failed to read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img


def benchmark(session: ort.InferenceSession, input_name: str, data: np.ndarray, runs: int, warmup: int):
    for _ in range(max(0, warmup)):
        session.run(None, {input_name: data})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: data})
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    ap = argparse.ArgumentParser(description="ONNX latency benchmark helper")
    ap.add_argument("--model", type=Path, required=True, help="Path to ONNX model")
    ap.add_argument("--image", type=Path, default=Path("assets/test.jpg"), help="Input image for the benchmark")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    ap.add_argument("--runs", type=int, default=30, help="Timed runs")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    ap.add_argument("--providers", type=str, default="cpu", help="Comma-separated providers, e.g., cpu,cuda")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    providers = parse_providers(args.providers)

    print(f"[bench] loading model: {args.model}")
    print(f"[bench] providers: {providers}")
    so = ort.SessionOptions()
    session = ort.InferenceSession(str(args.model), providers=providers, sess_options=so)
    input_name = session.get_inputs()[0].name

    data = load_image(args.image, args.imgsz)
    times = benchmark(session, input_name, data, runs=args.runs, warmup=args.warmup)
    if not times:
        raise SystemExit("No timing collected")

    stats = {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p90_ms": np.percentile(times, 90).item(),
        "p95_ms": np.percentile(times, 95).item(),
        "min_ms": min(times),
        "max_ms": max(times),
        "fps_mean": 1000.0 / statistics.mean(times),
        "fps_p95": 1000.0 / np.percentile(times, 95),
    }

    result = {
        "model": str(args.model),
        "image": str(args.image),
        "imgsz": args.imgsz,
        "providers": session.get_providers(),
        "runs": args.runs,
        "warmup": args.warmup,
        "metrics": stats,
    }

    print("[bench] results:")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"[bench] saved: {args.out}")


if __name__ == "__main__":
    main()
