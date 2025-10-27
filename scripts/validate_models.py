#!/usr/bin/env python3
"""Compare ONNX and RKNN models on a sample image (PC simulation).

Usage:
  python scripts/validate_models.py \
      --onnx artifacts/models/yolo11n.onnx \
      --rknn artifacts/models/yolo11n_int8.rknn \
      --image assets/test.jpg --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _load_image(path: Path, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise SystemExit(f"Failed to read image: {path}")
    img_resized = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    # ONNX (Ultralytics) expects float32 RGB normalized to 0-1
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    onnx_input = img_rgb.astype(np.float32) / 255.0
    onnx_input = np.transpose(onnx_input, (2, 0, 1))[None, ...]  # (1,3,H,W)

    # RKNN (current pipeline) ingests uint8 BGR with mean=0, std=255
    rknn_input = img_resized[None, ...]  # (1,H,W,3) uint8

    return onnx_input, rknn_input


def _run_onnx(onnx_path: Path, tensor: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(f"onnxruntime not installed. pip install onnxruntime. Error: {exc}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: tensor})
    return outputs[0]


def _run_rknn(rknn_path: Path, tensor: np.ndarray) -> np.ndarray:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise SystemExit(f"rknn-toolkit2 not installed. pip install rknn-toolkit2. Error: {exc}")

    rknn = RKNN(verbose=False)
    if rknn.load_rknn(str(rknn_path)) != 0:
        raise SystemExit("load_rknn failed")
    if rknn.init_runtime(target="rk3588", device_id=None) != 0:
        # fallback to CPU simulation
        if rknn.init_runtime() != 0:
            raise SystemExit("init_runtime failed")

    outputs = rknn.inference(inputs=[tensor])
    rknn.release()
    return np.array(outputs[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ONNX and RKNN outputs on a sample image")
    ap.add_argument("--onnx", type=Path, required=True, help="Path to exported ONNX model")
    ap.add_argument("--rknn", type=Path, required=True, help="Path to RKNN model")
    ap.add_argument("--image", type=Path, required=True, help="Test image")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size (square)")
    args = ap.parse_args()

    onnx_tensor, rknn_tensor = _load_image(args.image, args.imgsz)
    print(f"Loaded image {args.image} -> tensors: ONNX {onnx_tensor.shape}, RKNN {rknn_tensor.shape}")

    onnx_out = _run_onnx(args.onnx, onnx_tensor)
    print(f"ONNX output shape: {onnx_out.shape}, dtype={onnx_out.dtype}")

    rknn_out = _run_rknn(args.rknn, rknn_tensor)
    print(f"RKNN output shape: {rknn_out.shape}, dtype={rknn_out.dtype}")

    if rknn_out.dtype != np.float32:
        rknn_out = rknn_out.astype(np.float32)

    # Align shapes if RKNN drops batch dimension
    if onnx_out.shape != rknn_out.shape:
        rknn_out = rknn_out.reshape(onnx_out.shape)

    diff = onnx_out - rknn_out
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))

    print("=== Comparison Metrics ===")
    print(f"Max absolute error : {max_abs:.6f}")
    print(f"Mean absolute error: {mean_abs:.6f}")
    print(f"RMS error          : {rms:.6f}")

    # Optional: show top-k elements for quick sanity check
    flat_onnx = onnx_out.flatten()
    flat_rknn = rknn_out.flatten()
    top_idx = np.argsort(flat_onnx)[-5:][::-1]
    print("Top-5 ONNX logits vs RKNN logits:")
    for idx in top_idx:
        print(f"  idx {idx:>5}: onnx={flat_onnx[idx]:>8.4f} | rknn={flat_rknn[idx]:>8.4f} | diff={flat_onnx[idx]-flat_rknn[idx]:>8.4f}")


if __name__ == "__main__":
    main()

