#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import sys
import numpy as np
import cv2


def make_input(img_path: Path, size: int) -> np.ndarray:
    img0 = None
    path_str = str(img_path) if img_path is not None else ""
    if path_str:
        img0 = cv2.imread(path_str)

    synthetic = img0 is None
    if synthetic:
        # synthetic image
        rng = np.random.RandomState(0)
        img0 = (rng.rand(size, size, 3) * 255).astype(np.uint8)
        if hasattr(cv2, "putText"):
            font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
            cv2.putText(img0, "SYNTH", (10, size // 2), font, 2.0, (0, 255, 0), 3)

    if img0.ndim == 2:
        img0 = np.stack([img0] * 3, axis=-1)

    needs_resize = not synthetic and (img0.shape[0] != size or img0.shape[1] != size)
    if needs_resize and hasattr(cv2, "resize"):
        img = cv2.resize(img0, (size, size), interpolation=getattr(cv2, "INTER_LINEAR", 1))
    else:
        if img0.shape[0] != size or img0.shape[1] != size:
            # simple nearest-neighbor fallback using numpy resize
            img = np.resize(img0, (size, size, img0.shape[2]))
        else:
            img = img0

    x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # NCHW
    return x


def main():
    ap = argparse.ArgumentParser(description="ONNXRuntime 基准测试（纯软件仿真）")
    ap.add_argument("--onnx", type=Path, required=True, help="ONNX 模型路径")
    ap.add_argument("--img", type=Path, default=Path(""), help="测试图片（留空用合成图）")
    ap.add_argument("--imgsz", type=int, default=640, help="输入尺寸（正方形）")
    ap.add_argument("--loops", type=int, default=100, help="统计循环次数")
    ap.add_argument("--warmup", type=int, default=10, help="预热次数")
    ap.add_argument(
        "--provider", type=str, default="CPUExecutionProvider", help="ONNXRuntime Provider"
    )
    args = ap.parse_args()

    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    try:
        import onnxruntime as ort
    except ImportError as e:
        print("请先安装 onnxruntime，例如: pip install onnxruntime")
        print("错误:", e)
        sys.exit(1)

    x = make_input(args.img, args.imgsz)

    sess = ort.InferenceSession(str(onnx_path), providers=[args.provider])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # warmup
    for _ in range(max(0, args.warmup)):
        _ = sess.run([out_name], {in_name: x})

    times = []
    for _ in range(max(1, args.loops)):
        t0 = time.perf_counter()
        _ = sess.run([out_name], {in_name: x})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times, dtype=np.float64)
    avg = float(times.mean())
    p50 = float(np.percentile(times, 50))
    p90 = float(np.percentile(times, 90))
    fps = 1000.0 / avg if avg > 0 else 0.0

    print("=== ONNXRuntime 基准（纯软件）===")
    print(f"model={onnx_path.name}")
    print(f"provider={args.provider}")
    print(f"imgsz={args.imgsz}  loops={args.loops}  warmup={args.warmup}")
    print(f"avg_ms={avg:.2f}  p50_ms={p50:.2f}  p90_ms={p90:.2f}  fps_avg={fps:.2f}")


if __name__ == "__main__":
    main()
