import argparse
import os
import time
import shutil
from glob import glob
from typing import List, Tuple
from pathlib import Path

import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = REPO_ROOT / "runs" / "export" / "onnx_dynamic_fp16"


def run_ort(model_path: str, size: int, batch: int, warmup: int, runs: int, prefer_trt: bool) -> Tuple[float, float, List[str], str]:
    providers = [
        ('TensorrtExecutionProvider', {'trt_fp16_enable': True}),
        'CUDAExecutionProvider',
    ] if prefer_trt else ['CUDAExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    inp = session.get_inputs()[0]
    name = inp.name
    dtype = np.float16 if '16' in inp.type else np.float32
    x = np.random.rand(batch, 3, size, size).astype(dtype)

    for _ in range(warmup):
        session.run(None, {name: x})

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        session.run(None, {name: x})
    torch.cuda.synchronize()
    dt = (time.time() - t0) / runs
    latency_ms = dt * 1000.0
    fps = batch / dt
    return latency_ms, fps, session.get_providers(), str(dtype)


def find_latest_best_onnx(search_root: str) -> str:
    candidates = glob(os.path.join(search_root, "**", "weights", "best.onnx"), recursive=True)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def export_static_onnx(pt_path: str, size: int, batch: int, half: bool, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, 'best.onnx')
    if os.path.isfile(target_path):
        return target_path

    model = YOLO(pt_path)
    # Ultralytics saves ONNX next to the .pt weights by default
    model.export(format='onnx', half=half, simplify=True, device=0, imgsz=size, batch=batch)

    # Pick up the just-exported file
    pt_dir = os.path.dirname(pt_path)
    candidate = os.path.join(pt_dir, 'best.onnx')
    saved = candidate if os.path.isfile(candidate) else find_latest_best_onnx(str(REPO_ROOT))
    if not saved or not os.path.isfile(saved):
        raise FileNotFoundError(f'Exported ONNX not found. looked_for={candidate}')
    try:
        shutil.copy2(saved, target_path)
    except (IOError, OSError, PermissionError) as e:
        # Fallback to using the original path
        logger.warning(f"Failed to copy ONNX file: {e}, using original path")
        target_path = saved
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Export dynamic FP16 ONNX and benchmark with ORT.')
    parser.add_argument('--pt', required=True, help='Path to .pt model')
    parser.add_argument('--out_dir', default=str(DEFAULT_EXPORT_DIR), help='Export output dir')
    parser.add_argument('--sizes', default='640,1536', help='Comma-separated sizes to test')
    parser.add_argument('--batches', default='1,4', help='Comma-separated batches to test')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--trt', action='store_true', help='Prefer TensorRT EP if available')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Export and Bench for each size×batch as static ONNX (FP16, fallback to FP32 on failure)
    sizes = [int(s.strip()) for s in args.sizes.split(',') if s.strip()]
    batches = [int(b.strip()) for b in args.batches.split(',') if b.strip()]
    for s in sizes:
        for b in batches:
            subdir = os.path.join(args.out_dir, f'onxx_{s}_b{b}_fp16')
            # Export FP16 static
            try:
                onnx_path = export_static_onnx(args.pt, s, b, True, subdir)
                lat, fps, prov, dtype = run_ort(onnx_path, s, b, args.warmup, args.runs, args.trt)
                print(f'size={s} batch={b} latency_ms={lat:.3f} fps={fps:.2f} providers={prov} dtype={dtype} path={onnx_path}')
                continue
            except (RuntimeError, ValueError, FileNotFoundError) as e_fp16:
                print(f'size={s} batch={b} FP16 export/run failed: {e_fp16} -> falling back to FP32')
            # Fallback to FP32
            subdir32 = os.path.join(args.out_dir, f'onxx_{s}_b{b}_fp32')
            onnx_path32 = export_static_onnx(args.pt, s, b, False, subdir32)
            lat, fps, prov, dtype = run_ort(onnx_path32, s, b, args.warmup, args.runs, args.trt)
            print(f'size={s} batch={b} latency_ms={lat:.3f} fps={fps:.2f} providers={prov} dtype={dtype} path={onnx_path32}')


if __name__ == '__main__':
    main()


