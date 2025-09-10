#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build RKNN INT8 model from ONNX for RK3588")
    ap.add_argument("--onnx", default="best.onnx", help="Path to ONNX model")
    ap.add_argument("--dataset", default="dataset.txt", help="Calibration image list (one path per line)")
    ap.add_argument("--out", default="best_int8.rknn", help="Output RKNN model path")
    ap.add_argument("--target", default="rk3588", help="Target platform (rk3588)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    dataset_path = Path(args.dataset).resolve()
    out_path = Path(args.out).resolve()

    if not onnx_path.exists():
        print(f"[ERROR] ONNX not found: {onnx_path}")
        return 2
    if not dataset_path.exists():
        print(f"[WARN] dataset.txt not found: {dataset_path} (quantization will likely fail)")

    try:
        from rknn.api import RKNN  # type: ignore
    except Exception as exc:
        print("[ERROR] rknn-toolkit2 not installed. See: https://github.com/airockchip/rknn-toolkit2")
        print(f"{exc}")
        return 2

    rknn = RKNN(verbose=False)
    print("--> Config")
    rknn.config(
        target_platform=args.target,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype='asymmetric_quantized-8',
        optimization_level=3,
    )

    print(f"--> Load ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print("[ERROR] load_onnx failed")
        return 3

    print("--> Build (INT8 quant)")
    ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
    if ret != 0:
        print("[ERROR] build failed")
        return 4

    print(f"--> Export: {out_path}")
    ret = rknn.export_rknn(str(out_path))
    if ret != 0:
        print("[ERROR] export_rknn failed")
        return 5

    print("[OK] RKNN saved:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


