#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX vs RKNN(sim) equivalence comparison tool.

This script compares inference outputs between ONNX Runtime and RKNN simulator
to validate model conversion accuracy.
"""
import argparse
import time
import logging
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import cv2
import onnxruntime as ort
from rknn.api import RKNN

# Ensure repo root on sys.path when run as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from apps.exceptions import ModelLoadError, ConfigurationError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def _make_synth(size=640):
    rnd = np.random.RandomState(0)
    im = (rnd.rand(size, size, 3) * 255).astype(np.uint8)
    cv2.rectangle(im, (size//4, size//4), (size//2, size//2), (0, 255, 0), 2)
    cv2.circle(im, (int(size*0.75), int(size*0.25)), size//10, (255, 0, 0), 3)
    return im


def preprocess(img_path: Path, size=640):
    im = None
    if img_path is not None and str(img_path):
        im = cv2.imread(str(img_path))
    if im is None:
        im = _make_synth(size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (im.astype(np.float32) / 255.).transpose(2, 0, 1)[None]  # NCHW
    return x

def to_CA(y):
    y = np.array(y)
    # (1, A, C) -> (1, C, A)
    if y.ndim == 3 and y.shape[-1] < y.shape[1]:
        y = np.transpose(y, (0, 2, 1))
    return y

def sigmoid(x): return 1. / (1. + np.exp(-x))

def decode_plain(y, num_classes=0, conf_thres=0.1, apply_sigmoid=True):
    """
    非 DFL 头：(1, C, A) with C = 4 + 1 + num_classes
    返回 Nx6: x1,y1,x2,y2,score,cls
    """
    y = y[0]
    bx, by, bw, bh = y[0], y[1], y[2], y[3]
    obj = y[4]
    cls = y[5:5 + num_classes] if num_classes and num_classes > 0 else y[5:]
    if apply_sigmoid:
        obj = sigmoid(obj)
        cls = sigmoid(cls)
    conf = obj * cls.max(axis=0)
    sel = conf > conf_thres
    if not np.any(sel):
        return np.empty((0, 6), dtype=np.float32)
    x1 = bx[sel] - bw[sel] / 2
    y1 = by[sel] - bh[sel] / 2
    x2 = bx[sel] + bw[sel] / 2
    y2 = by[sel] + bh[sel] / 2
    cid = cls[:, sel].argmax(axis=0).astype(np.float32)
    return np.stack([x1, y1, x2, y2, conf[sel].astype(np.float32), cid], axis=1)

def nms(boxes, iou_thres=0.7):
    if boxes.shape[0] == 0: return boxes
    x1, y1, x2, y2, s, c = [boxes[:, i] for i in range(6)]
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        inter = w * h
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_j = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        iou = inter / (area_i + area_j - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return boxes[keep]

def iou_matrix(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    x11, y11, x12, y12 = [A[:, i][:, None] for i in range(4)]
    x21, y21, x22, y22 = [B[:, i][None, :] for i in range(4)]
    xx1 = np.maximum(x11, x21); yy1 = np.maximum(y11, y21)
    xx2 = np.minimum(x12, x22); yy2 = np.minimum(y12, y22)
    w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
    inter = w * h
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    return inter / (area1 + area2 - inter + 1e-9)

def run_onnx(onnx_path: Path, inp):
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    t0 = time.time()
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    t1 = time.time()
    return to_CA(out), (t1 - t0) * 1000

def run_rknn_from_onnx(
    onnx_path: Path,
    inp: np.ndarray,
    imgsz: int,
    quant: bool = False,
    calib_dir: Optional[Path] = None
) -> Tuple[np.ndarray, float]:
    """Run inference using RKNN simulator built from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model file
        inp: Input tensor in NCHW format
        imgsz: Input image size
        quant: Whether to apply INT8 quantization
        calib_dir: Calibration images directory (required if quant=True)
        
    Returns:
        Tuple of (output_tensor, inference_time_ms)
        
    Raises:
        ConfigurationError: If RKNN configuration fails
        ModelLoadError: If model loading or building fails
    """
    rk = RKNN(verbose=False)
    
    try:
        # Configure for RK3588
        if quant:
            ret = rk.config(
                target_platform='rk3588',
                optimization_level=3,
                quantized_dtype='w8a8'  # RKNN-Toolkit2 2.x format
            )
        else:
            ret = rk.config(target_platform='rk3588', optimization_level=3)
        
        if ret != 0:
            raise ConfigurationError(f"RKNN config failed with code {ret}")
        
        ret = rk.load_onnx(str(onnx_path), input_size_list=[[1, 3, imgsz, imgsz]])
        if ret != 0:
            raise ModelLoadError(f"Failed to load ONNX model: {onnx_path}")
        
        if quant:
            if calib_dir is None or not calib_dir.exists():
                raise ConfigurationError(
                    f"INT8 quantization requires valid --calib-dir, got: {calib_dir}"
                )
            # Collect up to 200 calibration images
            imgs = []
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                imgs.extend(sorted(calib_dir.glob(ext)))
            imgs = [str(p) for p in imgs[:200]]
            
            if not imgs:
                raise ConfigurationError(f"No calibration images found in {calib_dir}")
            
            tmp = Path('/tmp/rknn_ds.txt')
            tmp.write_text('\n'.join(imgs))
            ret = rk.build(do_quantization=True, dataset=str(tmp))
        else:
            ret = rk.build(do_quantization=False)
        
        if ret != 0:
            raise ModelLoadError(f"RKNN build failed with code {ret}")
        
        ret = rk.init_runtime()
        if ret != 0:
            raise ModelLoadError(f"RKNN runtime init failed with code {ret}")
        
        t0 = time.time()
        out = rk.inference([inp], data_format='nchw')[0]
        t1 = time.time()
        
        return to_CA(out), (t1 - t0) * 1000
    finally:
        rk.release()


def main():
    ap = argparse.ArgumentParser(description='ONNX vs RKNN(sim) 等价性对比')
    ap.add_argument('--onnx', required=True, help='ONNX 路径')
    ap.add_argument('--img',  required=False, default='', help='测试图片路径（留空则用合成图）')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--num-classes', type=int, default=0, help='分类数，0则自动推断')
    ap.add_argument('--conf', type=float, default=0.1)
    ap.add_argument('--iou',  type=float, default=0.7)
    ap.add_argument('--quant', action='store_true', help='在模拟器上做 INT8，对齐量化结果')
    ap.add_argument('--calib-dir', default='', help='校准集目录（配合 --quant）')
    ap.add_argument('--outdir', default='artifacts/logs/compare', help='把日志也写入文件（目录不存在会自动建）')
    args = ap.parse_args()

    onnx = Path(args.onnx).expanduser().resolve()
    img  = Path(args.img).expanduser().resolve() if args.img else None
    calib_dir = Path(args.calib_dir).expanduser().resolve() if args.calib_dir else None

    if not onnx.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx}")
    # allow missing image (synthetic)

    x = preprocess(img, args.imgsz)

    # ONNX (CPU)
    y_onnx, t_onnx = run_onnx(onnx, x)

    # RKNN (simulator，从 ONNX 构建；不要 load_rknn)
    y_rknn, t_rknn = run_rknn_from_onnx(onnx, x, args.imgsz, args.quant, calib_dir)

    # 原始误差
    diff = y_onnx - y_rknn
    absdiff = np.abs(diff)
    print(f"ONNX: {t_onnx:.2f} ms, RKNN(sim): {t_rknn:.2f} ms")
    print(f"shape onnx {y_onnx.shape} rknn {y_rknn.shape}")
    print(f"MSE {np.mean(diff**2):.6f}  MAE {np.mean(absdiff):.6f}  MaxAbs {absdiff.max():.6f}")

    # 每通道
    C = y_onnx.shape[1]
    names = ['bx','by','bw','bh','obj'] + [f'cls{i}' for i in range(C-5)]
    mae_per = diff.reshape(C, -1).mean(axis=1)
    mx_per  = absdiff.reshape(C, -1).max(axis=1)
    for i, n in enumerate(names[:C]):
        print(f'{n:4s}  MAE={mae_per[i]:.6f}  MaxAbs={mx_per[i]:.6f}')

    # post-NMS 比较
    b1 = nms(decode_plain(y_onnx, args.num_classes, args.conf), args.iou)
    b2 = nms(decode_plain(y_rknn, args.num_classes, args.conf), args.iou)
    K = min(100, len(b1), len(b2))
    if K > 0:
        b1k = b1[np.argsort(-b1[:, 4])[:K]]
        b2k = b2[np.argsort(-b2[:, 4])[:K]]
        I = iou_matrix(b1k, b2k)
        match_iou = I.max(axis=1).mean()
        conf_gap  = np.abs(b1k[:, 4] - b2k[:, 4]).mean()
    else:
        match_iou, conf_gap = 1.0, 0.0
    print(f"post-NMS avg IoU(top{K}): {match_iou:.4f},  mean |Δconf|: {conf_gap:.6f}")

    # 选写日志
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H%M%S')
    (outdir / f'compare_{ts}.txt').write_text(
        "\n".join([
            f"onnx={onnx}", f"img={img if img else '(synthetic)'}",
            f"ONNX {t_onnx:.2f} ms | RKNN(sim) {t_rknn:.2f} ms",
            f"shape onnx {y_onnx.shape} rknn {y_rknn.shape}",
            f"MSE {np.mean(diff**2):.6f}  MAE {np.mean(absdiff):.6f}  MaxAbs {absdiff.max():.6f}",
            *(f"{names[i]:4s}  MAE={mae_per[i]:.6f}  MaxAbs={mx_per[i]:.6f}" for i in range(min(C, len(names)))),
            f"post-NMS avg IoU(top{K}): {match_iou:.4f},  mean |Δconf|: {conf_gap:.6f}",
        ])
    )

if __name__ == '__main__':
    main()
