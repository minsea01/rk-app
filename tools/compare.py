#!/usr/bin/env python3
"""Unified ONNX vs RKNN comparison tool."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


def _make_synth(size: int = 640) -> np.ndarray:
    rnd = np.random.RandomState(0)
    image = (rnd.rand(size, size, 3) * 255).astype(np.uint8)
    cv2.rectangle(image, (size // 4, size // 4), (size // 2, size // 2), (0, 255, 0), 2)
    cv2.circle(image, (int(size * 0.75), int(size * 0.25)), size // 10, (255, 0, 0), 3)
    return image


def load_and_preprocess_inputs(img_path: Optional[Path], imgsz: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ONNX(NCHW float32) and RKNN(NHWC uint8) inputs."""
    image = None
    if img_path:
        image = cv2.imread(str(img_path))
    if image is None:
        image = _make_synth(imgsz)

    resized = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    onnx_input = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    rknn_input = resized[None]
    return onnx_input, rknn_input


def _to_ca(output: np.ndarray) -> np.ndarray:
    """Normalize output to (1, C, A) when possible."""
    y = np.array(output)
    if y.ndim == 2:
        y = y[None, ...]
    # If shape is (1, A, C), transpose to (1, C, A)
    if y.ndim == 3 and y.shape[-1] < y.shape[1]:
        y = np.transpose(y, (0, 2, 1))
    return y


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_plain(
    output: np.ndarray,
    num_classes: int = 0,
    conf_thres: float = 0.1,
    apply_sigmoid: bool = True,
) -> np.ndarray:
    """Decode raw non-DFL tensor to Nx6: x1,y1,x2,y2,score,cls."""
    y = _to_ca(output)[0]
    if y.shape[0] < 5:
        return np.empty((0, 6), dtype=np.float32)

    bx, by, bw, bh = y[0], y[1], y[2], y[3]
    obj = y[4]
    cls = y[5:5 + num_classes] if num_classes > 0 else y[5:]
    if apply_sigmoid:
        obj = _sigmoid(obj)
        cls = _sigmoid(cls)

    if cls.size == 0:
        conf = obj
        cls_ids = np.zeros_like(conf, dtype=np.float32)
    else:
        conf = obj * cls.max(axis=0)
        cls_ids = cls.argmax(axis=0).astype(np.float32)

    sel = conf > conf_thres
    if not np.any(sel):
        return np.empty((0, 6), dtype=np.float32)

    x1 = bx[sel] - bw[sel] / 2
    y1 = by[sel] - bh[sel] / 2
    x2 = bx[sel] + bw[sel] / 2
    y2 = by[sel] + bh[sel] / 2
    return np.stack([x1, y1, x2, y2, conf[sel].astype(np.float32), cls_ids[sel]], axis=1)


def nms(boxes: np.ndarray, iou_thres: float = 0.7) -> np.ndarray:
    """Run NMS on Nx6 boxes."""
    if boxes.shape[0] == 0:
        return boxes
    x1, y1, x2, y2, s, _ = [boxes[:, i] for i in range(6)]
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_j = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        iou = inter / (area_i + area_j - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return boxes[keep]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x11, y11, x12, y12 = [a[:, i][:, None] for i in range(4)]
    x21, y21, x22, y22 = [b[:, i][None, :] for i in range(4)]
    xx1 = np.maximum(x11, x21)
    yy1 = np.maximum(y11, y21)
    xx2 = np.minimum(x12, x22)
    yy2 = np.minimum(y12, y22)
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    return inter / (area1 + area2 - inter + 1e-9)


def compute_tensor_metrics(onnx_out: np.ndarray, rknn_out: np.ndarray) -> Dict[str, float]:
    if onnx_out.shape != rknn_out.shape:
        if onnx_out.size != rknn_out.size:
            raise ValueError(f"Output shape mismatch: {onnx_out.shape} vs {rknn_out.shape}")
        rknn_out = rknn_out.reshape(onnx_out.shape)

    diff = onnx_out - rknn_out
    abs_diff = np.abs(diff)
    return {
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "median_abs": float(np.median(abs_diff)),
        "rms": float(np.sqrt(np.mean(diff ** 2))),
    }


def compute_post_metrics(
    onnx_out: np.ndarray,
    rknn_out: np.ndarray,
    *,
    num_classes: int,
    conf: float,
    iou: float,
) -> Dict[str, float]:
    boxes_onnx = nms(decode_plain(onnx_out, num_classes=num_classes, conf_thres=conf), iou)
    boxes_rknn = nms(decode_plain(rknn_out, num_classes=num_classes, conf_thres=conf), iou)

    k = min(100, len(boxes_onnx), len(boxes_rknn))
    if k > 0:
        top_onnx = boxes_onnx[np.argsort(-boxes_onnx[:, 4])[:k]]
        top_rknn = boxes_rknn[np.argsort(-boxes_rknn[:, 4])[:k]]
        iou_vals = iou_matrix(top_onnx[:, :4], top_rknn[:, :4])
        avg_iou = float(iou_vals.max(axis=1).mean())
        conf_gap = float(np.abs(top_onnx[:, 4] - top_rknn[:, 4]).mean())
    else:
        avg_iou = 1.0
        conf_gap = 0.0

    return {
        "onnx_detections": float(len(boxes_onnx)),
        "rknn_detections": float(len(boxes_rknn)),
        "topk": float(k),
        "avg_iou_topk": avg_iou,
        "mean_conf_abs_diff_topk": conf_gap,
    }


def parse_metric_set(metric_spec: str) -> Set[str]:
    valid = {"tensor", "post"}
    selected = {m.strip() for m in metric_spec.split(",") if m.strip()}
    if not selected:
        selected = {"tensor"}
    invalid = selected - valid
    if invalid:
        raise ValueError(f"Unsupported metrics: {sorted(invalid)}")
    return selected


def _infer_num_classes(output: np.ndarray, configured: int) -> int:
    if configured > 0:
        return configured
    y = _to_ca(output)
    if y.ndim != 3 or y.shape[1] < 6:
        return 0
    return max(0, y.shape[1] - 5)


def build_report(
    onnx_out: np.ndarray,
    rknn_out: np.ndarray,
    *,
    metrics: Iterable[str],
    num_classes: int,
    conf: float,
    iou: float,
    onnx_latency_ms: Optional[float] = None,
    rknn_latency_ms: Optional[float] = None,
) -> Dict[str, object]:
    metric_set = set(metrics)
    resolved_num_classes = _infer_num_classes(onnx_out, num_classes)
    results: Dict[str, object] = {}

    if "tensor" in metric_set:
        results["tensor"] = compute_tensor_metrics(np.array(onnx_out), np.array(rknn_out))
    if "post" in metric_set:
        results["post"] = compute_post_metrics(
            np.array(onnx_out),
            np.array(rknn_out),
            num_classes=resolved_num_classes,
            conf=conf,
            iou=iou,
        )

    return {
        "schema_version": "1.0",
        "meta": {
            "onnx_shape": list(np.array(onnx_out).shape),
            "rknn_shape": list(np.array(rknn_out).shape),
            "num_classes": resolved_num_classes,
            "onnx_latency_ms": onnx_latency_ms,
            "rknn_latency_ms": rknn_latency_ms,
        },
        "results": results,
    }


def run_onnx(onnx_path: Path, onnx_input: np.ndarray) -> Tuple[np.ndarray, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(f"onnxruntime not installed: {exc}") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    t0 = time.time()
    output = session.run(None, {input_name: onnx_input})[0]
    t1 = time.time()
    return np.array(output), (t1 - t0) * 1000.0


def _collect_calib_images(calib_dir: Path, limit: int = 200) -> Sequence[str]:
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(sorted(calib_dir.glob(ext)))
    return [str(p) for p in images[:limit]]


def run_rknn_from_onnx(
    onnx_path: Path,
    onnx_input: np.ndarray,
    *,
    imgsz: int,
    quant: bool,
    calib_dir: Optional[Path],
    target: str,
) -> Tuple[np.ndarray, float]:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(f"rknn-toolkit2 not installed: {exc}") from exc

    rk = RKNN(verbose=False)
    try:
        ret = rk.config(target_platform=target, optimization_level=3)
        if ret != 0:
            raise RuntimeError(f"RKNN config failed: {ret}")

        ret = rk.load_onnx(str(onnx_path), input_size_list=[[1, 3, imgsz, imgsz]])
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed: {ret}")

        if quant:
            if calib_dir is None or not calib_dir.exists():
                raise RuntimeError(f"Calibration dir not found: {calib_dir}")
            images = _collect_calib_images(calib_dir)
            if not images:
                raise RuntimeError(f"No calibration images found in: {calib_dir}")
            dataset_file = Path("/tmp/rknn_compare_calib.txt")
            dataset_file.write_text("\n".join(images), encoding="utf-8")
            ret = rk.build(do_quantization=True, dataset=str(dataset_file))
        else:
            ret = rk.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed: {ret}")

        ret = rk.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed: {ret}")

        t0 = time.time()
        outputs = rk.inference([onnx_input], data_format="nchw")
        t1 = time.time()
        return np.array(outputs[0]), (t1 - t0) * 1000.0
    finally:
        rk.release()


def run_rknn_from_rknn(
    rknn_path: Path,
    onnx_input: np.ndarray,
    rknn_input: np.ndarray,
    *,
    target: str,
) -> Tuple[np.ndarray, float]:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(f"rknn-toolkit2 not installed: {exc}") from exc

    rk = RKNN(verbose=False)
    try:
        ret = rk.load_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_rknn failed: {ret}")
        ret = rk.init_runtime(target=target, device_id=None)
        if ret != 0:
            ret = rk.init_runtime()
            if ret != 0:
                raise RuntimeError("RKNN init_runtime failed")

        # Try common input forms for compatibility with model export variants.
        t0 = time.time()
        outputs = None
        errors = []
        for candidate, kwargs in (
            ([rknn_input], {}),
            ([onnx_input], {"data_format": "nchw"}),
            ([onnx_input], {}),
        ):
            try:
                outputs = rk.inference(inputs=candidate, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        t1 = time.time()
        if outputs is None:
            raise RuntimeError(f"RKNN inference failed: {' | '.join(errors)}")
        return np.array(outputs[0]), (t1 - t0) * 1000.0
    finally:
        rk.release()


def compare_models(args: argparse.Namespace) -> Dict[str, object]:
    onnx_input, rknn_input = load_and_preprocess_inputs(args.img, args.imgsz)
    onnx_out, onnx_latency_ms = run_onnx(args.onnx, onnx_input)

    if args.rknn is not None:
        rknn_out, rknn_latency_ms = run_rknn_from_rknn(
            args.rknn,
            onnx_input,
            rknn_input,
            target=args.target,
        )
    else:
        rknn_out, rknn_latency_ms = run_rknn_from_onnx(
            args.onnx,
            onnx_input,
            imgsz=args.imgsz,
            quant=args.quant,
            calib_dir=args.calib_dir,
            target=args.target,
        )

    report = build_report(
        onnx_out=onnx_out,
        rknn_out=rknn_out,
        metrics=parse_metric_set(args.metrics),
        num_classes=args.num_classes,
        conf=args.conf,
        iou=args.iou,
        onnx_latency_ms=onnx_latency_ms,
        rknn_latency_ms=rknn_latency_ms,
    )
    report["meta"]["onnx"] = str(args.onnx)
    report["meta"]["rknn"] = str(args.rknn) if args.rknn else None
    report["meta"]["img"] = str(args.img) if args.img else None
    report["meta"]["imgsz"] = args.imgsz
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified ONNX vs RKNN comparison tool")
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX model path")
    parser.add_argument("--rknn", type=Path, default=None, help="Optional RKNN model path")
    parser.add_argument("--img", type=Path, default=None, help="Optional image path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--metrics", type=str, default="tensor,post", help="Comma list: tensor,post")
    parser.add_argument("--num-classes", type=int, default=0, help="0 means auto-infer from output channels")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold for post metrics")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS/post metrics")
    parser.add_argument("--quant", action="store_true", help="Build RKNN from ONNX with INT8 quantization")
    parser.add_argument("--calib-dir", type=Path, default=None, help="Calibration image directory for --quant")
    parser.add_argument("--target", type=str, default="rk3588", help="RKNN target platform")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    return parser


def _print_report(report: Dict[str, object]) -> None:
    meta = report["meta"]
    print(f"ONNX latency: {meta['onnx_latency_ms']:.2f} ms")
    print(f"RKNN latency: {meta['rknn_latency_ms']:.2f} ms")
    print(f"ONNX shape: {tuple(meta['onnx_shape'])}, RKNN shape: {tuple(meta['rknn_shape'])}")
    for key, value in report["results"].items():
        print(f"[{key}]")
        for metric_name, metric_value in value.items():
            print(f"  {metric_name}: {metric_value}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.onnx.exists():
        raise SystemExit(f"ONNX model not found: {args.onnx}")
    if args.rknn is not None and not args.rknn.exists():
        raise SystemExit(f"RKNN model not found: {args.rknn}")
    if args.img is not None and not args.img.exists():
        raise SystemExit(f"Image not found: {args.img}")

    try:
        report = compare_models(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    _print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report to: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

