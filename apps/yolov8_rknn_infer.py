#!/usr/bin/env python3
import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from apps.utils.yolo_post import letterbox, postprocess_yolov8
from apps.utils.headless import safe_imshow, safe_waitKey
from apps.exceptions import RKNNError, PreprocessError, InferenceError, ModelLoadError
from apps.logger import setup_logger


def decode_predictions(pred, imgsz, conf_thres, iou_thres, head='auto', ratio_pad=(1.0, (0.0, 0.0)), orig_shape=None):
    # pred: (1, N, C) or (1, C, N) or (N, C)
    from apps.utils.yolo_post import sigmoid, nms
    # unify to (1, N, C)
    if pred.ndim == 2:
        pred = pred[None, ...]
    # Normalize to (1, N, C)
    if pred.shape[1] >= pred.shape[2]:
        pred_nc = pred
    else:
        pred_nc = pred.transpose(0, 2, 1)
    C = pred_nc.shape[2]
    if head == 'auto':
        head = 'dfl' if C >= 64 else 'raw'

    if head == 'dfl':
        boxes, confs, cls_ids = postprocess_yolov8(
            pred_nc, imgsz, orig_shape or (imgsz, imgsz), ratio_pad, conf_thres, iou_thres
        )
        return boxes, confs, cls_ids

    # raw path as heuristic decode: [cx,cy,w,h,obj,cls...]
    p = pred_nc[0]
    if C < 5:
        return np.empty((0, 4)), np.array([]), np.array([])
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    obj = sigmoid(p[:, 4])
    if C > 5:
        cls_scores = sigmoid(p[:, 5:])
        cls_ids = cls_scores.argmax(axis=1)
        cls_conf = cls_scores.max(axis=1)
    else:
        cls_ids = np.zeros_like(obj, dtype=np.int64)
        cls_conf = np.ones_like(obj)
    conf = obj * cls_conf
    m = conf >= conf_thres
    if not np.any(m):
        return np.empty((0, 4)), np.array([]), np.array([])
    scale_needed = (np.percentile(w[m], 95) < 1.0) or (np.percentile(h[m], 95) < 1.0)
    s = float(imgsz) if scale_needed else 1.0
    cx *= s; cy *= s; w *= s; h *= s
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)[m]
    conf = conf[m]
    cls_ids = cls_ids[m]
    keep = nms(boxes, conf, iou_thres=iou_thres)
    boxes = boxes[keep]; conf = conf[keep]; cls_ids = cls_ids[keep]

    # Scale back to original coordinates if provided
    r, (dw, dh) = ratio_pad
    if orig_shape is not None and r is not None:
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= r
        h0, w0 = orig_shape
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0 - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0 - 1)
    return boxes, conf, cls_ids


def load_labels(names_path: Path):
    if names_path and names_path.exists():
        return [x.strip() for x in names_path.read_text().splitlines() if x.strip()]
    return None


def draw_boxes(img, boxes, confs, class_ids, names=None):
    for (x1, y1, x2, y2), c, cls in zip(boxes.astype(int), confs, class_ids):
        label = f"{int(cls)}:{c:.2f}"
        if names and 0 <= int(cls) < len(names):
            label = f"{names[int(cls)]}:{c:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description='YOLOv8 RKNNLite inference on RK3588')
    ap.add_argument('--model', type=Path, required=True, help='path to .rknn model file')
    ap.add_argument('--source', type=Path, help='image path; if omitted, opens /dev/video0')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='confidence threshold (default: 0.5, recommended >=0.5 for production to avoid NMS bottleneck)'
    )
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--names', type=Path, default=None, help='names.txt (one class per line)')
    ap.add_argument('--head', type=str, choices=['auto', 'dfl', 'raw'], default='auto', help='head decode type')
    ap.add_argument('--core-mask', type=lambda x: int(x, 0), default=0x7, help='NPU core mask (e.g., 0x7 for 3 cores)')
    ap.add_argument('--save', type=Path, default=None, help='save annotated result image')
    ap.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='logging verbosity'
    )
    args = ap.parse_args()

    logger = setup_logger(__name__, level=getattr(logging, args.log_level.upper()))

    try:
        from rknnlite.api import RKNNLite
    except ImportError as e:
        raise SystemExit(
            f"rknn-toolkit-lite2 not installed on device.\n"
            f"Install with: pip install rknn-toolkit-lite2\n"
            f"Error: {e}"
        )

    rknn = RKNNLite()

    # Load RKNN model
    logger.info('Loading RKNN: %s', args.model)
    if not args.model.exists():
        raise ModelLoadError(f"Model file not found: {args.model}")

    try:
        ret = rknn.load_rknn(str(args.model))
        if ret != 0:
            raise ModelLoadError(f'Failed to load RKNN model: {args.model}')
    except ModelLoadError:
        raise
    except (IOError, OSError) as e:
        raise ModelLoadError(f'Error reading RKNN model file: {e}') from e

    # Initialize runtime
    logger.info('Initializing runtime, core_mask=%s', hex(args.core_mask))
    try:
        ret = rknn.init_runtime(core_mask=args.core_mask)
        if ret != 0:
            raise RKNNError(f'Failed to initialize RKNN runtime with core_mask={hex(args.core_mask)}')
    except RKNNError:
        raise
    except RuntimeError as e:
        raise RKNNError(f'RKNN runtime initialization error: {e}') from e

    class_names = load_labels(args.names)

    if args.source and args.source.exists():
        # Load and preprocess image
        try:
            img0 = cv2.imread(str(args.source))
            if img0 is None:
                raise PreprocessError(f'Failed to read image: {args.source}')
            img, r, d = letterbox(img0, args.imgsz)
        except PreprocessError:
            raise
        except (cv2.error, ValueError) as e:
            raise PreprocessError(f'Error preprocessing image: {e}') from e

        # Run inference
        # RKNN preproc set to BGR->RGB and /255 in conversion; feed BGR uint8
        input_data = img
        t0 = time.time()
        try:
            outputs = rknn.inference(inputs=[input_data])
        except (RuntimeError, ValueError) as e:
            raise InferenceError(f'RKNN inference failed: {e}') from e
        dt = (time.time() - t0) * 1000
        logger.info('Inference time: %.2f ms', dt)
        pred = outputs[0]
        if pred.ndim == 2:
            pred = pred[None, ...]
        boxes, confs, cls_ids = decode_predictions(pred, args.imgsz, args.conf, args.iou, args.head, (r, d), img0.shape[:2])
        logger.info('Detections: %d', len(boxes))
        vis = draw_boxes(img0.copy(), boxes, confs, cls_ids, class_names)
        if args.save:
            cv2.imwrite(str(args.save), vis)
            logger.info('Saved: %s', args.save)
        else:
            # Auto-handles headless mode (saves to file instead of displaying)
            safe_imshow('result', vis, fallback_path='artifacts/result.jpg', wait_key=0)
        rknn.release()
        return

    # Fallback to camera
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Important: Release cap even if it failed to open
            # cv2.VideoCapture() returns an object even on failure
            if cap is not None:
                cap.release()
                cap = None  # Prevent double-release in finally block
            raise PreprocessError('Failed to open camera (/dev/video0)')

        fps_hist = []
        while True:
            ret, img0 = cap.read()
            if not ret:
                logger.info("Video capture ended")
                break
            try:
                img, r, d = letterbox(img0, args.imgsz)
                t0 = time.time()
                outputs = rknn.inference(inputs=[img])
                pred = outputs[0]
            except (cv2.error, RuntimeError, ValueError) as e:
                logger.warning('Inference error (skipping frame): %s', e)
                continue
            if pred.ndim == 2:
                pred = pred[None, ...]
            boxes, confs, cls_ids = decode_predictions(pred, args.imgsz, args.conf, args.iou, args.head, (r, d), img0.shape[:2])
            t1 = time.time()
            fps = 1.0 / max(1e-6, (t1 - t0))
            fps_hist.append(fps)
            vis = draw_boxes(img0.copy(), boxes, confs, cls_ids, class_names)
            cv2.putText(vis, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            # Auto-handles headless mode (saves to file instead of displaying)
            safe_imshow('result', vis, fallback_path='artifacts/camera_frame.jpg')

            # Check for ESC key to exit (GUI mode only)
            if safe_waitKey(1) & 0xFF == 27:
                break
    except PreprocessError:
        # Re-raise preprocessing errors
        raise
    except (cv2.error, RuntimeError) as e:
        # Wrap known errors that can occur during video processing
        raise PreprocessError(f'Camera processing error: {e}') from e
    finally:
        # Always clean up resources
        if cap is not None:
            cap.release()
        rknn.release()
        cv2.destroyAllWindows()
        if fps_hist:
            logger.info(
                'Avg FPS: %.2f, P90: %.2f',
                np.mean(fps_hist),
                np.percentile(fps_hist, 90)
            )


if __name__ == '__main__':
    main()
