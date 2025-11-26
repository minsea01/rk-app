#!/usr/bin/env python3
"""High-performance streaming RKNN inference pipeline.

This module provides a multi-threaded pipeline for real-time object detection
using RKNN models on RK3588 hardware.
"""
import argparse
import logging
import time
from pathlib import Path
from threading import Thread, Event
from queue import Queue, Full, Empty
import json
from typing import Union, List, Dict, Any, Optional, Tuple
from urllib import request, error

import cv2
import numpy as np

from apps.utils.yolo_post import letterbox, nms, sigmoid
from apps.utils.headless import safe_imshow, safe_waitKey

# Setup logging
logger = logging.getLogger(__name__)


def parse_source(src: str) -> Union[int, str]:
    """Parse video source string to camera index or path.
    
    Args:
        src: Source string - digit for camera index, else path/URL
        
    Returns:
        Integer camera index if src is numeric, otherwise original string
    """
    if src and src.isdigit():
        return int(src)
    return src


def decode_predictions(
    pred: np.ndarray,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    head: str = 'auto'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO model predictions to bounding boxes.
    
    Args:
        pred: Model predictions array
        imgsz: Input image size used for inference
        conf_thres: Confidence threshold for filtering
        iou_thres: IOU threshold for NMS
        head: Decoding method ('auto', 'dfl', 'raw')
        
    Returns:
        Tuple of (boxes, confidences, class_ids)
    """
    # Normalize pred to (1, N, C)
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
        # Lazy import to avoid circular deps
        from apps.utils.yolo_post import postprocess_yolov8
        # Here we assume decode on resized image; caller will scale to original if needed
        boxes, confs, cls_ids = postprocess_yolov8(pred_nc, imgsz, (imgsz, imgsz), (1.0, (0.0, 0.0)), conf_thres, iou_thres)
        return boxes, confs, cls_ids
    # raw: [cx,cy,w,h,obj,cls...]
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
    return boxes[keep], conf[keep], cls_ids[keep]


class StageStats:
    """Statistics tracker for pipeline stage timing."""
    
    def __init__(self) -> None:
        self.n: int = 0
        self.t_sum: float = 0.0
        self.t_min: float = 1e9
        self.t_max: float = 0.0

    def reset(self) -> None:
        """Reset all statistics to initial values."""
        self.n = 0
        self.t_sum = 0.0
        self.t_min = 1e9
        self.t_max = 0.0

    def add(self, dt: float) -> None:
        """Add a timing measurement.
        
        Args:
            dt: Time duration in seconds
        """
        self.n += 1
        self.t_sum += dt
        self.t_min = min(self.t_min, dt)
        self.t_max = max(self.t_max, dt)

    def summary(self) -> Dict[str, float]:
        """Get statistics summary.
        
        Returns:
            Dictionary with n, avg_ms, min_ms, max_ms
        """
        avg = self.t_sum / max(1, self.n)
        return {'n': self.n, 'avg_ms': avg * 1000, 'min_ms': self.t_min * 1000, 'max_ms': self.t_max * 1000}


def run_stream(args) -> None:
    """Run the streaming inference pipeline.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        SystemExit: If RKNN toolkit is not installed or model fails to load
    """
    try:
        from rknnlite.api import RKNNLite
    except ImportError as e:
        raise SystemExit(f"rknn-toolkit-lite2 not installed: {e}")

    cap_src = parse_source(args.source)
    cap = cv2.VideoCapture(cap_src, cv2.CAP_GSTREAMER if isinstance(cap_src, str) else 0)
    if args.width > 0 and args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise SystemExit(f'Failed to open source: {args.source}')

    stop = Event()
    q_cap = Queue(maxsize=args.queue)
    q_pre = Queue(maxsize=args.queue)
    q_out = Queue(maxsize=args.queue)
    q_up = Queue(maxsize=max(2, args.queue // 2)) if args.upload_http else None

    stats = {
        'capture': StageStats(),
        'preproc': StageStats(),
        'infer': StageStats(),
        'post': StageStats(),
    }
    # shared upload throttle state
    pending_upload = None
    last_sent = 0.0

    def t_capture() -> None:
        """Capture thread: reads frames from video source."""
        dropped_frames = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                logger.warning("Video capture ended or failed")
                break
            t1 = time.perf_counter()
            try:
                q_cap.put((frame, t0, t1), timeout=0.1)
            except Full:
                dropped_frames += 1
                if dropped_frames % 100 == 1:
                    logger.debug(f"Capture queue full, dropped {dropped_frames} frames")
            stats['capture'].add(t1 - t0)
        stop.set()

    def t_preproc() -> None:
        """Preprocessing thread: letterbox resize and prepare input."""
        dropped_frames = 0
        while not stop.is_set():
            try:
                frame, t0, t1 = q_cap.get(timeout=0.1)
            except Empty:
                continue
            t2 = time.perf_counter()
            img, r, d = letterbox(frame, args.imgsz)
            # feed BGR uint8 to RKNN; conversion handled by runtime config (BGR->RGB, /255)
            t3 = time.perf_counter()
            try:
                q_pre.put((frame, img, r, d, t0, t1, t2, t3), timeout=0.1)
            except Full:
                dropped_frames += 1
                if dropped_frames % 100 == 1:
                    logger.debug(f"Preproc queue full, dropped {dropped_frames} frames")
            stats['preproc'].add(t3 - t2)
        stop.set()

    rknn = RKNNLite()
    if rknn.load_rknn(str(args.model)) != 0:
        raise SystemExit('load_rknn failed')
    if rknn.init_runtime(core_mask=args.core_mask) != 0:
        raise SystemExit('init_runtime failed')

    def t_infer() -> None:
        """Inference thread: runs RKNN model on preprocessed frames."""
        dropped_frames = 0
        # Warmup
        for _ in range(max(0, args.warmup)):
            try:
                frame, img, r, d, *ts = q_pre.get(timeout=1.0)
            except Empty:
                continue
            rknn.inference(inputs=[img])
        logger.debug(f"Inference warmup complete ({args.warmup} frames)")
        # Main
        while not stop.is_set():
            try:
                frame, img, r, d, t0, t1, t2, t3 = q_pre.get(timeout=0.1)
            except Empty:
                continue
            t4 = time.perf_counter()
            outputs = rknn.inference(inputs=[img])
            pred = outputs[0]
            if pred.ndim == 2:
                pred = pred[None, ...]
            t5 = time.perf_counter()
            try:
                q_out.put((frame, pred, r, d, t0, t1, t2, t3, t4, t5), timeout=0.1)
            except Full:
                dropped_frames += 1
                if dropped_frames % 100 == 1:
                    logger.debug(f"Output queue full, dropped {dropped_frames} frames")
            stats['infer'].add(t5 - t4)
        stop.set()

    def t_post() -> None:
        """Post-processing thread: decode predictions, draw boxes, save/display."""
        names = None
        if args.names and Path(args.names).exists():
            names = [x.strip() for x in Path(args.names).read_text().splitlines() if x.strip()]
        writer = None
        writer_size = None
        count = 0
        dropped_uploads = 0
        t_start = time.perf_counter()
        while not stop.is_set():
            try:
                frame, pred, r, d, t0, t1, t2, t3, t4, t5 = q_out.get(timeout=0.1)
            except Empty:
                continue
            t6 = time.perf_counter()
            boxes, confs, cls_ids = decode_predictions(pred, args.imgsz, args.conf, args.iou, head=args.head)
            # scale boxes back to original frame size when using letterbox
            if len(boxes) > 0:
                boxes[:, [0, 2]] -= d[0]
                boxes[:, [1, 3]] -= d[1]
                boxes /= r
                boxes[:, 0::2] = boxes[:, 0::2].clip(0, frame.shape[1] - 1)
                boxes[:, 1::2] = boxes[:, 1::2].clip(0, frame.shape[0] - 1)
            # draw
            for (x1, y1, x2, y2), c, cls in zip(boxes.astype(int), confs, cls_ids):
                label = f"{int(cls)}:{c:.2f}"
                if names and 0 <= int(cls) < len(names):
                    label = f"{names[int(cls)]}:{c:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            t7 = time.perf_counter()
            stats['post'].add(t7 - t6)
            count += 1
            now = time.perf_counter()
            if q_up is not None:
                dets = [
                    {
                        'xyxy': [int(x1), int(y1), int(x2), int(y2)],
                        'conf': float(c),
                        'cls': int(ci),
                    }
                    for (x1, y1, x2, y2), c, ci in zip(boxes, confs, cls_ids)
                ]
                payload = {'ts': time.time(), 'frame': count, 'detections': dets}
                try:
                    q_up.put(payload, timeout=0.01)
                except Full:
                    dropped_uploads += 1
                    if dropped_uploads % 100 == 1:
                        logger.debug(f"Upload queue full, dropped {dropped_uploads} payloads")
            if args.display:
                fps = count / max(1e-6, (now - t_start))
                cv2.putText(frame, f'FPS:{fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                # safe_imshow handles headless fallback
                if not safe_imshow('result', frame):
                    logger.debug("safe_imshow failed; continuing without display")
                if safe_waitKey(1) & 0xFF == 27:
                    stop.set()
                    break
            if writer:
                writer.write(frame)
            elif args.save:
                writer_size = (frame.shape[1], frame.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(args.save), fourcc, max(1, args.fps or 25), writer_size)
                if writer.isOpened():
                    writer.write(frame)
                else:
                    logger.error(f"Failed to open VideoWriter for {args.save}")
                    writer = None
            if args.max_frames > 0 and count >= args.max_frames:
                stop.set()
                break
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    def t_upload() -> None:
        """Upload thread: POST detection results to HTTP endpoint."""
        nonlocal last_sent, pending_upload
        url = args.upload_http
        headers = {'Content-Type': 'application/json'}
        failed_uploads = 0
        while not stop.is_set():
            if q_up is None:
                break
            try:
                payload = q_up.get(timeout=0.2)
            except Empty:
                continue
            now = time.time()
            # Throttle: keep latest payload if not yet time to send
            if args.upload_interval > 0 and (now - last_sent) < args.upload_interval:
                pending_upload = payload
                continue
            # If there is a pending newer payload, replace current
            if pending_upload is not None:
                payload = pending_upload
                pending_upload = None
            req = request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
            try:
                with request.urlopen(req, timeout=1.0) as resp:
                    _ = resp.read()
                last_sent = now
            except (error.URLError, error.HTTPError, TimeoutError) as e:
                # Log but don't block pipeline - network issues are expected in production
                failed_uploads += 1
                if failed_uploads % 50 == 1:
                    logger.warning(f"Upload failed ({failed_uploads} total): {type(e).__name__}")

    ths = [
        Thread(target=t_capture, daemon=True),
        Thread(target=t_preproc, daemon=True),
        Thread(target=t_infer, daemon=True),
        Thread(target=t_post, daemon=True),
    ]
    if q_up is not None:
        ths.append(Thread(target=t_upload, daemon=True))
    for t in ths:
        t.start()
    try:
        while not stop.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()
    for t in ths:
        t.join(timeout=1.0)

    # Summary
    total_time = sum(s.t_sum for s in stats.values())
    logger.info('Stage stats (ms):')
    for k, s in stats.items():
        logger.info(f"  {k}: {s.summary()}")


def main():
    p = argparse.ArgumentParser(description='High-performance streaming RKNN inference pipeline')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--source', type=str, default='0', help='camera index (as string) or gstreamer/rtsp/file path')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--head', type=str, choices=['auto', 'dfl', 'raw'], default='auto')
    p.add_argument('--names', type=Path, default=None)
    p.add_argument('--core-mask', type=lambda x: int(x, 0), default=0x7)
    p.add_argument('--queue', type=int, default=4)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--fps', type=int, default=0)
    p.add_argument('--width', type=int, default=0)
    p.add_argument('--height', type=int, default=0)
    p.add_argument('--display', action='store_true', default=False, help='show result window (auto-fallback to save when headless)')
    p.add_argument('--save', type=Path, default=None)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--upload-http', type=str, default=None, help='POST detections JSON to this URL')
    p.add_argument('--upload-interval', type=float, default=0.0, help='seconds between uploads (throttle)')
    args = p.parse_args()
    run_stream(args)


if __name__ == '__main__':
    main()
