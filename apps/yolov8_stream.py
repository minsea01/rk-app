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
from typing import Union, List, Dict, Any, Optional, Tuple, Mapping
from urllib import request, error

import cv2
import numpy as np

from apps.deprecation import warn_deprecated
from apps.exceptions import InferenceError, PreprocessError, RKAppException, RKNNError
from apps.logger import setup_logger
from apps.utils.decode import decode_predictions as _decode_predictions
from apps.utils.decode_meta import load_decode_meta
from apps.utils.preprocess_pipeline import (
    PreprocessState,
    build_preprocess_config,
    map_boxes_back,
    run_preprocess,
)
from apps.utils.headless import safe_imshow, safe_waitKey

# Setup logging
logger = setup_logger(__name__)


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


def _should_use_gstreamer(src: str) -> bool:
    """Heuristic to decide if GStreamer backend is required."""
    if not src:
        return False
    lowered = src.lower()
    return (
        lowered.startswith("rtsp://")
        or lowered.startswith("rtsps://")
        or lowered.startswith("gst:")
        or "!" in src
    )


def _should_retry_capture(src: str, cap_src: Union[int, str]) -> bool:
    """Return True if the source is a live stream that should reconnect on failure."""
    if isinstance(cap_src, int):
        return True
    if not src:
        return False
    lowered = src.lower()
    if lowered.startswith(("/dev/video", "rtsp://", "rtsps://", "http://", "https://")):
        return True
    if lowered.startswith("gst:") or "!" in src:
        return True
    return False


def decode_predictions(
    pred: np.ndarray,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    head: str = 'auto',
    decode_meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO predictions in resize-space for stream processing."""
    warn_deprecated(
        "apps.yolov8_stream.decode_predictions",
        "apps.utils.decode.decode_predictions",
        once=True,
    )
    return _decode_predictions(
        pred=pred,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        head=head,
        decode_meta=decode_meta,
        orig_shape=None,
        ratio_pad=(1.0, (0.0, 0.0)),
    )


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
        raise RKNNError(f"rknn-toolkit-lite2 not installed: {e}") from e

    preprocess_overrides = {
        'profile': args.pp_profile,
        'undistort_enable': args.undistort_enable,
        'calibration_file': args.undistort_calib,
        'roi_enable': args.roi_enable,
        'roi_mode': args.roi_mode,
        'roi_normalized_xywh': args.roi_norm,
        'roi_pixel_xywh': args.roi_px,
        'roi_min_size': args.roi_min_size,
        'roi_clamp': args.roi_clamp,
        'gamma_enable': args.gamma_enable,
        'gamma_value': args.gamma,
        'white_balance_enable': args.white_balance_enable,
        'white_balance_clip_percent': args.white_balance_clip,
        'denoise_enable': args.denoise_enable,
        'denoise_method': args.denoise_method,
        'denoise_d': args.denoise_d,
        'denoise_sigma_color': args.denoise_sigma_color,
        'denoise_sigma_space': args.denoise_sigma_space,
        'input_format': args.input_format,
    }
    try:
        preprocess_config = build_preprocess_config(args.cfg, preprocess_overrides, logger=logger)
    except (ValueError, OSError) as e:
        raise PreprocessError(f"Failed to load preprocess config: {e}") from e
    preprocess_state = PreprocessState()
    logger.info(
        "Preprocess profile=%s roi=%s wb=%s gamma=%s denoise=%s",
        preprocess_config.profile,
        preprocess_config.roi_enable,
        preprocess_config.white_balance_enable,
        preprocess_config.gamma_enable,
        preprocess_config.denoise_enable,
    )

    cap_src = parse_source(args.source)
    use_gst = isinstance(cap_src, str) and _should_use_gstreamer(cap_src)

    def _open_capture():
        cap_obj = cv2.VideoCapture(cap_src, cv2.CAP_GSTREAMER if use_gst else 0)
        if args.width > 0 and args.height > 0:
            cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.fps > 0:
            cap_obj.set(cv2.CAP_PROP_FPS, args.fps)
        return cap_obj

    cap = _open_capture()
    if not cap.isOpened():
        raise PreprocessError(f'Failed to open source: {args.source}')

    # Debug: print video properties
    logger.info(f"Video opened: {args.source}")
    logger.info(f"  Size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    logger.info(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    logger.info(f"  Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

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
        nonlocal cap
        dropped_frames = 0
        retry_on_fail = _should_retry_capture(args.source, cap_src)
        logger.info(f"Capture thread started (retry_on_fail={retry_on_fail})")
        reconnect_delay = 0.5
        reconnect_max = 5.0
        reconnect_attempts = 0
        frame_num = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            frame_num += 1
            if not ok:
                logger.warning(f"Read failed at frame {frame_num}, retry_on_fail={retry_on_fail}")
                if not retry_on_fail:
                    logger.warning("Video capture ended or failed")
                    break
                reconnect_attempts += 1
                if reconnect_attempts % 10 == 1:
                    logger.warning("Capture read failed; reconnecting (attempt %d)", reconnect_attempts)
                cap.release()
                time.sleep(reconnect_delay)
                cap = _open_capture()
                if cap.isOpened():
                    reconnect_attempts = 0
                    reconnect_delay = 0.5
                    continue
                reconnect_delay = min(reconnect_max, reconnect_delay * 2)
                continue
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
            try:
                t2 = time.perf_counter()
                try:
                    img, frame_meta, coord_frame = run_preprocess(
                        frame, args.imgsz, preprocess_config, preprocess_state, logger=logger
                    )
                except (ValueError, cv2.error, RuntimeError) as e:
                    raise PreprocessError(f"run_preprocess failed: {e}") from e
                # feed BGR uint8 to RKNN; conversion handled by runtime config (BGR->RGB, /255)
                t3 = time.perf_counter()
                try:
                    q_pre.put((coord_frame, img, frame_meta, t0, t1, t2, t3), timeout=0.1)
                except Full:
                    dropped_frames += 1
                    if dropped_frames % 100 == 1:
                        logger.debug(f"Preproc queue full, dropped {dropped_frames} frames")
                stats['preproc'].add(t3 - t2)
            except PreprocessError as e:
                logger.error(f"Preproc thread error: {e}")
                stop.set()
                break
        stop.set()

    rknn = RKNNLite()
    if rknn.load_rknn(str(args.model)) != 0:
        raise RKNNError('load_rknn failed')
    if rknn.init_runtime(core_mask=args.core_mask) != 0:
        raise RKNNError('init_runtime failed')
    decode_meta = load_decode_meta(args.model, logger=logger)

    def t_infer() -> None:
        """Inference thread: runs RKNN model on preprocessed frames."""
        dropped_frames = 0
        try:
            # Warmup
            for _ in range(max(0, args.warmup)):
                try:
                    frame, img, frame_meta, *ts = q_pre.get(timeout=1.0)
                except Empty:
                    continue
                # Add batch dimension for RKNN
                img_batch = img[None, ...] if img.ndim == 3 else img
                try:
                    rknn.inference(inputs=[img_batch])
                except (RuntimeError, ValueError, TypeError) as e:
                    raise RKNNError(f"Warmup inference failed: {e}") from e
            logger.debug(f"Inference warmup complete ({args.warmup} frames)")
            # Main
            while not stop.is_set():
                try:
                    frame, img, frame_meta, t0, t1, t2, t3 = q_pre.get(timeout=0.1)
                except Empty:
                    continue
                t4 = time.perf_counter()
                # Add batch dimension: (H, W, 3) → (1, H, W, 3)
                img_batch = img[None, ...] if img.ndim == 3 else img
                try:
                    outputs = rknn.inference(inputs=[img_batch])
                except (RuntimeError, ValueError, TypeError) as e:
                    raise RKNNError(f"Inference failed: {e}") from e
                pred = outputs[0]
                if pred.ndim == 2:
                    pred = pred[None, ...]
                t5 = time.perf_counter()
                try:
                    q_out.put((frame, pred, frame_meta, t0, t1, t2, t3, t4, t5), timeout=0.1)
                except Full:
                    dropped_frames += 1
                    if dropped_frames % 100 == 1:
                        logger.debug(f"Output queue full, dropped {dropped_frames} frames")
                stats['infer'].add(t5 - t4)
        except RKNNError as e:
            logger.error(f"Infer thread error: {e}")
        finally:
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
                frame, pred, frame_meta, t0, t1, t2, t3, t4, t5 = q_out.get(timeout=0.1)
            except Empty:
                continue
            t6 = time.perf_counter()
            try:
                try:
                    boxes, confs, cls_ids = decode_predictions(
                        pred, args.imgsz, args.conf, args.iou, head=args.head, decode_meta=decode_meta
                    )
                    boxes = map_boxes_back(boxes, frame_meta)
                except (ValueError, TypeError, cv2.error) as e:
                    raise InferenceError(f"Postprocess failed: {e}") from e
            except InferenceError as e:
                logger.error(f"Postprocess error: {e}")
                stop.set()
                break
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
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Ignore error in headless environment
            pass

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
        try:
            while not stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            stop.set()
        logger.info("Stopping threads...")
        for t in ths:
            t.join(timeout=1.0)
        logger.info("All threads stopped")
    finally:
        if cap:
            cap.release()
        try:
            rknn.release()
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning("RKNN release failed: %s", e)
        # Summary
        total_time = sum(s.t_sum for s in stats.values())
        logger.info('Stage stats (ms):')
        for k, s in stats.items():
            logger.info(f"  {k}: {s.summary()}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='High-performance streaming RKNN inference pipeline')
    p.set_defaults(
        undistort_enable=None,
        roi_enable=None,
        gamma_enable=None,
        white_balance_enable=None,
        denoise_enable=None,
        roi_clamp=None,
    )
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--cfg', type=Path, default=None, help='optional YAML config file')
    p.add_argument('--source', type=str, default='0', help='camera index (as string) or gstreamer/rtsp/file path')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--head', type=str, choices=['auto', 'dfl', 'raw'], default='auto')
    p.add_argument('--pp-profile', type=str, default=None, choices=['speed', 'balanced', 'quality'])
    p.add_argument('--undistort-enable', action='store_true')
    p.add_argument('--undistort-calib', type=str, default=None)
    p.add_argument('--roi-enable', action='store_true')
    p.add_argument('--roi-mode', type=str, default=None, choices=['normalized', 'pixel'])
    p.add_argument('--roi-norm', type=str, default=None, help='normalized ROI x,y,w,h in [0,1]')
    p.add_argument('--roi-px', type=str, default=None, help='pixel ROI x,y,w,h')
    p.add_argument('--roi-min-size', type=int, default=None)
    p.add_argument('--roi-no-clamp', dest='roi_clamp', action='store_false')
    p.add_argument('--gamma-enable', action='store_true')
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--white-balance-enable', action='store_true')
    p.add_argument('--white-balance-clip', type=float, default=None)
    p.add_argument('--denoise-enable', action='store_true')
    p.add_argument('--denoise-method', type=str, default=None, choices=['bilateral'])
    p.add_argument('--denoise-d', type=int, default=None)
    p.add_argument('--denoise-sigma-color', type=float, default=None)
    p.add_argument('--denoise-sigma-space', type=float, default=None)
    p.add_argument(
        '--input-format',
        type=str,
        default=None,
        choices=['auto', 'bgr', 'rgb', 'gray', 'bayer_rg', 'bayer_bg', 'bayer_gr', 'bayer_gb'],
    )
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
    p.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='logging verbosity',
    )
    return p


def main():
    global logger
    p = build_arg_parser()
    args = p.parse_args()
    logger = setup_logger(__name__, level=getattr(logging, args.log_level.upper()))
    try:
        run_stream(args)
    except RKAppException as e:
        logger.error("%s", e)
        raise SystemExit(str(e)) from e


if __name__ == '__main__':
    main()
