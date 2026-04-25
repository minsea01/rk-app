#!/usr/bin/env python3
"""Live viewer for RK3588 detection JSON streams with embedded JPEG frames."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger("live_viewer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("LISTEN_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LISTEN_PORT", "9000")))
    parser.add_argument("--window-title", default="RK3588 YOLO Live")
    parser.add_argument("--save-latest", type=Path, default=Path("artifacts/live_view/latest.jpg"))
    parser.add_argument("--save-dir", type=Path, default=None, help="Optionally save every shown frame")
    parser.add_argument("--headless", action="store_true", help="Do not call cv2.imshow")
    parser.add_argument("--draw-local", action="store_true", help="Draw detections if image has no overlays")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames, 0 means unlimited")
    parser.add_argument("--stats-interval", type=float, default=2.0)
    parser.add_argument("--idle-timeout", type=float, default=0.0, help="Exit after idle seconds, 0 disables")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def decode_image(result: dict[str, Any]) -> np.ndarray | None:
    image_info = result.get("image")
    if not isinstance(image_info, dict):
        return None
    encoded = image_info.get("data_base64")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        LOGGER.warning("Invalid base64 image payload: %s", exc)
        return None
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        LOGGER.warning("Failed to decode JPEG payload")
    return frame


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for det in detections:
        try:
            x = int(round(float(det.get("x", 0))))
            y = int(round(float(det.get("y", 0))))
            w = int(round(float(det.get("w", 0))))
            h = int(round(float(det.get("h", 0))))
            conf = float(det.get("confidence", 0.0))
            name = str(det.get("class_name", det.get("class_id", "")))
        except (TypeError, ValueError):
            continue
        x0 = max(0, min(width - 1, x))
        y0 = max(0, min(height - 1, y))
        x1 = max(0, min(width - 1, x + w))
        y1 = max(0, min(height - 1, y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(output, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{name} {conf:.2f}".strip()
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(text_size[1] + baseline, y0)
        cv2.rectangle(
            output,
            (x0, label_y - text_size[1] - baseline),
            (min(width - 1, x0 + text_size[0]), label_y),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            output,
            label,
            (x0, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def should_use_headless(args: argparse.Namespace) -> bool:
    if args.headless:
        return True
    if sys.platform.startswith("linux") and not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
        LOGGER.warning("No DISPLAY/WAYLAND_DISPLAY detected; using headless latest-frame mode")
        return True
    return False


class LiveViewer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.headless = should_use_headless(args)
        self.frames = 0
        self.images = 0
        self.bytes_received = 0
        self.start_time = time.monotonic()
        self.last_stats = self.start_time
        self.last_rx = self.start_time
        self.running = True
        args.save_latest.parent.mkdir(parents=True, exist_ok=True)
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.args.host, self.args.port))
            server.listen(1)
            server.settimeout(1.0)
            LOGGER.info("Live viewer listening on %s:%d", self.args.host, self.args.port)
            LOGGER.info("Press q/Esc in the viewer window to exit")
            while self.running:
                try:
                    client, address = server.accept()
                except socket.timeout:
                    if self._idle_expired():
                        break
                    continue
                with client:
                    LOGGER.info("Connection from %s:%s", address[0], address[1])
                    self._handle_client(client)
            if not self.headless:
                cv2.destroyAllWindows()
            self._log_stats(force=True)

    def _handle_client(self, client: socket.socket) -> None:
        client.settimeout(1.0)
        buffer = bytearray()
        while self.running:
            try:
                chunk = client.recv(65536)
            except socket.timeout:
                if self._idle_expired():
                    break
                self._log_stats()
                continue
            if not chunk:
                break
            self.bytes_received += len(chunk)
            self.last_rx = time.monotonic()
            buffer.extend(chunk)
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    break
                line = bytes(buffer[:newline]).strip()
                del buffer[: newline + 1]
                if line:
                    self._process_line(line)
        if buffer:
            self._process_line(bytes(buffer).strip())

    def _process_line(self, line: bytes) -> None:
        try:
            result = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Invalid JSON frame: %s", exc)
            return
        self.frames += 1
        frame = decode_image(result)
        if frame is not None:
            self.images += 1
            image_info = result.get("image") if isinstance(result.get("image"), dict) else {}
            if self.args.draw_local and not image_info.get("contains_overlays", False):
                detections = result.get("detections", [])
                if isinstance(detections, list):
                    frame = draw_detections(frame, detections)
            self._show_or_save(frame, result)
        self._log_stats()
        if self.args.max_frames > 0 and self.frames >= self.args.max_frames:
            self.running = False

    def _show_or_save(self, frame: np.ndarray, result: dict[str, Any]) -> None:
        cv2.imwrite(str(self.args.save_latest), frame)
        if self.args.save_dir is not None:
            frame_id = result.get("frame_id", self.frames)
            cv2.imwrite(str(self.args.save_dir / f"frame_{int(frame_id):06d}.jpg"), frame)
        if self.headless:
            return
        try:
            cv2.imshow(self.args.window_title, frame)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            LOGGER.warning("cv2.imshow failed, switching to headless mode: %s", exc)
            self.headless = True
            return
        if key in {ord("q"), 27}:
            self.running = False

    def _idle_expired(self) -> bool:
        return self.args.idle_timeout > 0 and time.monotonic() - self.last_rx > self.args.idle_timeout

    def _log_stats(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_stats < self.args.stats_interval:
            return
        elapsed = max(1e-6, now - self.start_time)
        LOGGER.info(
            "frames=%d images=%d fps=%.2f rx=%.2f MB latest=%s",
            self.frames,
            self.images,
            self.frames / elapsed,
            self.bytes_received / (1024 * 1024),
            self.args.save_latest,
        )
        self.last_stats = now


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    LiveViewer(args).run()


if __name__ == "__main__":
    main()
