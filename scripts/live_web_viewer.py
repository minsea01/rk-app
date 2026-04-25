#!/usr/bin/env python3
"""Browser viewer for RK3588 detection JSON streams with embedded JPEG frames."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("live_web_viewer")
MJPEG_BOUNDARY = "rkappframe"
DEFAULT_MAX_LINE_BYTES = 32 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcp-host", default=os.getenv("LISTEN_HOST", "0.0.0.0"))
    parser.add_argument("--tcp-port", type=int, default=int(os.getenv("LISTEN_PORT", "9000")))
    parser.add_argument("--http-host", default=os.getenv("HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--http-port", type=int, default=int(os.getenv("HTTP_PORT", "8080")))
    parser.add_argument("--save-latest", type=Path, default=Path("artifacts/live_view/latest.jpg"))
    parser.add_argument(
        "--save-dir", type=Path, default=None, help="Optionally save every received JPEG"
    )
    parser.add_argument(
        "--save-dir-max-files",
        type=int,
        default=1000,
        help="Maximum retained files in --save-dir, 0 disables pruning",
    )
    parser.add_argument(
        "--max-line-bytes",
        type=int,
        default=DEFAULT_MAX_LINE_BYTES,
        help="Maximum bytes allowed for one newline-delimited JSON result",
    )
    parser.add_argument(
        "--max-mjpeg-clients",
        type=int,
        default=4,
        help="Maximum concurrent /stream.mjpg clients, 0 means unlimited",
    )
    parser.add_argument("--stats-interval", type=float, default=2.0)
    parser.add_argument(
        "--idle-timeout", type=float, default=0.0, help="Stop after idle seconds, 0 disables"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def decode_embedded_jpeg(result: dict[str, Any]) -> bytes | None:
    image_info = result.get("image")
    if not isinstance(image_info, dict):
        return None

    encoded = image_info.get("data_base64")
    if not isinstance(encoded, str) or not encoded:
        return None

    try:
        return base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Invalid base64 JPEG payload: %s", exc)
        return None


def sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(result)
    image_info = sanitized.get("image")
    if isinstance(image_info, dict):
        image_info.pop("data_base64", None)
    return sanitized


class LiveFrameState:
    def __init__(
        self,
        save_latest: Path,
        save_dir: Path | None,
        save_dir_max_files: int = 0,
    ):
        self.save_latest = save_latest
        self.save_dir = save_dir
        self.save_dir_max_files = max(0, save_dir_max_files)
        self.saved_frame_paths: deque[Path] = deque()
        self.saved_frame_path_set: set[Path] = set()
        self.condition = threading.Condition()
        self.started_at = time.monotonic()
        self.last_rx_at = 0.0
        self.results = 0
        self.images = 0
        self.bytes_received = 0
        self.image_sequence = 0
        self.latest_jpeg: bytes | None = None
        self.latest_result: dict[str, Any] | None = None
        self.latest_frame_id: int | str | None = None
        self.latest_saved_path: str | None = None

        self.save_latest.parent.mkdir(parents=True, exist_ok=True)
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def update_from_result(self, result: dict[str, Any], raw_size: int) -> None:
        jpeg = decode_embedded_jpeg(result)
        sanitized = sanitize_result(result)
        frame_id = sanitized.get("frame_id")
        saved_path = None

        if jpeg is not None:
            self.save_latest.write_bytes(jpeg)
            saved_path = str(self.save_latest)
            if self.save_dir is not None:
                per_frame = self.save_dir / f"frame_{self._format_frame_id(frame_id)}.jpg"
                per_frame.write_bytes(jpeg)
                self._remember_saved_frame(per_frame)

        with self.condition:
            self.results += 1
            self.bytes_received += raw_size
            self.last_rx_at = time.monotonic()
            self.latest_result = sanitized
            self.latest_frame_id = frame_id
            if jpeg is not None:
                self.images += 1
                self.image_sequence += 1
                self.latest_jpeg = jpeg
                self.latest_saved_path = saved_path
            self.condition.notify_all()

    def wait_for_image(self, last_sequence: int, timeout: float = 10.0) -> tuple[bytes | None, int]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.image_sequence <= last_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_sequence
                self.condition.wait(timeout=remaining)
            return self.latest_jpeg, self.image_sequence

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            stats = self._stats_locked()
            return {
                "stats": stats,
                "latest_result": copy.deepcopy(self.latest_result),
                "latest_frame_id": self.latest_frame_id,
                "latest_saved_path": self.latest_saved_path,
                "has_image": self.latest_jpeg is not None,
            }

    def stats(self) -> dict[str, Any]:
        with self.condition:
            return self._stats_locked()

    def current_jpeg(self) -> bytes | None:
        with self.condition:
            return self.latest_jpeg

    def _stats_locked(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = max(now - self.started_at, 1e-6)
        idle_seconds = None if self.last_rx_at == 0.0 else now - self.last_rx_at
        return {
            "results": self.results,
            "images": self.images,
            "result_fps": self.results / elapsed,
            "image_fps": self.images / elapsed,
            "bytes_received": self.bytes_received,
            "rx_mib": self.bytes_received / (1024 * 1024),
            "idle_seconds": idle_seconds,
            "uptime_seconds": elapsed,
        }

    def _remember_saved_frame(self, path: Path) -> None:
        if path not in self.saved_frame_path_set:
            self.saved_frame_paths.append(path)
            self.saved_frame_path_set.add(path)

        if self.save_dir_max_files <= 0:
            return

        while len(self.saved_frame_paths) > self.save_dir_max_files:
            old_path = self.saved_frame_paths.popleft()
            self.saved_frame_path_set.discard(old_path)
            if old_path == path:
                continue
            try:
                old_path.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("Failed to prune old live frame %s: %s", old_path, exc)

    @staticmethod
    def _format_frame_id(frame_id: Any) -> str:
        try:
            return f"{int(frame_id):06d}"
        except (TypeError, ValueError):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            return timestamp


class DetectionTCPServer:
    def __init__(
        self,
        host: str,
        port: int,
        state: LiveFrameState,
        idle_timeout: float,
        max_line_bytes: int,
    ):
        self.host = host
        self.port = port
        self.state = state
        self.idle_timeout = idle_timeout
        self.max_line_bytes = max(1, max_line_bytes)
        self.running = threading.Event()
        self.running.set()
        self.ready = threading.Event()
        self.start_error: OSError | None = None
        self.socket: socket.socket | None = None

    def serve_forever(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                self.socket = server
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.host, self.port))
                server.listen(4)
                server.settimeout(1.0)
                self.ready.set()
                LOGGER.info("TCP receiver listening on %s:%d", self.host, self.port)

                while self.running.is_set():
                    try:
                        client, address = server.accept()
                    except socket.timeout:
                        if self._idle_expired():
                            LOGGER.info("TCP receiver idle timeout reached")
                            self.stop()
                        continue
                    except OSError as exc:
                        if self.running.is_set():
                            LOGGER.error("TCP receiver accept failed: %s", exc)
                        break

                    LOGGER.info("Detection stream connected from %s:%s", address[0], address[1])
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(client, address),
                        name=f"detection-client-{address[0]}:{address[1]}",
                        daemon=True,
                    )
                    thread.start()
        except OSError as exc:
            self.start_error = exc
            self.ready.set()
            LOGGER.error("TCP receiver failed to start on %s:%d: %s", self.host, self.port, exc)
        finally:
            self.running.clear()

    def stop(self) -> None:
        self.running.clear()
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass

    def _handle_client(self, client: socket.socket, address: tuple[str, int]) -> None:
        with client:
            client.settimeout(1.0)
            buffer = bytearray()
            while self.running.is_set():
                try:
                    chunk = client.recv(65536)
                except socket.timeout:
                    if self._idle_expired():
                        break
                    continue
                except OSError as exc:
                    LOGGER.warning("Socket error from %s:%s: %s", address[0], address[1], exc)
                    break

                if not chunk:
                    break

                buffer.extend(chunk)
                while True:
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        break
                    if newline > self.max_line_bytes:
                        LOGGER.warning(
                            "Dropping oversized JSON frame from %s:%s (%d bytes)",
                            address[0],
                            address[1],
                            newline,
                        )
                        del buffer[: newline + 1]
                        continue
                    line = bytes(buffer[:newline]).strip()
                    del buffer[: newline + 1]
                    if line:
                        self._process_line(line)
                if len(buffer) > self.max_line_bytes:
                    LOGGER.warning(
                        "Closing %s:%s after oversized partial JSON frame (%d bytes)",
                        address[0],
                        address[1],
                        len(buffer),
                    )
                    return

            if buffer:
                if len(buffer) > self.max_line_bytes:
                    LOGGER.warning(
                        "Ignoring oversized final JSON frame from %s:%s (%d bytes)",
                        address[0],
                        address[1],
                        len(buffer),
                    )
                    return
                line = bytes(buffer).strip()
                if line:
                    self._process_line(line)

    def _process_line(self, line: bytes) -> None:
        try:
            result = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Invalid detection JSON frame: %s", exc)
            return

        if not isinstance(result, dict):
            LOGGER.warning("Ignoring non-object detection payload")
            return

        self.state.update_from_result(result, raw_size=len(line))

    def _idle_expired(self) -> bool:
        if self.idle_timeout <= 0:
            return False
        stats = self.state.stats()
        idle_seconds = stats["idle_seconds"]
        return idle_seconds is not None and idle_seconds > self.idle_timeout


class LiveWebHandler(BaseHTTPRequestHandler):
    server_version = "RKAppLiveWeb/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.debug("HTTP %s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_html(INDEX_HTML)
        elif self.path == "/latest.jpg":
            self._send_latest_jpeg()
        elif self.path == "/stream.mjpg":
            self._send_mjpeg_stream()
        elif self.path == "/api/latest":
            self._send_json(self._state().snapshot())
        elif self.path == "/api/stats":
            self._send_json(self._state().stats())
        elif self.path == "/healthz":
            self._send_json({"ok": True})
        elif self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_latest_jpeg(self) -> None:
        jpeg = self._state().current_jpeg()
        if jpeg is None:
            self.send_error(HTTPStatus.NOT_FOUND, "No frame received yet")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpeg)

    def _send_mjpeg_stream(self) -> None:
        semaphore = self._mjpeg_semaphore()
        acquired = False
        if semaphore is not None:
            acquired = semaphore.acquire(blocking=False)
            if not acquired:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Too many MJPEG clients")
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}")
        self.end_headers()

        try:
            last_sequence = 0
            while True:
                jpeg, last_sequence = self._state().wait_for_image(last_sequence)
                if jpeg is None:
                    continue
                try:
                    self.wfile.write(f"--{MJPEG_BOUNDARY}\r\n".encode("ascii"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
        finally:
            if acquired and semaphore is not None:
                semaphore.release()

    def _state(self) -> LiveFrameState:
        return self.server.state  # type: ignore[attr-defined]

    def _mjpeg_semaphore(self) -> threading.BoundedSemaphore | None:
        return self.server.mjpeg_semaphore  # type: ignore[attr-defined]


class LiveWebHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        state: LiveFrameState,
        max_mjpeg_clients: int,
    ):
        super().__init__(server_address, handler_class)
        self.state = state
        self.mjpeg_semaphore = (
            threading.BoundedSemaphore(max_mjpeg_clients) if max_mjpeg_clients > 0 else None
        )


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RK3588 Live Detection</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f5f2;
      --panel: #ffffff;
      --ink: #1c1f22;
      --muted: #626a70;
      --line: #d6dbd7;
      --accent: #0b7a53;
      --warn: #b45b1a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      padding: 16px;
      height: calc(100vh - 56px);
    }
    .stage {
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #111;
      border: 1px solid #0b0b0b;
      border-radius: 6px;
      overflow: hidden;
    }
    .stage img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    aside {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 12px;
    }
    .metrics, .json {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      min-width: 0;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1px;
      overflow: hidden;
      background: var(--line);
    }
    .metric {
      min-width: 0;
      padding: 12px;
      background: var(--panel);
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: 20px;
      font-weight: 700;
      line-height: 1.1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .json {
      min-height: 0;
      overflow: auto;
    }
    pre {
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      color: var(--muted);
      font-weight: 600;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--warn);
    }
    .status.live .dot { background: var(--accent); }
    @media (max-width: 900px) {
      header { height: auto; min-height: 56px; align-items: flex-start; flex-direction: column; padding: 12px; }
      main { grid-template-columns: 1fr; grid-template-rows: minmax(280px, 58vh) auto; height: auto; padding: 12px; }
      aside { grid-template-rows: auto minmax(240px, 42vh); }
    }
  </style>
</head>
<body>
  <header>
    <h1>RK3588 Live Detection</h1>
    <div id="status" class="status"><span class="dot"></span><span>waiting</span></div>
  </header>
  <main>
    <section class="stage">
      <img src="/stream.mjpg" alt="live detection stream">
    </section>
    <aside>
      <section class="metrics">
        <div class="metric"><span>results</span><strong id="results">0</strong></div>
        <div class="metric"><span>images</span><strong id="images">0</strong></div>
        <div class="metric"><span>result fps</span><strong id="resultFps">0.0</strong></div>
        <div class="metric"><span>image fps</span><strong id="imageFps">0.0</strong></div>
        <div class="metric"><span>rx MiB</span><strong id="rxMib">0.0</strong></div>
        <div class="metric"><span>idle s</span><strong id="idle">-</strong></div>
      </section>
      <section class="json">
        <pre id="latest">{}</pre>
      </section>
    </aside>
  </main>
  <script>
    const fields = {
      results: document.getElementById("results"),
      images: document.getElementById("images"),
      resultFps: document.getElementById("resultFps"),
      imageFps: document.getElementById("imageFps"),
      rxMib: document.getElementById("rxMib"),
      idle: document.getElementById("idle"),
      latest: document.getElementById("latest"),
      status: document.getElementById("status")
    };

    function fmt(value, digits = 1) {
      return Number.isFinite(value) ? value.toFixed(digits) : "-";
    }

    async function refresh() {
      try {
        const response = await fetch("/api/latest", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        const stats = payload.stats || {};
        fields.results.textContent = stats.results ?? 0;
        fields.images.textContent = stats.images ?? 0;
        fields.resultFps.textContent = fmt(stats.result_fps);
        fields.imageFps.textContent = fmt(stats.image_fps);
        fields.rxMib.textContent = fmt(stats.rx_mib);
        fields.idle.textContent = stats.idle_seconds === null ? "-" : fmt(stats.idle_seconds);
        fields.latest.textContent = JSON.stringify(payload.latest_result || {}, null, 2);
        const live = payload.has_image && (stats.idle_seconds === null || stats.idle_seconds < 5);
        fields.status.className = live ? "status live" : "status";
        fields.status.lastElementChild.textContent = live ? "live" : "waiting";
      } catch (err) {
        fields.status.className = "status";
        fields.status.lastElementChild.textContent = "offline";
      }
    }

    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    state = LiveFrameState(
        save_latest=args.save_latest,
        save_dir=args.save_dir,
        save_dir_max_files=args.save_dir_max_files,
    )
    tcp_server = DetectionTCPServer(
        host=args.tcp_host,
        port=args.tcp_port,
        state=state,
        idle_timeout=args.idle_timeout,
        max_line_bytes=args.max_line_bytes,
    )
    tcp_thread = threading.Thread(target=tcp_server.serve_forever, name="tcp-receiver", daemon=True)
    tcp_thread.start()
    tcp_server.ready.wait(timeout=2.0)
    if tcp_server.start_error is not None:
        raise SystemExit(f"TCP receiver failed: {tcp_server.start_error}")

    httpd = LiveWebHTTPServer(
        (args.http_host, args.http_port),
        LiveWebHandler,
        state=state,
        max_mjpeg_clients=args.max_mjpeg_clients,
    )
    http_host, http_port = httpd.server_address[:2]
    LOGGER.info("Web viewer available at http://%s:%d", http_host, http_port)

    def monitor_runtime() -> None:
        while True:
            time.sleep(1.0)
            if not tcp_thread.is_alive():
                LOGGER.error("TCP receiver stopped; shutting down web viewer")
                httpd.shutdown()
                return
            if args.idle_timeout > 0 and tcp_server._idle_expired():
                LOGGER.info("Idle timeout reached; shutting down web viewer")
                tcp_server.stop()
                httpd.shutdown()
                return

    monitor_thread = threading.Thread(target=monitor_runtime, name="runtime-monitor", daemon=True)
    monitor_thread.start()

    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOGGER.info("Stopping live web viewer")
    finally:
        tcp_server.stop()
        httpd.server_close()
        tcp_thread.join(timeout=2.0)
        LOGGER.info("Stopped")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
