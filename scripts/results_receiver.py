#!/usr/bin/env python3
"""
Results aggregation server for detection output.

Receives detection results from RK3588 board and stores them. When the sender
embeds JPEG image payloads in the JSON stream, the server also decodes and
saves those frames alongside the detection metadata.
Used in docker-compose.dual-nic.yml for results_server service.

Usage:
    python3 results_receiver.py --host 192.168.137.1 --port 9000

Environment variables:
    LISTEN_HOST: TCP server bind address (default: 0.0.0.0)
    LISTEN_PORT: TCP server port (default: 9000)
    RESULTS_DIR: Directory to store results (default: artifacts/received_results)
"""

import argparse
import os
import sys
import json
import time
import base64
import socket
import logging
import threading
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsServer:
    """TCP server for receiving detection results."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9000, output_dir: str = "/artifacts"):
        """Initialize results server.

        Args:
            host: Bind address
            port: TCP port
            output_dir: Directory to save results
        """
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.socket = None
        self.running = False
        self.result_count = 0
        self._count_lock = threading.Lock()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start TCP server (blocking)."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)

            self.running = True
            logger.info(f"Results server listening on {self.host}:{self.port}")
            logger.info(f"Output directory: {self.output_dir}")

            while self.running:
                try:
                    logger.info("Waiting for connection...")
                    client_socket, client_address = self.socket.accept()
                    logger.info(f"Connection from {client_address}")

                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except socket.timeout:
                    continue
                except (OSError, ConnectionError) as e:
                    logger.error(f"Accept error: {e}")
                    break

        except (OSError, socket.error) as e:
            logger.error(f"Server error: {e}")
            sys.exit(1)
        finally:
            self.stop()

    def handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle individual client connection.

        Args:
            client_socket: Connected socket
            client_address: Client address tuple
        """
        try:
            # Set timeout for receiving data
            client_socket.settimeout(30)

            # Receive newline-delimited JSON incrementally to avoid buffering
            # long-running image streams in memory.
            buffer = bytearray()
            while True:
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while True:
                        newline_pos = buffer.find(b"\n")
                        if newline_pos < 0:
                            break
                        line = bytes(buffer[:newline_pos]).strip()
                        del buffer[:newline_pos + 1]
                        if line:
                            self.process_results(line, client_address)
                except socket.timeout:
                    break

            if buffer:
                self.process_results(bytes(buffer).strip(), client_address)

            # Send ACK
            client_socket.send(b"OK")

        except (OSError, socket.error, UnicodeDecodeError) as e:
            logger.error(f"Error handling client {client_address}: {e}")
        finally:
            client_socket.close()

    def process_results(self, data: bytes, client_address: tuple):
        """Process received detection results.

        Args:
            data: Raw data from client
            client_address: Source address
        """
        try:
            # Try to parse as JSON
            result_str = data.decode('utf-8').strip()

            # Handle multiple JSON objects (one per line)
            for line in result_str.split('\n'):
                if not line.strip():
                    continue

                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_address}: {line[:100]}")
                    continue

                # Save result
                self.save_result(result)

        except UnicodeDecodeError:
            logger.warning(f"Non-UTF8 data from {client_address}, size: {len(data)} bytes")

    def save_result(self, result: dict):
        """Save single detection result.

        Args:
            result: Detection result dictionary
        """
        try:
            with self._count_lock:
                self.result_count += 1
                count_snapshot = self.result_count

            # Create result file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_id = result.get("frame_id", count_snapshot)
            sanitized_result = dict(result)
            image_filename = None
            image_info = result.get("image")
            if isinstance(image_info, dict):
                image_filename = self._save_image_payload(
                    image_info=image_info,
                    timestamp=timestamp,
                    frame_id=frame_id,
                )
                if image_filename is not None:
                    image_meta = {k: v for k, v in image_info.items() if k != "data_base64"}
                    image_meta["path"] = image_filename
                    sanitized_result["image"] = image_meta

            filename = f"result_{timestamp}_frame{frame_id}.json"
            filepath = self.output_dir / filename

            # Save to file
            with open(filepath, "w") as f:
                json.dump(sanitized_result, f, indent=2)

            # Log summary
            detections = len(result.get("detections", []))
            latency = result.get("latency_ms", 0)
            image_log = f", image={image_filename}" if image_filename else ""
            logger.info(
                f"Result #{count_snapshot}: {detections} detections, "
                f"latency={latency:.1f}ms{image_log} → {filename}"
            )

        except (IOError, OSError, TypeError) as e:
            logger.error(f"Error saving result: {e}")

    def _save_image_payload(self, image_info: dict, timestamp: str, frame_id: int):
        """Persist base64-encoded image payload if present.

        Args:
            image_info: Image object from result JSON
            timestamp: Timestamp token used for output naming
            frame_id: Frame index

        Returns:
            Relative image filename when saved successfully, else None.
        """
        encoded = image_info.get("data_base64")
        encoding = image_info.get("encoding", "jpeg")
        if not encoded or encoding.lower() not in {"jpg", "jpeg"}:
            return None

        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid base64 image payload for frame {frame_id}: {e}")
            return None

        image_filename = f"result_{timestamp}_frame{frame_id}.jpg"
        image_path = self.output_dir / image_filename
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        return image_filename

    def stop(self):
        """Stop server gracefully."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except (OSError, socket.error) as e:
                logger.error(f"Error closing socket: {e}")
        logger.info(f"Server stopped. Total results received: {self.result_count}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("LISTEN_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LISTEN_PORT", "9000")))
    parser.add_argument(
        "--output-dir",
        default=os.getenv("RESULTS_DIR", "artifacts/received_results"),
        help="Directory to store JSON results and optional JPEG frames",
    )
    parser.add_argument("--health-host", default=os.getenv("HEALTH_HOST", "0.0.0.0"))
    parser.add_argument("--health-port", type=int, default=int(os.getenv("HEALTH_PORT", "8080")))
    parser.add_argument("--no-health", action="store_true", help="Disable HTTP health endpoint")
    return parser.parse_args()


def health_check_server(host: str, port: int):
    """Run simple HTTP health check server (for Docker healthcheck)."""
    try:
        import http.server
        import socketserver

        class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    response = json.dumps({
                        "status": "healthy",
                        "timestamp": datetime.now().isoformat()
                    })
                    self.wfile.write(response.encode())
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                # Suppress access logs
                pass

        handler = HealthCheckHandler
        httpd = socketserver.TCPServer((host, port), handler)
        logger.info(f"Health check HTTP server running on {host}:{port}")
        httpd.serve_forever()

    except (OSError, socket.error) as e:
        logger.error(f"Health check server error: {e}")


def main():
    args = parse_args()

    logger.info("Results Server Configuration:")
    logger.info(f"  Listen: {args.host}:{args.port}")
    logger.info(f"  Results Dir: {args.output_dir}")

    if not args.no_health:
        health_thread = threading.Thread(
            target=health_check_server,
            args=(args.health_host, args.health_port),
            daemon=True,
        )
        health_thread.start()

    # Start main results server
    server = ResultsServer(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir
    )
    server.start()


if __name__ == "__main__":
    main()
