#!/usr/bin/env python3
"""
Results aggregation server for detection output.

Receives detection results from RK3588 board and stores them.
Used in docker-compose.dual-nic.yml for results_server service.

Usage:
    python3 results_receiver.py

Environment variables:
    LISTEN_HOST: TCP server bind address (default: 0.0.0.0)
    LISTEN_PORT: TCP server port (default: 9000)
    RESULTS_DIR: Directory to store results (default: /artifacts)
"""

import os
import sys
import json
import time
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

            # Receive data
            buffer = b""
            while True:
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                except socket.timeout:
                    break

            if buffer:
                self.process_results(buffer, client_address)

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
            self.result_count += 1

            # Create result file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_id = result.get("frame_id", self.result_count)
            filename = f"result_{timestamp}_frame{frame_id}.json"
            filepath = self.output_dir / filename

            # Save to file
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)

            # Log summary
            detections = len(result.get("detections", []))
            latency = result.get("latency_ms", 0)
            logger.info(
                f"Result #{self.result_count}: {detections} detections, "
                f"latency={latency:.1f}ms → {filename}"
            )

        except (IOError, OSError, TypeError) as e:
            logger.error(f"Error saving result: {e}")

    def stop(self):
        """Stop server gracefully."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except (OSError, socket.error) as e:
                logger.error(f"Error closing socket: {e}")
        logger.info(f"Server stopped. Total results received: {self.result_count}")


def health_check_server():
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
        httpd = socketserver.TCPServer(("0.0.0.0", 8080), handler)
        logger.info("Health check HTTP server running on 0.0.0.0:8080")
        httpd.serve_forever()

    except (OSError, socket.error) as e:
        logger.error(f"Health check server error: {e}")


def main():
    # Get config from environment
    listen_host = os.getenv('LISTEN_HOST', '0.0.0.0')
    listen_port = int(os.getenv('LISTEN_PORT', '9000'))
    results_dir = os.getenv('RESULTS_DIR', '/artifacts')

    logger.info("Results Server Configuration:")
    logger.info(f"  Listen: {listen_host}:{listen_port}")
    logger.info(f"  Results Dir: {results_dir}")

    # Start health check server in background
    health_thread = threading.Thread(target=health_check_server, daemon=True)
    health_thread.start()

    # Start main results server
    server = ResultsServer(
        host=listen_host,
        port=listen_port,
        output_dir=results_dir
    )
    server.start()


if __name__ == "__main__":
    main()
