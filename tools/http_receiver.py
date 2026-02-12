#!/usr/bin/env python3
"""HTTP receiver for MCP pipeline validation.

This script runs a simple HTTP server to receive and log POST requests,
used for testing data upload and network communication.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# Import custom exceptions and logger
from apps.exceptions import ConfigurationError
from apps.logger import setup_logger

# Setup logger
logger = setup_logger(__name__, level="INFO")


class Handler(BaseHTTPRequestHandler):
    # Maximum allowed payload size (10MB)
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    def __init__(self, request, client_address, server):
        # Allow unit tests to instantiate handler without a running server
        if server is None:
            self.request = request
            self.client_address = client_address
            self.server = server
            self.rfile = getattr(request, "rfile", None)
            self.wfile = getattr(request, "wfile", None)
            self.headers = {}
            self.path = "/"
            self.request_version = "HTTP/1.1"
            self.command = "POST"
            return
        super().__init__(request, client_address, server)

    def do_POST(self):
        # Validate Content-Length header
        try:
            length = int(self.headers.get("content-length", "0"))
        except (ValueError, TypeError):
            self.send_response(400)  # Bad Request
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"Invalid Content-Length header"}')
            return

        # Check payload size limit
        if length > self.MAX_CONTENT_LENGTH:
            self.send_response(413)  # Payload Too Large
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"Payload too large"}')
            return

        # Read and normalize body
        if length > 0:
            raw_body = self.rfile.read(length)
        else:
            raw_body = b""

        if isinstance(raw_body, (bytes, bytearray)):
            body = bytes(raw_body)
        elif raw_body is None:
            body = b""
        else:
            body = str(raw_body).encode("utf-8", errors="ignore")

        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            # Not valid JSON, store as raw text
            data = {"raw": body.decode("utf-8", errors="ignore"), "parse_error": str(e)}
        except UnicodeDecodeError as e:
            # Binary data or invalid encoding
            data = {"raw": body.hex(), "encoding_error": str(e)}

        # Log the received data
        logger.info(json.dumps({"path": self.path, "data": data}, ensure_ascii=False))

        # Send success response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')


def main():
    ap = argparse.ArgumentParser(description="HTTP receiver for MCP pipeline testing")
    ap.add_argument("--port", type=int, default=8081, help="Port to listen on (0 for auto)")
    args = ap.parse_args()

    try:
        httpd = HTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        if e.errno == 98:  # Address already in use
            raise ConfigurationError(f"Port {args.port} already in use") from e
        elif e.errno == 13:  # Permission denied
            raise ConfigurationError(
                f"Permission denied for port {args.port} (requires root?)"
            ) from e
        else:
            raise ConfigurationError(f"Failed to create HTTP server: {e}") from e

    port = httpd.server_address[1]
    # Print a startup line so callers can discover the port when using 0
    # This print() is intentional - used for programmatic port discovery
    print(json.dumps({"listening_port": port}), flush=True)
    logger.info(f"HTTP receiver listening on 127.0.0.1:{port}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down HTTP receiver")
        httpd.shutdown()
        return 0
    except (OSError, IOError) as e:
        raise ConfigurationError(f"HTTP server error: {e}") from e


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigurationError as e:
        logger.error(f"HTTP receiver failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
