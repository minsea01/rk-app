#!/usr/bin/env python3
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    # Maximum allowed payload size (10MB)
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    def do_POST(self):
        # Validate Content-Length header
        try:
            length = int(self.headers.get('content-length', '0'))
        except (ValueError, TypeError):
            self.send_response(400)  # Bad Request
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error":"Invalid Content-Length header"}')
            return

        # Check payload size limit
        if length > self.MAX_CONTENT_LENGTH:
            self.send_response(413)  # Payload Too Large
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error":"Payload too large"}')
            return

        # Read and parse body
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as e:
            # Not valid JSON, store as raw text
            data = {'raw': body.decode('utf-8', errors='ignore'), 'parse_error': str(e)}
        except UnicodeDecodeError as e:
            # Binary data or invalid encoding
            data = {'raw': body.hex(), 'encoding_error': str(e)}

        # Log the received data
        print(json.dumps({'path': self.path, 'data': data}, ensure_ascii=False))

        # Send success response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8081)
    args = ap.parse_args()
    httpd = HTTPServer(('127.0.0.1', args.port), Handler)
    port = httpd.server_address[1]
    # Print a startup line so callers can discover the port when using 0
    print(json.dumps({'listening_port': port}), flush=True)
    httpd.serve_forever()

if __name__ == '__main__':
    main()
