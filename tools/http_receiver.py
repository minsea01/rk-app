#!/usr/bin/env python3
import argparse, json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('content-length', '0'))
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            data = {'raw': body.decode('utf-8', errors='ignore')}
        print(json.dumps({'path': self.path, 'data': data}, ensure_ascii=False))
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
