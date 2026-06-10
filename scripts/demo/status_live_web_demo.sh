#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

BOARD_HOST=${BOARD_HOST:-192.168.137.226}
BOARD_USER=${BOARD_USER:-root}
BOARD_ROOT=${BOARD_ROOT:-/opt/rk_app_current}
HTTP_PORT=${HTTP_PORT:-8080}
TCP_PORT=${TCP_PORT:-9000}
LOCAL_ARTIFACT_DIR=${LOCAL_ARTIFACT_DIR:-artifacts/live_view}
PYTHON=${PYTHON:-python3}

VIEWER_PID_FILE="$LOCAL_ARTIFACT_DIR/live_web_viewer.pid"
TUNNEL_PID_FILE="$LOCAL_ARTIFACT_DIR/live_web_tunnel.pid"

pid_status() {
  local label=$1
  local file=$2
  local pid="-"
  if [[ -f "$file" ]]; then
    pid=$(cat "$file" 2>/dev/null || echo "-")
  fi

  if [[ "$pid" != "-" && -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$label: running pid=$pid"
  else
    echo "$label: stopped"
  fi
}

echo "Local status"
pid_status "  web viewer" "$VIEWER_PID_FILE"
pid_status "  ssh tunnel" "$TUNNEL_PID_FILE"
ss -ltnp 2>/dev/null | grep -E ":(${TCP_PORT}|${HTTP_PORT})" || true

echo
echo "Latest stream stats"
if command -v "$PYTHON" >/dev/null 2>&1; then
  "$PYTHON" - "$HTTP_PORT" <<'PY' || true
import json
import sys
from urllib.request import urlopen

port = sys.argv[1]
payload = json.loads(urlopen(f"http://127.0.0.1:{port}/api/latest", timeout=2).read().decode())
stats = payload.get("stats", {})
latest = payload.get("latest_result") or {}
print(f"  browser: http://localhost:{port}")
print(f"  frame_id: {payload.get('latest_frame_id')}")
print(f"  results: {stats.get('results')}")
print(f"  images: {stats.get('images')}")
idle = stats.get("idle_seconds")
print(f"  idle_s: {'-' if idle is None else round(idle, 2)}")
print(f"  detections_latest: {len(latest.get('detections', []))}")
print(f"  latest_image: {payload.get('latest_saved_path')}")
PY
else
  echo "  python not found: $PYTHON"
fi

echo
echo "Board status"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$BOARD_USER@$BOARD_HOST" \
  "echo '  host: '\"\$(hostname)\"; \
echo '  tunnel/listen:'; ss -ltnp | grep ':${TCP_PORT}' || true; \
echo '  demo processes:'; ps -ef | grep -E 'detect_cli|run_live_web_demo' | grep -v grep || true; \
echo '  recent log:'; tail -12 '$BOARD_ROOT/artifacts/live_web_fake_camera_loop.log' 2>/dev/null || tail -12 '$BOARD_ROOT/artifacts/live_web_real_camera_loop.log' 2>/dev/null || true" \
  || echo "  board unreachable: $BOARD_USER@$BOARD_HOST"
