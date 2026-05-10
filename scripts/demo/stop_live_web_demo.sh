#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

BOARD_HOST=${BOARD_HOST:-192.168.137.226}
BOARD_USER=${BOARD_USER:-root}
BOARD_ROOT=${BOARD_ROOT:-/opt/rk_app_current}
TCP_PORT=${TCP_PORT:-9000}
HTTP_PORT=${HTTP_PORT:-8080}
LOCAL_ARTIFACT_DIR=${LOCAL_ARTIFACT_DIR:-artifacts/live_view}

VIEWER_PID_FILE="$LOCAL_ARTIFACT_DIR/live_web_viewer.pid"
TUNNEL_PID_FILE="$LOCAL_ARTIFACT_DIR/live_web_tunnel.pid"

LOCAL_ONLY=0
BOARD_ONLY=0

usage() {
  cat <<EOF
Usage: $0 [--local-only|--board-only]

Stops the local web viewer, SSH reverse tunnel, and board-side demo runner.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-only)
      LOCAL_ONLY=1
      shift
      ;;
    --board-only)
      BOARD_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ssh_board() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$BOARD_USER@$BOARD_HOST" "$@"
}

kill_pid_file() {
  local file=$1
  if [[ ! -f "$file" ]]; then
    return 0
  fi

  local pid
  pid=$(cat "$file" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping pid $pid from $file"
    kill "$pid" 2>/dev/null || true
    sleep 0.3
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$file"
}

stop_local() {
  kill_pid_file "$VIEWER_PID_FILE"
  kill_pid_file "$TUNNEL_PID_FILE"

  pkill -f "scripts/live_web_viewer.py .*--tcp-port ${TCP_PORT} .*--http-port ${HTTP_PORT}" \
    2>/dev/null || true
  pkill -f "ssh .* -R 127.0.0.1:${TCP_PORT}:127.0.0.1:${TCP_PORT} ${BOARD_USER}@${BOARD_HOST}" \
    2>/dev/null || true
}

stop_board() {
  ssh_board "pkill -f 'artifacts/[r]un_live_web_demo.sh' 2>/dev/null || true; \
pkill -f 'detect_cli --cfg artifacts/[d]etect_fake_camera_live_web_tunnel.yaml' 2>/dev/null || true; \
pkill -f 'detect_cli --cfg artifacts/[d]etect_real_live_web_tunnel.yaml' 2>/dev/null || true; \
rm -f '$BOARD_ROOT/artifacts/live_web_demo_loop.pid' 2>/dev/null || true" \
    >/dev/null 2>&1 || true
}

if [[ "$BOARD_ONLY" -eq 0 ]]; then
  stop_local
fi
if [[ "$LOCAL_ONLY" -eq 0 ]]; then
  stop_board
fi

echo "Live web demo stopped."
