#!/usr/bin/env bash
set -euo pipefail

# Start the WSL-friendly live web demo:
#   board detect_cli -> board 127.0.0.1:9000 -> SSH reverse tunnel -> WSL web viewer.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

BOARD_HOST=${BOARD_HOST:-192.168.137.56}
BOARD_USER=${BOARD_USER:-root}
BOARD_ROOT=${BOARD_ROOT:-/root/rk-app-new}
MODE=${MODE:-fake}                 # fake|real
TRANSPORT=${TRANSPORT:-tunnel}     # tunnel|direct
HTTP_HOST=${HTTP_HOST:-127.0.0.1}
HTTP_PORT=${HTTP_PORT:-8080}
TCP_PORT=${TCP_PORT:-9000}
LOCAL_ARTIFACT_DIR=${LOCAL_ARTIFACT_DIR:-artifacts/live_view}
PYTHON=${PYTHON:-python3}
BOARD_LOG_LEVEL=${BOARD_LOG_LEVEL:-WARN}
RESTART_DELAY=${RESTART_DELAY:-1}
BOARD_JSON_OUT=${BOARD_JSON_OUT:-}
BOARD_JSON_MAX_BYTES=${BOARD_JSON_MAX_BYTES:-268435456}
DIRECT_HOST=${DIRECT_HOST:-192.168.137.1}
DIRECT_BIND_IP=${DIRECT_BIND_IP:-192.168.137.56}
DIRECT_BIND_INTERFACE=${DIRECT_BIND_INTERFACE:-eth1}

VIEWER_PID_FILE="$LOCAL_ARTIFACT_DIR/live_web_viewer.pid"
TUNNEL_PID_FILE="$LOCAL_ARTIFACT_DIR/live_web_tunnel.pid"
VIEWER_LOG="$LOCAL_ARTIFACT_DIR/live_web_viewer.log"
TUNNEL_LOG="$LOCAL_ARTIFACT_DIR/live_web_tunnel.log"
START_SUCCEEDED=0

usage() {
  cat <<EOF
Usage: $0 [--mode fake|real] [--transport tunnel|direct]

Environment overrides:
  BOARD_HOST=$BOARD_HOST
  BOARD_USER=$BOARD_USER
  BOARD_ROOT=$BOARD_ROOT
  HTTP_PORT=$HTTP_PORT
  TCP_PORT=$TCP_PORT
  BOARD_JSON_OUT=$BOARD_JSON_OUT
  BOARD_JSON_MAX_BYTES=$BOARD_JSON_MAX_BYTES

Default mode is WSL-safe:
  MODE=fake TRANSPORT=tunnel $0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --transport)
      TRANSPORT="$2"
      shift 2
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

if [[ "$MODE" != "fake" && "$MODE" != "real" ]]; then
  echo "MODE must be fake or real, got: $MODE" >&2
  exit 2
fi
if [[ "$TRANSPORT" != "tunnel" && "$TRANSPORT" != "direct" ]]; then
  echo "TRANSPORT must be tunnel or direct, got: $TRANSPORT" >&2
  exit 2
fi
if ! [[ "$BOARD_JSON_MAX_BYTES" =~ ^[0-9]+$ ]]; then
  echo "BOARD_JSON_MAX_BYTES must be an integer byte limit, got: $BOARD_JSON_MAX_BYTES" >&2
  exit 2
fi

mkdir -p "$LOCAL_ARTIFACT_DIR"

ssh_board() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$BOARD_USER@$BOARD_HOST" "$@"
}

kill_pid_file() {
  local file=$1
  if [[ -f "$file" ]]; then
    local pid
    pid=$(cat "$file" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 0.2
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

wait_for_http() {
  local url=$1
  "$PYTHON" - "$url" <<'PY'
import sys
import time
from urllib.request import urlopen

url = sys.argv[1]
deadline = time.time() + 8
while time.time() < deadline:
    try:
        with urlopen(url, timeout=1) as response:
            if response.status == 200:
                sys.exit(0)
    except Exception:
        time.sleep(0.2)
sys.exit(1)
PY
}

start_detached() {
  local pid_file=$1
  local log_file=$2
  shift 2

  if command -v setsid >/dev/null 2>&1; then
    # shellcheck disable=SC2016
    setsid -f bash -c '
      pid_file=$1
      log_file=$2
      shift 2
      echo $$ > "$pid_file"
      exec "$@" > "$log_file" 2>&1
    ' bash "$pid_file" "$log_file" "$@"
  else
    nohup "$@" > "$log_file" 2>&1 &
    echo $! > "$pid_file"
  fi
}

stop_previous() {
  kill_pid_file "$VIEWER_PID_FILE"
  kill_pid_file "$TUNNEL_PID_FILE"
  pkill -f "scripts/live_web_viewer.py .*--tcp-port ${TCP_PORT} .*--http-port ${HTTP_PORT}" 2>/dev/null || true
  pkill -f "ssh .* -R 127.0.0.1:${TCP_PORT}:127.0.0.1:${TCP_PORT} ${BOARD_USER}@${BOARD_HOST}" \
    2>/dev/null || true

  ssh_board "pkill -f 'artifacts/[r]un_live_web_demo.sh' 2>/dev/null || true; \
pkill -f 'detect_cli --cfg artifacts/[d]etect_fake_camera_live_web_tunnel.yaml' 2>/dev/null || true; \
pkill -f 'detect_cli --cfg artifacts/[d]etect_real_live_web_tunnel.yaml' 2>/dev/null || true" \
    >/dev/null 2>&1 || true
}

cleanup_on_error() {
  local code=$?
  local line_no=${1:-unknown}
  if [[ "$START_SUCCEEDED" -eq 0 ]]; then
    echo "start_live_web_demo failed near line ${line_no}; cleaning up partial demo state" >&2
    stop_previous >/dev/null 2>&1 || true
  fi
  exit "$code"
}

start_viewer() {
  if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "python not found: $PYTHON" >&2
    exit 1
  fi
  if [[ ! -f scripts/live_web_viewer.py ]]; then
    echo "Missing scripts/live_web_viewer.py" >&2
    exit 1
  fi

  start_detached "$VIEWER_PID_FILE" "$VIEWER_LOG" \
    "$PYTHON" scripts/live_web_viewer.py \
    --tcp-host 0.0.0.0 \
    --tcp-port "$TCP_PORT" \
    --http-host "$HTTP_HOST" \
    --http-port "$HTTP_PORT" \
    --save-latest "$LOCAL_ARTIFACT_DIR/latest.jpg" \
    --log-level INFO

  if ! wait_for_http "http://127.0.0.1:${HTTP_PORT}/healthz"; then
    echo "Web viewer did not become ready. Log:" >&2
    tail -40 "$VIEWER_LOG" >&2 || true
    exit 1
  fi
}

start_tunnel() {
  if [[ "$TRANSPORT" != "tunnel" ]]; then
    return 0
  fi

  start_detached "$TUNNEL_PID_FILE" "$TUNNEL_LOG" \
    ssh -N \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -R "127.0.0.1:${TCP_PORT}:127.0.0.1:${TCP_PORT}" \
    "$BOARD_USER@$BOARD_HOST"
  sleep 0.8

  if ! kill -0 "$(cat "$TUNNEL_PID_FILE")" 2>/dev/null; then
    echo "SSH reverse tunnel failed. Log:" >&2
    cat "$TUNNEL_LOG" >&2 || true
    exit 1
  fi

  if ! ssh_board "ss -ltn | grep -q '127.0.0.1:${TCP_PORT}'"; then
    echo "SSH reverse tunnel is running locally, but board is not listening on 127.0.0.1:${TCP_PORT}" >&2
    exit 1
  fi
}

prepare_board_config() {
  local base_cfg out_cfg target_host bind_ip bind_interface
  if [[ "$MODE" == "fake" ]]; then
    base_cfg="config/detection/detect_fake_camera.yaml"
    out_cfg="artifacts/detect_fake_camera_live_web_tunnel.yaml"
  else
    base_cfg="${REAL_CFG:-config/detection/detect_rknn.yaml}"
    out_cfg="artifacts/detect_real_live_web_tunnel.yaml"
  fi

  if [[ "$TRANSPORT" == "tunnel" ]]; then
    target_host="127.0.0.1"
    bind_ip=""
    bind_interface=""
  else
    target_host="$DIRECT_HOST"
    bind_ip="$DIRECT_BIND_IP"
    bind_interface="$DIRECT_BIND_INTERFACE"
  fi

  ssh_board "cd '$BOARD_ROOT' && mkdir -p artifacts && \
BASE_CFG='$base_cfg' OUT_CFG='$out_cfg' TARGET_HOST='$target_host' \
BIND_IP='$bind_ip' BIND_INTERFACE='$bind_interface' python3 - <<'PY'
import os
from pathlib import Path

base = Path(os.environ['BASE_CFG'])
out = Path(os.environ['OUT_CFG'])
host = os.environ['TARGET_HOST']
bind_ip = os.environ['BIND_IP']
bind_interface = os.environ['BIND_INTERFACE']

lines = base.read_text(encoding='utf-8').splitlines()
updated = []
in_output = False
in_tcp = False
output_indent = -1
tcp_indent = -1
touched = set()

for line in lines:
    stripped = line.strip()
    indent = line[: len(line) - len(line.lstrip())]
    indent_len = len(indent)

    if stripped and not stripped.startswith('#'):
        if in_tcp and indent_len <= tcp_indent:
            in_tcp = False
        if in_output and indent_len <= output_indent and stripped != 'output:':
            in_output = False
            in_tcp = False

    if stripped == 'output:' and not stripped.startswith('#'):
        in_output = True
        in_tcp = False
        output_indent = indent_len
    elif in_output and stripped == 'tcp:' and indent_len > output_indent:
        in_tcp = True
        tcp_indent = indent_len

    if in_tcp and indent_len > tcp_indent and stripped.startswith('host: '):
        updated.append(f'{indent}host: \"{host}\"')
        touched.add('host')
    elif in_tcp and indent_len > tcp_indent and stripped.startswith('bind_ip: '):
        updated.append(f'{indent}bind_ip: \"{bind_ip}\"')
        touched.add('bind_ip')
    elif in_tcp and indent_len > tcp_indent and stripped.startswith('bind_interface: '):
        updated.append(f'{indent}bind_interface: \"{bind_interface}\"')
        touched.add('bind_interface')
    elif in_tcp and indent_len > tcp_indent and stripped.startswith('include_image: '):
        updated.append(f'{indent}include_image: true')
        touched.add('include_image')
    else:
        updated.append(line)

missing = {'host', 'bind_ip', 'bind_interface', 'include_image'} - touched
if missing:
    raise SystemExit(f'{base} is missing output.tcp keys: {sorted(missing)}')

out.write_text('\\n'.join(updated) + '\\n', encoding='utf-8')
print(out)
PY"
}

install_board_runner() {
  ssh_board "cd '$BOARD_ROOT' && cat > artifacts/run_live_web_demo.sh && chmod +x artifacts/run_live_web_demo.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CFG=${CFG:-artifacts/detect_fake_camera_live_web_tunnel.yaml}
JSON_OUT=${JSON_OUT:-}
JSON_MAX_BYTES=${JSON_MAX_BYTES:-268435456}
BOARD_LOG_LEVEL=${BOARD_LOG_LEVEL:-WARN}
RESTART_DELAY=${RESTART_DELAY:-1}

while true; do
  cmd=(./build/board/detect_cli --cfg "$CFG" --log-level "$BOARD_LOG_LEVEL")
  if [[ -n "$JSON_OUT" ]]; then
    if ! [[ "$JSON_MAX_BYTES" =~ ^[0-9]+$ ]]; then
      echo "JSON_MAX_BYTES must be an integer byte limit, got: $JSON_MAX_BYTES" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$JSON_OUT")"
    if [[ -f "$JSON_OUT" ]]; then
      current_size=$(wc -c < "$JSON_OUT" 2>/dev/null || echo 0)
      if (( current_size >= JSON_MAX_BYTES )); then
        echo "truncating $JSON_OUT at ${current_size} bytes (limit ${JSON_MAX_BYTES})" >&2
        : > "$JSON_OUT"
      fi
    fi
    cmd+=(--json "$JSON_OUT")
  fi
  set +e
  "${cmd[@]}"
  code=$?
  set -e
  echo "detect_cli exited with code $code, restarting in ${RESTART_DELAY}s" >&2
  sleep "$RESTART_DELAY"
done
EOS
}

start_board_runner() {
  local cfg json_out log_out
  if [[ "$MODE" == "fake" ]]; then
    cfg="artifacts/detect_fake_camera_live_web_tunnel.yaml"
    log_out="artifacts/live_web_fake_camera_loop.log"
  else
    cfg="artifacts/detect_real_live_web_tunnel.yaml"
    log_out="artifacts/live_web_real_camera_loop.log"
  fi
  json_out="$BOARD_JSON_OUT"

  ssh_board "cd '$BOARD_ROOT' && \
(CFG='$cfg' JSON_OUT='$json_out' JSON_MAX_BYTES='$BOARD_JSON_MAX_BYTES' \
BOARD_LOG_LEVEL='$BOARD_LOG_LEVEL' RESTART_DELAY='$RESTART_DELAY' \
nohup bash artifacts/run_live_web_demo.sh > '$log_out' 2>&1 < /dev/null & \
echo \$! > artifacts/live_web_demo_loop.pid)"
  sleep 1
}

print_summary() {
  local browser_url="http://localhost:${HTTP_PORT}"
  cat <<EOF
Live web demo started.

Browser:
  $browser_url

Local:
  viewer pid: $(cat "$VIEWER_PID_FILE" 2>/dev/null || echo "-")
  viewer log: $VIEWER_LOG
  tunnel pid: $(cat "$TUNNEL_PID_FILE" 2>/dev/null || echo "-")

Board:
  ssh: $BOARD_USER@$BOARD_HOST
  root: $BOARD_ROOT
  mode: $MODE
  transport: $TRANSPORT
  json log: ${BOARD_JSON_OUT:-disabled}

Use:
  scripts/demo/status_live_web_demo.sh
  scripts/demo/stop_live_web_demo.sh
EOF
}

stop_previous
trap 'cleanup_on_error "$LINENO"' ERR
start_viewer
start_tunnel
prepare_board_config
install_board_runner
start_board_runner
START_SUCCEEDED=1
trap - ERR
print_summary
