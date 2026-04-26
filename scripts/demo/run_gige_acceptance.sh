#!/usr/bin/env bash
set -euo pipefail

# Host-side one-command GigE acceptance runner.
# It syncs board time, optionally tests eth1 throughput, starts a TCP sink,
# runs the board-side GigE preparation script, and pulls evidence back.

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT"

BOARD_HOST="${BOARD_HOST:-192.168.137.56}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_ROOT="${BOARD_ROOT:-/opt/rk_app_current}"
PC_HOST="${PC_HOST:-192.168.137.1}"
PC_PORT="${PC_PORT:-9000}"
CAMERA_NAME="${CAMERA_NAME:-auto}"
CAMERA_IFACE="${CAMERA_IFACE:-eth0}"
CAMERA_ADDR="${CAMERA_ADDR:-192.168.1.10/24}"
UPLOAD_ADDR="${UPLOAD_ADDR:-192.168.137.56/24}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1200}"
FPS="${FPS:-30}"
FORMAT="${FORMAT:-BGR}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.55}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.35}"
IMAGE_INTERVAL="${IMAGE_INTERVAL:-3}"
INCLUDE_IMAGE="${INCLUDE_IMAGE:-true}"
GRAB_FRAMES=0
RUN_DETECT_SECONDS=0
EXPECT_CAMERA=0
APPLY_NETWORK=0
RUN_IPERF=0
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVIDENCE_DIR="${EVIDENCE_DIR:-artifacts/gige_acceptance_${TIMESTAMP}}"

usage() {
  cat <<EOF
Usage: $0 [options]

Common:
  --software-only            Readiness checks only; no camera required (default)
  --full                     Apply eth0 config, expect camera, grab 30 frames, detect 30s, run iperf
  --expect-camera            Treat camera link/discovery/grab failures as failures
  --apply-network            Configure board eth0 before checks
  --grab <frames>            Grab N frames through aravissrc
  --run-detect <seconds>     Run detect_cli for N seconds
  --iperf                    Run eth1 iperf3 host receiver test

Board:
  --host <ip>                Board IP (default: ${BOARD_HOST})
  --user <name>              SSH user (default: ${BOARD_USER})
  --board-root <path>        Runtime dir (default: ${BOARD_ROOT})

Camera:
  --camera-name <name|auto>  Aravis camera name; auto selects first camera (default: ${CAMERA_NAME})
  --camera-addr <cidr>       Board camera NIC address (default: ${CAMERA_ADDR})
  --upload-addr <cidr>       Board upload NIC address (default: ${UPLOAD_ADDR})
  --format <fmt>             BGR, BayerRG8, Mono8, ... (default: ${FORMAT})
  --width <px>               Capture width (default: ${WIDTH})
  --height <px>              Capture height (default: ${HEIGHT})
  --fps <n>                  Capture FPS (default: ${FPS})

Detection/output:
  --conf-threshold <v>       Confidence threshold (default: ${CONF_THRESHOLD})
  --nms-threshold <v>        NMS threshold (default: ${NMS_THRESHOLD})
  --include-image <0|1>      Include JPEGs in TCP output (default: ${INCLUDE_IMAGE})
  --image-interval <n>       JPEG interval (default: ${IMAGE_INTERVAL})
  --evidence-dir <path>      Local evidence dir (default: ${EVIDENCE_DIR})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --software-only) EXPECT_CAMERA=0; APPLY_NETWORK=0; GRAB_FRAMES=0; RUN_DETECT_SECONDS=0; RUN_IPERF=0; shift ;;
    --full) EXPECT_CAMERA=1; APPLY_NETWORK=1; GRAB_FRAMES=30; RUN_DETECT_SECONDS=30; RUN_IPERF=1; shift ;;
    --expect-camera) EXPECT_CAMERA=1; shift ;;
    --apply-network) APPLY_NETWORK=1; shift ;;
    --grab) GRAB_FRAMES="$2"; shift 2 ;;
    --run-detect) RUN_DETECT_SECONDS="$2"; shift 2 ;;
    --iperf) RUN_IPERF=1; shift ;;
    --host) BOARD_HOST="$2"; shift 2 ;;
    --user) BOARD_USER="$2"; shift 2 ;;
    --board-root) BOARD_ROOT="$2"; shift 2 ;;
    --camera-name) CAMERA_NAME="$2"; shift 2 ;;
    --camera-addr) CAMERA_ADDR="$2"; shift 2 ;;
    --upload-addr) UPLOAD_ADDR="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --conf-threshold) CONF_THRESHOLD="$2"; shift 2 ;;
    --nms-threshold) NMS_THRESHOLD="$2"; shift 2 ;;
    --include-image)
      case "$2" in
        1|true|TRUE|yes|YES|on|ON) INCLUDE_IMAGE=true ;;
        0|false|FALSE|no|NO|off|OFF) INCLUDE_IMAGE=false ;;
        *) echo "--include-image expects 0/1 or true/false, got: $2" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --image-interval) IMAGE_INTERVAL="$2"; shift 2 ;;
    --evidence-dir) EVIDENCE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$EVIDENCE_DIR"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${BOARD_USER}@${BOARD_HOST}")
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no)

cleanup_pids=()
# shellcheck disable=SC2329  # Invoked by the EXIT trap below.
cleanup() {
  for pid in "${cleanup_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "[gige] evidence: $EVIDENCE_DIR"
"$ROOT/scripts/deploy/sync_board_clock.sh" --host "$BOARD_HOST" --user "$BOARD_USER" \
  > "$EVIDENCE_DIR/clock_sync.log" 2>&1

"${SSH[@]}" "cd '$BOARD_ROOT' && hostname; date; ip -br addr show dev eth0; ip -br addr show dev eth1; cat DEPLOYMENT_MANIFEST.txt" \
  > "$EVIDENCE_DIR/board_status.txt" 2>&1 || true

if [[ "$RUN_IPERF" -eq 1 ]]; then
  if command -v iperf3 >/dev/null 2>&1; then
    board_upload_ip="${UPLOAD_ADDR%%/*}"
    iperf3 -s -1 -B "$PC_HOST" -p 5201 > "$EVIDENCE_DIR/iperf_eth1_server.log" 2>&1 &
    cleanup_pids+=("$!")
    sleep 0.8
    "${SSH[@]}" "iperf3 -c '$PC_HOST' -B '$board_upload_ip' -p 5201 -t 10 -i 1" \
      > "$EVIDENCE_DIR/iperf_eth1_client.log" 2>&1 || true
  else
    echo "iperf3 missing on host" > "$EVIDENCE_DIR/iperf_eth1_client.log"
  fi
fi

if [[ "$RUN_DETECT_SECONDS" -gt 0 ]]; then
  nc -l -p "$PC_PORT" > "$EVIDENCE_DIR/tcp_sink_results.jsonl" 2>"$EVIDENCE_DIR/tcp_sink.err" &
  cleanup_pids+=("$!")
  sleep 0.5
else
  : > "$EVIDENCE_DIR/tcp_sink_results.jsonl"
fi

board_config="artifacts/gige_acceptance_effective.yaml"
board_cmd=(
  scripts/deploy/prepare_hikrobot_gige.sh
  --camera-iface "$CAMERA_IFACE"
  --camera-addr "$CAMERA_ADDR"
  --camera-name "$CAMERA_NAME"
  --upload-addr "$UPLOAD_ADDR"
  --pc-host "$PC_HOST"
  --pc-port "$PC_PORT"
  --width "$WIDTH"
  --height "$HEIGHT"
  --fps "$FPS"
  --format "$FORMAT"
  --conf-threshold "$CONF_THRESHOLD"
  --nms-threshold "$NMS_THRESHOLD"
  --image-interval "$IMAGE_INTERVAL"
  --include-image "$INCLUDE_IMAGE"
  --generate-config "$board_config"
)
if [[ "$EXPECT_CAMERA" -eq 1 ]]; then board_cmd+=(--expect-camera); fi
if [[ "$APPLY_NETWORK" -eq 1 ]]; then board_cmd+=(--apply-network); fi
if [[ "$GRAB_FRAMES" -gt 0 ]]; then board_cmd+=(--grab "$GRAB_FRAMES"); fi
if [[ "$RUN_DETECT_SECONDS" -gt 0 ]]; then board_cmd+=(--run-detect "$RUN_DETECT_SECONDS"); fi

remote_board_cmd="$(printf '%q ' "${board_cmd[@]}")"
printf '[gige] board command: %s\n' "$remote_board_cmd" | tee "$EVIDENCE_DIR/command.txt"

set +e
"${SSH[@]}" "cd '$BOARD_ROOT' && $remote_board_cmd" \
  > "$EVIDENCE_DIR/gige_acceptance_board.log" 2>&1
board_status=$?
set -e

if [[ "$RUN_DETECT_SECONDS" -gt 0 ]]; then
  sleep 0.5
fi

"${SCP[@]}" "${BOARD_USER}@${BOARD_HOST}:${BOARD_ROOT}/artifacts/gige_acceptance_effective.yaml" \
  "$EVIDENCE_DIR/effective_config.yaml" 2>/dev/null || true
if [[ "$RUN_DETECT_SECONDS" -gt 0 ]]; then
  "${SCP[@]}" "${BOARD_USER}@${BOARD_HOST}:${BOARD_ROOT}/artifacts/hikrobot_mv_ca020_20gc_${RUN_DETECT_SECONDS}s.log" \
    "$EVIDENCE_DIR/board_detect.log" 2>/dev/null || true
  "${SCP[@]}" "${BOARD_USER}@${BOARD_HOST}:${BOARD_ROOT}/artifacts/hikrobot_mv_ca020_20gc_${RUN_DETECT_SECONDS}s.json" \
    "$EVIDENCE_DIR/board_detect_results.jsonl" 2>/dev/null || true
fi

python3 "$ROOT/tools/generate_gige_acceptance_report.py" --evidence-dir "$EVIDENCE_DIR" \
  > "$EVIDENCE_DIR/report_generator.log"

echo "[gige] report: $EVIDENCE_DIR/REPORT.md"
echo "[gige] summary: $EVIDENCE_DIR/summary.json"
exit "$board_status"
