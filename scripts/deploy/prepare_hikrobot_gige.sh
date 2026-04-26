#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

CAMERA_IFACE="${CAMERA_IFACE:-eth0}"
CAMERA_ADDR="${CAMERA_ADDR:-192.168.1.10/24}"
CAMERA_NAME="${CAMERA_NAME:-MV-CA020-20GC}"
UPLOAD_IFACE="${UPLOAD_IFACE:-eth1}"
UPLOAD_ADDR="${UPLOAD_ADDR:-192.168.137.56/24}"
PC_HOST="${PC_HOST:-192.168.137.1}"
PC_PORT="${PC_PORT:-9000}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1200}"
FPS="${FPS:-30}"
FORMAT="${FORMAT:-BGR}"
MTU="${MTU:-1500}"
CONFIG="${CONFIG:-config/detection/detect_hikrobot_mv_ca020_20gc.yaml}"
BINARY="${BINARY:-build/board/detect_cli}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.40}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.35}"
IMAGE_INTERVAL="${IMAGE_INTERVAL:-3}"
INCLUDE_IMAGE="${INCLUDE_IMAGE:-true}"
GENERATED_CONFIG="${GENERATED_CONFIG:-}"

APPLY_NETWORK=0
EXPECT_CAMERA=0
GRAB_FRAMES=0
RUN_DETECT_SECONDS=0

PASS=0
WARN=0
FAIL=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run on the RK3588 board before and after the Hikrobot MV-CA020-20GC arrives.
Default mode checks software readiness only. Add --expect-camera when the
camera is powered and connected.

Options:
  --apply-network          Configure camera NIC only: ${CAMERA_IFACE}=${CAMERA_ADDR}, mtu=${MTU}
  --expect-camera          Treat eth0 link / camera discovery / grab failures as failures
  --grab [frames]          Try a GStreamer aravissrc grab (default frames: 30)
  --run-detect <seconds>   Run detect_cli for N seconds with the Hikrobot config
  --camera-iface <iface>   Camera NIC (default: ${CAMERA_IFACE})
  --camera-addr <cidr>     Camera NIC address (default: ${CAMERA_ADDR})
  --camera-name <name>     Aravis camera name; empty string means first camera
  --upload-iface <iface>   Uplink NIC (default: ${UPLOAD_IFACE})
  --upload-addr <cidr>     Uplink NIC address for bind_ip (default: ${UPLOAD_ADDR})
  --pc-host <ip>           PC receiver IP (default: ${PC_HOST})
  --pc-port <port>         PC receiver port (default: ${PC_PORT})
  --width <px>             Capture width (default: ${WIDTH})
  --height <px>            Capture height (default: ${HEIGHT})
  --fps <n>                Capture FPS (default: ${FPS})
  --format <fmt>           Caps format: BGR, BayerRG8, Mono8, ... (default: ${FORMAT})
  --conf-threshold <v>     Detection confidence threshold (default: ${CONF_THRESHOLD})
  --nms-threshold <v>      NMS IoU threshold (default: ${NMS_THRESHOLD})
  --image-interval <n>     TCP JPEG interval (default: ${IMAGE_INTERVAL})
  --include-image <0|1>    Include JPEG frames in TCP output (default: ${INCLUDE_IMAGE})
  --generate-config <path> Write an effective detect_cli config and validate it
  --mtu <n>                Camera NIC MTU when applying network (default: ${MTU})
  --config <path>          detect_cli config (default: ${CONFIG})
  --binary <path>          detect_cli binary (default: ${BINARY})
  -h, --help               Show this help

Examples:
  scripts/deploy/prepare_hikrobot_gige.sh
  scripts/deploy/prepare_hikrobot_gige.sh --apply-network
  scripts/deploy/prepare_hikrobot_gige.sh --expect-camera --grab 30
  scripts/deploy/prepare_hikrobot_gige.sh --expect-camera --run-detect 30
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply-network) APPLY_NETWORK=1; shift ;;
    --expect-camera) EXPECT_CAMERA=1; shift ;;
    --grab)
      GRAB_FRAMES="${2:-30}"
      if [[ $# -gt 1 && "$2" =~ ^[0-9]+$ ]]; then shift 2; else GRAB_FRAMES=30; shift; fi
      ;;
    --run-detect) RUN_DETECT_SECONDS="$2"; shift 2 ;;
    --camera-iface) CAMERA_IFACE="$2"; shift 2 ;;
    --camera-addr) CAMERA_ADDR="$2"; shift 2 ;;
    --camera-name) CAMERA_NAME="$2"; shift 2 ;;
    --upload-iface) UPLOAD_IFACE="$2"; shift 2 ;;
    --upload-addr) UPLOAD_ADDR="$2"; shift 2 ;;
    --pc-host) PC_HOST="$2"; shift 2 ;;
    --pc-port) PC_PORT="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --conf-threshold) CONF_THRESHOLD="$2"; shift 2 ;;
    --nms-threshold) NMS_THRESHOLD="$2"; shift 2 ;;
    --image-interval) IMAGE_INTERVAL="$2"; shift 2 ;;
    --include-image)
      case "$2" in
        1|true|TRUE|yes|YES|on|ON) INCLUDE_IMAGE=true ;;
        0|false|FALSE|no|NO|off|OFF) INCLUDE_IMAGE=false ;;
        *) echo "--include-image expects 0/1 or true/false, got: $2"; exit 2 ;;
      esac
      shift 2
      ;;
    --generate-config) GENERATED_CONFIG="$2"; shift 2 ;;
    --mtu) MTU="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$ROOT/$path"
  fi
}

section() {
  echo
  echo "== $1 =="
}

ok() {
  echo "[PASS] $1"
  PASS=$((PASS + 1))
}

warn() {
  echo "[WARN] $1"
  WARN=$((WARN + 1))
}

fail() {
  echo "[FAIL] $1"
  FAIL=$((FAIL + 1))
}

require_or_fail() {
  local desc="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    ok "$desc"
  else
    fail "$desc"
  fi
}

optional_or_warn() {
  local desc="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    ok "$desc"
  else
    warn "$desc"
  fi
}

find_arv_tool() {
  for cmd in arv-tool-0.10 arv-tool-0.8 arv-tool-0.6 arv-tool; do
    if command -v "$cmd" >/dev/null 2>&1; then
      printf '%s\n' "$cmd"
      return 0
    fi
  done
  return 1
}

canonical_format() {
  local token
  token="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr -d '_- ')"
  case "$token" in
    BGR|BGR8|BGR888) echo "BGR" ;;
    RGB|RGB8|RGB888) echo "RGB" ;;
    MONO8|GRAY8|Y8) echo "GRAY8" ;;
    BAYERRG8|BAYERRGGB8|RGGB8|RGGB) echo "rggb" ;;
    BAYERBG8|BAYERBGGR8|BGGR8|BGGR) echo "bggr" ;;
    BAYERGR8|BAYERGRBG8|GRBG8|GRBG) echo "grbg" ;;
    BAYERGB8|BAYERGBRG8|GBRG8|GBRG) echo "gbrg" ;;
    *) echo "$1" ;;
  esac
}

media_type_for_format() {
  case "$(canonical_format "$1")" in
    rggb|bggr|grbg|gbrg) echo "video/x-bayer" ;;
    *) echo "video/x-raw" ;;
  esac
}

camera_name_is_auto() {
  local lowered
  lowered="$(printf '%s' "$CAMERA_NAME" | tr '[:upper:]' '[:lower:]')"
  [[ -z "$CAMERA_NAME" || "$lowered" == "auto" || "$lowered" == "first" ]]
}

upload_ip() {
  printf '%s\n' "${UPLOAD_ADDR%%/*}"
}

build_source_uri() {
  local uri=""
  if ! camera_name_is_auto; then
    uri="camera-name=${CAMERA_NAME},"
  fi
  uri+="width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1,format=${FORMAT},pull-timeout-ms=500,max-failures=10"
  printf '%s\n' "$uri"
}

generate_detect_config() {
  local out_path out_abs source_uri
  out_path="$1"
  out_abs="$(abs_path "$out_path")"
  source_uri="$(build_source_uri)"
  mkdir -p "$(dirname "$out_abs")"
  cat > "$out_abs" <<EOF
# Generated by scripts/deploy/prepare_hikrobot_gige.sh
source:
  type: gige
  uri: "${source_uri}"

engine:
  type: rknn
  model: "artifacts/models/best_person_aug_416_norm_int8.rknn"
  input_size: [416, 416]
  use_npu_multicore: true
  use_zero_copy: false

preprocess:
  profile: speed
  use_rga_preprocess: true
  undistort:
    enable: false
    calibration_file: ""
  roi:
    enable: false
  gamma:
    enable: false
  white_balance:
    enable: false
  denoise:
    enable: false

postprocess:
  conf_threshold: ${CONF_THRESHOLD}
  nms_threshold: ${NMS_THRESHOLD}
  max_detections: 300

tracking:
  enable: true
  match_iou: 0.30
  ema_alpha: 0.65
  confirm_hits: 2
  max_misses: 4
  keep_missing_tracks: true
  missing_conf_decay: 0.08

runtime:
  warmup: 10
  async: true

output:
  type: tcp
  tcp:
    host: "${PC_HOST}"
    port: ${PC_PORT}
    queue_size: 32
    bind_ip: "$(upload_ip)"
    bind_interface: "${UPLOAD_IFACE}"
    include_image: ${INCLUDE_IMAGE}
    image_quality: 80
    image_interval: ${IMAGE_INTERVAL}
    draw_detections: true

logging:
  level: "INFO"

classes: "config/person_classes.txt"
EOF
  CONFIG="$out_path"
  ok "Generated effective config: $out_path"
}

link_detected() {
  ethtool "$1" 2>/dev/null | awk -F': ' '/Link detected/ {print $2; exit}'
}

camera_link_check() {
  local link
  link="$(link_detected "$CAMERA_IFACE")"
  if [[ "$link" == "yes" ]]; then
    ok "${CAMERA_IFACE} link detected"
    ethtool "$CAMERA_IFACE" 2>/dev/null | awk '/Speed:|Duplex:|Link detected:/ {print "  " $0}'
  elif [[ "$EXPECT_CAMERA" -eq 1 ]]; then
    fail "${CAMERA_IFACE} link detected"
    ethtool "$CAMERA_IFACE" 2>/dev/null | awk '/Speed:|Duplex:|Link detected:/ {print "  " $0}' || true
  else
    warn "${CAMERA_IFACE} has no carrier yet (expected before camera power/cable)"
  fi
}

upload_link_check() {
  if ! command -v ethtool >/dev/null 2>&1; then
    warn "ethtool missing; cannot check ${UPLOAD_IFACE} link speed"
    return
  fi
  local speed duplex link
  speed="$(ethtool "$UPLOAD_IFACE" 2>/dev/null | awk -F': ' '/Speed/ {print $2; exit}')"
  duplex="$(ethtool "$UPLOAD_IFACE" 2>/dev/null | awk -F': ' '/Duplex/ {print $2; exit}')"
  link="$(link_detected "$UPLOAD_IFACE")"
  if [[ "$link" != "yes" ]]; then
    warn "${UPLOAD_IFACE} link is not detected"
    return
  fi
  if [[ "$speed" == "1000Mb/s" && "$duplex" == "Full" ]]; then
    ok "${UPLOAD_IFACE} link speed ${speed} ${duplex}"
  else
    warn "${UPLOAD_IFACE} link speed is ${speed:-unknown} ${duplex:-unknown}; expected 1000Mb/s Full for throughput KPI"
  fi
}

discover_camera() {
  local tool
  if tool="$(find_arv_tool)"; then
    echo "Using $tool list"
    if "$tool" list; then
      ok "Aravis camera list command"
    elif [[ "$EXPECT_CAMERA" -eq 1 ]]; then
      fail "Aravis camera discovery"
    else
      warn "Aravis camera discovery returned no camera"
    fi
  else
    warn "arv-tool is not installed; aravissrc can still be used through GStreamer"
    if command -v gst-device-monitor-1.0 >/dev/null 2>&1; then
      echo "Using gst-device-monitor-1.0 fallback for 5 seconds"
      local monitor_output
      monitor_output="$(timeout 5s gst-device-monitor-1.0 Video/Source 2>/dev/null \
        | grep -Ei 'aravis|gige|gev|mv-ca|hikrobot|hikvision' || true)"
      if [[ -n "$monitor_output" ]]; then
        printf '%s\n' "$monitor_output" | sed 's/^/  /'
      else
        echo "  No GigE Vision device was reported by gst-device-monitor."
      fi
    fi
  fi
}

grab_camera() {
  local fmt media caps timeout_s
  fmt="$(canonical_format "$FORMAT")"
  media="$(media_type_for_format "$FORMAT")"
  caps="${media},width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1,format=${fmt}"
  timeout_s=$((GRAB_FRAMES / FPS + 8))
  if [[ "$timeout_s" -lt 10 ]]; then timeout_s=10; fi

  local -a cmd=(timeout "${timeout_s}s" gst-launch-1.0 -q aravissrc)
  if ! camera_name_is_auto; then
    cmd+=("camera-name=${CAMERA_NAME}")
  fi
  cmd+=("num-buffers=${GRAB_FRAMES}" "!" "$caps" "!" fakesink sync=false)

  echo "Running: ${cmd[*]}"
  if "${cmd[@]}"; then
    ok "GStreamer grabbed ${GRAB_FRAMES} frame(s)"
  else
    fail "GStreamer grab failed"
  fi
}

validate_yaml_config() {
  local cfg
  cfg="$(abs_path "$CONFIG")"
  if [[ ! -f "$cfg" ]]; then
    fail "Config exists: $CONFIG"
    return
  fi
  ok "Config exists: $CONFIG"

  if ! python3 -c 'import yaml' >/dev/null 2>&1; then
    warn "PyYAML missing; skipping structured config validation"
    return
  fi

  if python3 - "$ROOT" "$cfg" <<'PY'
import pathlib
import sys

import yaml

root = pathlib.Path(sys.argv[1])
cfg_path = pathlib.Path(sys.argv[2])
cfg = yaml.safe_load(cfg_path.read_text()) or {}
errors = []
source = cfg.get("source", {})
engine = cfg.get("engine", {})
output = (cfg.get("output", {}) or {}).get("tcp", {})

if source.get("type") != "gige":
    errors.append("source.type must be gige")
model = root / str(engine.get("model", ""))
if not model.exists():
    errors.append(f"model missing: {model}")
if not output.get("host"):
    errors.append("output.tcp.host is required")
if output.get("bind_interface") != "eth1":
    errors.append("output.tcp.bind_interface should be eth1")
if errors:
    for error in errors:
        print(error)
    raise SystemExit(1)
PY
  then
    ok "Config sanity: $CONFIG"
  else
    fail "Config sanity: $CONFIG"
  fi
}

run_detect() {
  local bin cfg log json duration
  bin="$(abs_path "$BINARY")"
  duration="$RUN_DETECT_SECONDS"
  if [[ -z "$GENERATED_CONFIG" ]]; then
    GENERATED_CONFIG="artifacts/hikrobot_mv_ca020_20gc_effective.yaml"
  fi
  generate_detect_config "$GENERATED_CONFIG"
  cfg="$(abs_path "$CONFIG")"
  log="$ROOT/artifacts/hikrobot_mv_ca020_20gc_${duration}s.log"
  json="$ROOT/artifacts/hikrobot_mv_ca020_20gc_${duration}s.json"

  if [[ ! -x "$bin" ]]; then
    fail "detect_cli is executable: $BINARY"
    return
  fi
  mkdir -p "$ROOT/artifacts"
  echo "Running: timeout ${duration}s $bin --cfg $cfg --json $json"
  timeout "${duration}s" "$bin" --cfg "$cfg" --json "$json" 2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -eq 0 || "$status" -eq 124 ]]; then
    ok "detect_cli ran for ${duration}s"
  else
    fail "detect_cli failed with status ${status}"
  fi
  echo "Log: $log"
  echo "JSON: $json"
}

section "Configuration"
echo "Root:          $ROOT"
echo "Camera NIC:    $CAMERA_IFACE $CAMERA_ADDR mtu=$MTU"
echo "Camera name:   ${CAMERA_NAME:-first available camera}"
echo "Capture:       ${WIDTH}x${HEIGHT}@${FPS} format=${FORMAT}"
echo "Source URI:    $(build_source_uri)"
echo "Upload NIC:    $UPLOAD_IFACE expected $UPLOAD_ADDR"
echo "PC receiver:   ${PC_HOST}:${PC_PORT}"
echo "Config:        $CONFIG"
echo "Thresholds:    conf=${CONF_THRESHOLD} nms=${NMS_THRESHOLD}"
echo "TCP images:    include=${INCLUDE_IMAGE} interval=${IMAGE_INTERVAL}"

if [[ -n "$GENERATED_CONFIG" ]]; then
  generate_detect_config "$GENERATED_CONFIG"
fi

section "Software Readiness"
require_or_fail "ip command" command -v ip
optional_or_warn "ethtool command" command -v ethtool
require_or_fail "gst-inspect-1.0 command" command -v gst-inspect-1.0
require_or_fail "gst-launch-1.0 command" command -v gst-launch-1.0
require_or_fail "GStreamer aravissrc plugin" gst-inspect-1.0 aravissrc
require_or_fail "python3 command" command -v python3
optional_or_warn "live_viewer.py exists" test -f "$ROOT/scripts/live_viewer.py"
optional_or_warn "results_receiver.py exists" test -f "$ROOT/scripts/results_receiver.py"
validate_yaml_config

section "Network"
if [[ "$APPLY_NETWORK" -eq 1 ]]; then
  if [[ "$(id -u)" -ne 0 ]]; then
    fail "--apply-network requires root"
  else
    ip link set "$CAMERA_IFACE" up || fail "bring up $CAMERA_IFACE"
    ip addr flush dev "$CAMERA_IFACE" || fail "flush $CAMERA_IFACE"
    ip addr add "$CAMERA_ADDR" dev "$CAMERA_IFACE" || fail "assign $CAMERA_ADDR to $CAMERA_IFACE"
    ip link set dev "$CAMERA_IFACE" mtu "$MTU" || warn "failed to set $CAMERA_IFACE mtu=$MTU"
    ok "Applied camera NIC config"
  fi
fi

ip -brief addr show "$CAMERA_IFACE" 2>/dev/null || warn "camera interface missing: $CAMERA_IFACE"
ip -brief addr show "$UPLOAD_IFACE" 2>/dev/null || warn "upload interface missing: $UPLOAD_IFACE"
camera_link_check
upload_link_check
if ping -I "$UPLOAD_IFACE" -c 1 -W 1 "$PC_HOST" >/dev/null 2>&1; then
  ok "PC reachable from $UPLOAD_IFACE ($PC_HOST)"
else
  warn "PC ping failed from $UPLOAD_IFACE ($PC_HOST); TCP may still work if ICMP is blocked"
fi
ip route | sed 's/^/  /'

section "Camera Discovery"
discover_camera

if [[ "$GRAB_FRAMES" -gt 0 ]]; then
  section "Camera Grab"
  grab_camera
fi

if [[ "$RUN_DETECT_SECONDS" -gt 0 ]]; then
  section "Detect CLI"
  run_detect
fi

section "Summary"
echo "PASS=$PASS WARN=$WARN FAIL=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "Software side is ready. Remaining pending item is the physical camera link."
  exit 0
fi
exit 1
