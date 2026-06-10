#!/usr/bin/env bash
set -euo pipefail

# Host-side latency diagnosis runner for the RK3588 GigE pipeline.
#
# It switches the board to performance mode, samples thermal/frequency state,
# runs the normal GigE acceptance path, then regenerates the report with the
# corrected process/capture-wait timing fields.

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT"

BOARD_HOST="${BOARD_HOST:-192.168.137.226}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_ROOT="${BOARD_ROOT:-/opt/rk_app_current}"
PC_HOST="${PC_HOST:-192.168.137.1}"
RUN_SECONDS="${RUN_SECONDS:-120}"
INCLUDE_IMAGE="${INCLUDE_IMAGE:-0}"
IMAGE_INTERVAL="${IMAGE_INTERVAL:-999999}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.40}"
GRAB_FRAMES="${GRAB_FRAMES:-30}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EVIDENCE_DIR="${EVIDENCE_DIR:-artifacts/live_latency_diagnosis_${STAMP}}"
REMOTE_MONITOR="/tmp/rkapp_latency_monitor_${STAMP}.csv"
REMOTE_STOP="/tmp/rkapp_latency_monitor_${STAMP}.stop"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --host <ip>              Board IP (default: ${BOARD_HOST})
  --user <name>            SSH user (default: ${BOARD_USER})
  --board-root <path>      Board runtime root (default: ${BOARD_ROOT})
  --pc-host <ip>           PC receiver IP (default: ${PC_HOST})
  --seconds <n>            detect_cli runtime seconds (default: ${RUN_SECONDS})
  --include-image <0|1>    Include JPEGs in TCP output (default: ${INCLUDE_IMAGE})
  --image-interval <n>     JPEG interval (default: ${IMAGE_INTERVAL})
  --conf-threshold <v>     Confidence threshold (default: ${CONF_THRESHOLD})
  --grab <n>               Pre-test camera grab frames (default: ${GRAB_FRAMES})
  --evidence-dir <path>    Local evidence output dir (default: ${EVIDENCE_DIR})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) BOARD_HOST="$2"; shift 2 ;;
    --user) BOARD_USER="$2"; shift 2 ;;
    --board-root) BOARD_ROOT="$2"; shift 2 ;;
    --pc-host) PC_HOST="$2"; shift 2 ;;
    --seconds) RUN_SECONDS="$2"; shift 2 ;;
    --include-image) INCLUDE_IMAGE="$2"; shift 2 ;;
    --image-interval) IMAGE_INTERVAL="$2"; shift 2 ;;
    --conf-threshold) CONF_THRESHOLD="$2"; shift 2 ;;
    --grab) GRAB_FRAMES="$2"; shift 2 ;;
    --evidence-dir) EVIDENCE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$EVIDENCE_DIR"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${BOARD_USER}@${BOARD_HOST}")
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no)

echo "[diag] evidence: $EVIDENCE_DIR"

if [[ -f scripts/perf/rk3588_performance_mode.sh ]]; then
  "${SSH[@]}" "mkdir -p '$BOARD_ROOT/scripts/perf'" || true
  "${SCP[@]}" scripts/perf/rk3588_performance_mode.sh \
    "${BOARD_USER}@${BOARD_HOST}:${BOARD_ROOT}/scripts/perf/rk3588_performance_mode.sh" || true
  "${SSH[@]}" "chmod +x '$BOARD_ROOT/scripts/perf/rk3588_performance_mode.sh'" || true
fi

"${SSH[@]}" "bash '$BOARD_ROOT/scripts/perf/rk3588_performance_mode.sh'" \
  > "$EVIDENCE_DIR/performance_mode.log" 2>&1 || true

"${SSH[@]}" "bash -s" -- "$REMOTE_MONITOR" "$REMOTE_STOP" "$((RUN_SECONDS + 20))" <<'REMOTE' &
set -euo pipefail
out="$1"
stop_file="$2"
duration="$3"
(
  echo "ts,soc_temp_c,big0_temp_c,big1_temp_c,npu_temp_c,cpu0,cpu4,cpu6,npu_freq,npu_gov,load1"
  end=$((SECONDS + duration))
  while [[ "$SECONDS" -lt "$end" && ! -e "$stop_file" ]]; do
    ts="$(date +%s)"
    read_temp() {
      local p="$1"
      [[ -r "$p" ]] && awk '{printf "%.1f", $1/1000.0}' "$p" || printf "0"
    }
    soc="$(read_temp /sys/class/thermal/thermal_zone0/temp)"
    big0="$(read_temp /sys/class/thermal/thermal_zone1/temp)"
    big1="$(read_temp /sys/class/thermal/thermal_zone2/temp)"
    npu_temp="$(read_temp /sys/class/thermal/thermal_zone6/temp)"
    cpu0="$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq 2>/dev/null || echo 0)"
    cpu4="$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq 2>/dev/null || echo 0)"
    cpu6="$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq 2>/dev/null || echo 0)"
    npu_freq="$(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || echo 0)"
    npu_gov="$(cat /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null || echo none)"
    load1="$(awk '{print $1}' /proc/loadavg)"
    echo "$ts,$soc,$big0,$big1,$npu_temp,$cpu0,$cpu4,$cpu6,$npu_freq,$npu_gov,$load1"
    sleep 1
  done
) > "$out"
REMOTE
monitor_pid=$!

set +e
EVIDENCE_DIR="$EVIDENCE_DIR" PC_HOST="$PC_HOST" \
  scripts/demo/run_gige_acceptance.sh \
    --host "$BOARD_HOST" \
    --user "$BOARD_USER" \
    --board-root "$BOARD_ROOT" \
    --expect-camera \
    --apply-network \
    --grab "$GRAB_FRAMES" \
    --run-detect "$RUN_SECONDS" \
    --include-image "$INCLUDE_IMAGE" \
    --image-interval "$IMAGE_INTERVAL" \
    --conf-threshold "$CONF_THRESHOLD"
accept_rc=$?
set -e

"${SSH[@]}" "touch '$REMOTE_STOP'" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
"${SCP[@]}" "${BOARD_USER}@${BOARD_HOST}:${REMOTE_MONITOR}" \
  "$EVIDENCE_DIR/perf_monitor.csv" 2>/dev/null || true
"${SSH[@]}" "rm -f '$REMOTE_MONITOR' '$REMOTE_STOP'" 2>/dev/null || true

python3 tools/generate_gige_acceptance_report.py --evidence-dir "$EVIDENCE_DIR" \
  > "$EVIDENCE_DIR/report_generator.log" 2>&1 || true

python3 - "$EVIDENCE_DIR" "$accept_rc" <<'PY'
from pathlib import Path
import csv
import json
import math
import re
import statistics
import sys

evidence = Path(sys.argv[1])
accept_rc = int(sys.argv[2])

def stats(values):
    if not values:
        return {}
    ordered = sorted(values)
    def pct(q):
        return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]
    return {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 3),
        "p95_ms": round(pct(0.95), 3),
        "p99_ms": round(pct(0.99), 3),
        "max_ms": round(max(values), 3),
        "over_45ms_frames": sum(v > 45 for v in values),
    }

rows = []
for name in ("tcp_sink_results.jsonl", "board_detect_results.jsonl"):
    path = evidence / name
    if not path.exists():
        continue
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if rows:
        break

process = []
total = []
capture = []
output = []
for row in rows:
    timing = row.get("timing") or {}
    if "latency_ms" in row:
        process.append(float(row["latency_ms"]))
    elif "process_ms" in timing:
        process.append(float(timing["process_ms"]))
    if "total_with_capture_wait_ms" in timing:
        total.append(float(timing["total_with_capture_wait_ms"]))
    if "capture_wait_ms" in timing:
        capture.append(float(timing["capture_wait_ms"]))
    if "output_ms" in timing:
        output.append(float(timing["output_ms"]))

monitor = []
mon_path = evidence / "perf_monitor.csv"
if mon_path.exists():
    with mon_path.open() as handle:
        monitor = list(csv.DictReader(handle))
temps = [float(r["soc_temp_c"]) for r in monitor if r.get("soc_temp_c")]
npu_freqs = [int(r["npu_freq"]) for r in monitor if r.get("npu_freq")]

summary = {
    "acceptance_exit_code": accept_rc,
    "frames": len(rows),
    "process_latency": stats(process),
    "total_with_capture_wait_latency": stats(total),
    "capture_wait_latency": stats(capture),
    "output_latency": stats(output),
    "soc_temp_c": {
        "samples": len(temps),
        "min": min(temps) if temps else None,
        "max": max(temps) if temps else None,
    },
    "npu_freq_hz": {
        "samples": len(npu_freqs),
        "min": min(npu_freqs) if npu_freqs else None,
        "max": max(npu_freqs) if npu_freqs else None,
    },
}
(evidence / "latency_diagnosis_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[diag] done: $EVIDENCE_DIR"
exit "$accept_rc"
