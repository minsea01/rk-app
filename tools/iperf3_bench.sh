#!/usr/bin/env bash
set -euo pipefail

OUT_JSON=${1:-artifacts/iperf3.json}
PORT=${PORT:-5201}
DURATION=${DURATION:-3}

cleanup() {
  [[ -n "${SRV_PID:-}" ]] && kill ${SRV_PID} >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Start server with log and wait for readiness
LOG=/tmp/iperf3_server_$$.log
iperf3 -s -p "$PORT" >"$LOG" 2>&1 &
SRV_PID=$!
for i in {1..20}; do
  if grep -q "Server listening" "$LOG" 2>/dev/null; then break; fi
  sleep 0.2
done

# Run client to loopback with JSON output
if ! iperf3 -c 127.0.0.1 -p "$PORT" -t "$DURATION" -J | tee "$OUT_JSON" >/dev/null; then
  cat > "$OUT_JSON" <<EOF
{
  "error": "iperf3 failed in this environment"
}
EOF
fi
