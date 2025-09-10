#!/bin/bash
set -euo pipefail

# Dual-NIC throughput proof (实干版)
# Requires iperf3 and two endpoints via env: IPERF_ETH0, IPERF_ETH1

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if ! command -v iperf3 >/dev/null 2>&1; then
  echo "[SKIP] iperf3 not installed"
  exit 2
fi

if [ -z "${IPERF_ETH0:-}" ] || [ -z "${IPERF_ETH1:-}" ]; then
  echo "[SKIP] Set IPERF_ETH0 and IPERF_ETH1 to run this proof"
  exit 2
fi

echo "[NIC] Testing eth0 -> $IPERF_ETH0 and eth1 -> $IPERF_ETH1"
iperf3 -c "$IPERF_ETH0" -t 8 -i 1 2>&1 | tee "$LOG_DIR/iperf_eth0.log" | cat &
P0=$!
iperf3 -c "$IPERF_ETH1" -t 8 -i 1 2>&1 | tee "$LOG_DIR/iperf_eth1.log" | cat &
P1=$!
wait $P0 || true
wait $P1 || true

BW0=$(grep -E "sender|receiver" "$LOG_DIR/iperf_eth0.log" | tail -1 | awk '{print $(NF-1)}')
BW1=$(grep -E "sender|receiver" "$LOG_DIR/iperf_eth1.log" | tail -1 | awk '{print $(NF-1)}')
echo "eth0: ${BW0:-0} Mbits/sec, eth1: ${BW1:-0} Mbits/sec"

awk -v b0="${BW0:-0}" -v b1="${BW1:-0}" 'BEGIN{ exit (b0>=900 && b1>=900)?0:1 }' && {
  echo "[PASS] Both NICs >= 900 Mbps"
  exit 0
} || {
  echo "[FAIL] One or both NICs < 900 Mbps"
  exit 1
}


