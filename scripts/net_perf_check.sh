#!/usr/bin/env bash
set -euo pipefail

# Usage:
#  On a PC/server on the same LAN:
#    ./scripts/net_perf_check.sh server
#  On RK3588 (client):
#    ./scripts/net_perf_check.sh client <server_ip> [iface]
#
#  To test both eth0 and eth1 concurrently (on client):
#    ./scripts/net_perf_check.sh dual <server_ip> eth0 eth1

ROLE=${1:-}
SERVER_IP=${2:-}
IFACE=${3:-}
IFACE2=${4:-}

run_server() {
  echo "Starting iperf3 server..."
  iperf3 -s
}

run_client() {
  local server=$1
  local iface=${2:-}
  if [ -n "$iface" ]; then
    echo "Binding to interface $iface"
    ip addr show "$iface" || true
  fi
  echo "Upstream -> server"
  iperf3 -c "$server" -P 4 -t 30 ${iface:+-B $(ip -o -4 addr show $iface | awk '{print $4}' | cut -d/ -f1 | head -n1)}
  echo "Downstream <- server"
  iperf3 -c "$server" -P 4 -t 30 -R ${iface:+-B $(ip -o -4 addr show $iface | awk '{print $4}' | cut -d/ -f1 | head -n1)}
}

run_dual() {
  local server=$1
  local i1=$2
  local i2=$3
  echo "Running concurrent tests on $i1 and $i2"
  (iperf3 -c "$server" -P 4 -t 30 -B $(ip -o -4 addr show $i1 | awk '{print $4}' | cut -d/ -f1 | head -n1) &)
  (iperf3 -c "$server" -P 4 -t 30 -B $(ip -o -4 addr show $i2 | awk '{print $4}' | cut -d/ -f1 | head -n1) &)
  wait
  (iperf3 -c "$server" -P 4 -t 30 -R -B $(ip -o -4 addr show $i1 | awk '{print $4}' | cut -d/ -f1 | head -n1) &)
  (iperf3 -c "$server" -P 4 -t 30 -R -B $(ip -o -4 addr show $i2 | awk '{print $4}' | cut -d/ -f1 | head -n1) &)
  wait
}

case "$ROLE" in
  server)
    run_server
    ;;
  client)
    if [ -z "$SERVER_IP" ]; then echo "Missing server IP"; exit 1; fi
    run_client "$SERVER_IP" "$IFACE"
    ;;
  dual)
    if [ -z "$SERVER_IP" ] || [ -z "$IFACE" ] || [ -z "$IFACE2" ]; then echo "Usage: dual <server_ip> <iface1> <iface2>"; exit 1; fi
    run_dual "$SERVER_IP" "$IFACE" "$IFACE2"
    ;;
  *)
    echo "Usage: $0 server | client <server_ip> [iface] | dual <server_ip> <iface1> <iface2>"; exit 1;
    ;;
esac

