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

get_iface_ipv4() {
  local iface=$1
  ip -o -4 addr show "$iface" | awk '{print $4}' | cut -d/ -f1 | head -n1
}

run_server() {
  echo "Starting iperf3 server..."
  iperf3 -s
}

run_client() {
  local server=$1
  local iface=${2:-}
  local bind_ip=""
  local -a bind_opt=()
  if [ -n "$iface" ]; then
    echo "Binding to interface $iface"
    ip addr show "$iface" || true
    bind_ip="$(get_iface_ipv4 "$iface")"
    if [ -z "$bind_ip" ]; then
      echo "Unable to determine IPv4 address for interface $iface"
      return 1
    fi
    bind_opt=(-B "$bind_ip")
  fi
  echo "Upstream -> server"
  iperf3 -c "$server" -P 4 -t 30 "${bind_opt[@]}"
  echo "Downstream <- server"
  iperf3 -c "$server" -P 4 -t 30 -R "${bind_opt[@]}"
}

run_dual() {
  local server=$1
  local i1=$2
  local i2=$3
  local ip1=""
  local ip2=""
  ip1="$(get_iface_ipv4 "$i1")"
  ip2="$(get_iface_ipv4 "$i2")"
  if [ -z "$ip1" ] || [ -z "$ip2" ]; then
    echo "Unable to determine IPv4 address for interfaces: $i1($ip1), $i2($ip2)"
    return 1
  fi

  echo "Running concurrent tests on $i1 and $i2"
  iperf3 -c "$server" -P 4 -t 30 -B "$ip1" &
  iperf3 -c "$server" -P 4 -t 30 -B "$ip2" &
  wait
  iperf3 -c "$server" -P 4 -t 30 -R -B "$ip1" &
  iperf3 -c "$server" -P 4 -t 30 -R -B "$ip2" &
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
