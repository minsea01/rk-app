#!/usr/bin/env bash
# On RK3588 board: start two iperf3 servers bound to eth0/eth1 IPs.
# Usage: ./scripts/net/iperf_board.sh <eth0_ip> <eth1_ip>
set -euo pipefail
E0=${1:?eth0 ip required}
E1=${2:?eth1 ip required}
iperf3 -s -B "$E0" &
iperf3 -s -B "$E1" &
echo "Started iperf3 servers on $E0 and $E1"

