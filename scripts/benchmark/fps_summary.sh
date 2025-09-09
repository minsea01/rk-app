#!/usr/bin/env bash
# Parse detect_cli output and compute avg latency/FPS.
# Usage: ./scripts/fps_summary.sh run.log
set -euo pipefail
LOG=${1:-run.log}
awk -F'[()]' '/Frame/{split($2,a,"ms"); s+=a[1]; n++} END{if(n>0){avg=s/n; printf "avg=%.1f ms  -> %.1f FPS\n", avg, 1000/avg} else {print "no frames"}}' "$LOG"

