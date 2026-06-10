#!/usr/bin/env bash
set -euo pipefail

# Put RK3588 CPU/GPU/NPU governors into performance mode for reproducible
# latency tests. Run as root on the board before detect_cli benchmarks.

if [[ "$(id -u)" -ne 0 ]]; then
  echo "rk3588_performance_mode.sh must run as root" >&2
  exit 1
fi

echo "== CPU governors =="
for policy in /sys/devices/system/cpu/cpufreq/policy*; do
  [[ -d "$policy" ]] || continue
  if [[ -w "$policy/scaling_governor" ]] &&
     grep -qw performance "$policy/scaling_available_governors" 2>/dev/null; then
    echo performance > "$policy/scaling_governor"
  fi
  printf "%s governor=" "$(basename "$policy")"
  cat "$policy/scaling_governor" 2>/dev/null || true
  printf "%s cur_freq=" "$(basename "$policy")"
  cat "$policy/scaling_cur_freq" 2>/dev/null || true
done

echo "== Devfreq governors =="
for dev in /sys/class/devfreq/*; do
  [[ -d "$dev" ]] || continue
  if [[ -w "$dev/governor" ]] &&
     grep -qw performance "$dev/available_governors" 2>/dev/null; then
    echo performance > "$dev/governor" || true
  fi
  printf "%s governor=" "$(basename "$dev")"
  cat "$dev/governor" 2>/dev/null || true
  printf "%s cur_freq=" "$(basename "$dev")"
  cat "$dev/cur_freq" 2>/dev/null || true
done

echo "== Thermal zones =="
for zone in /sys/class/thermal/thermal_zone*; do
  [[ -r "$zone/temp" ]] || continue
  printf "%s " "$(basename "$zone")"
  cat "$zone/type" 2>/dev/null | tr -d '\n' || true
  printf " "
  awk '{printf "%.1f C\n", $1 / 1000.0}' "$zone/temp"
done
