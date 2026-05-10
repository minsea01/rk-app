#!/usr/bin/env bash
set -euo pipefail

# Sync RK3588 board time from the host over SSH.
#
# This is intentionally host-pushed instead of NTP-based: Windows ICS/WSL setups
# often block or omit UDP/123, while SSH is already required for deployment.

BOARD_HOST=${BOARD_HOST:-192.168.137.226}
BOARD_USER=${BOARD_USER:-root}
SSH_PORT=${SSH_PORT:-22}
SSH_KNOWN_HOSTS=${SSH_KNOWN_HOSTS:-/tmp/rk-app-known-hosts}
MAX_SKEW_SEC=${MAX_SKEW_SEC:-2}

usage() {
  cat <<EOF
Usage: $0 [--host IP] [--user USER] [--port PORT] [--max-skew SEC]

Environment overrides:
  BOARD_HOST=$BOARD_HOST
  BOARD_USER=$BOARD_USER
  SSH_PORT=$SSH_PORT
  SSH_KNOWN_HOSTS=$SSH_KNOWN_HOSTS
  MAX_SKEW_SEC=$MAX_SKEW_SEC

Example:
  $0 --host 192.168.137.226 --user root
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      BOARD_HOST="$2"
      shift 2
      ;;
    --user)
      BOARD_USER="$2"
      shift 2
      ;;
    --port)
      SSH_PORT="$2"
      shift 2
      ;;
    --max-skew)
      MAX_SKEW_SEC="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$SSH_PORT" =~ ^[0-9]+$ ]]; then
  echo "SSH_PORT must be an integer, got: $SSH_PORT" >&2
  exit 2
fi
if ! [[ "$MAX_SKEW_SEC" =~ ^[0-9]+$ ]]; then
  echo "MAX_SKEW_SEC must be an integer, got: $MAX_SKEW_SEC" >&2
  exit 2
fi

host_epoch=$(date -u +%s)
host_iso=$(date -u -d "@$host_epoch" '+%Y-%m-%dT%H:%M:%SZ')

echo "[clock-sync] host UTC: $host_iso ($host_epoch)"
echo "[clock-sync] board: $BOARD_USER@$BOARD_HOST:$SSH_PORT"

remote_output=$(
  ssh \
    -p "$SSH_PORT" \
    -o BatchMode=yes \
    -o ConnectTimeout=5 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile="$SSH_KNOWN_HOSTS" \
    "$BOARD_USER@$BOARD_HOST" \
    "bash -s" -- "$host_epoch" <<'REMOTE'
set -euo pipefail

target_epoch=$1
before_epoch=$(date -u +%s)
before_iso=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

date -u -s "@$target_epoch" >/dev/null

after_epoch=$(date -u +%s)
after_iso=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

hwclock_status=skipped
if command -v hwclock >/dev/null 2>&1; then
  hwclock_err=$(mktemp)
  if hwclock -w >"$hwclock_err" 2>&1; then
    hwclock_status=ok
  else
    hwclock_status="failed: $(tr '\n' ' ' < "$hwclock_err" | sed 's/[[:space:]]*$//')"
  fi
  rm -f "$hwclock_err"
fi

printf 'before_epoch=%s\n' "$before_epoch"
printf 'before_iso=%s\n' "$before_iso"
printf 'after_epoch=%s\n' "$after_epoch"
printf 'after_iso=%s\n' "$after_iso"
printf 'hwclock=%s\n' "$hwclock_status"
REMOTE
)

while IFS= read -r line; do
  echo "[clock-sync] board $line"
done <<<"$remote_output"

board_after=$(awk -F= '$1 == "after_epoch" {print $2}' <<<"$remote_output")
if ! [[ "$board_after" =~ ^[0-9]+$ ]]; then
  echo "[clock-sync] could not parse board after_epoch" >&2
  exit 1
fi

host_after=$(date -u +%s)
skew=$((board_after - host_after))
abs_skew=${skew#-}
echo "[clock-sync] final skew: ${skew}s"

if (( abs_skew > MAX_SKEW_SEC )); then
  echo "[clock-sync] skew exceeds ${MAX_SKEW_SEC}s" >&2
  exit 1
fi

echo "[clock-sync] ok"
