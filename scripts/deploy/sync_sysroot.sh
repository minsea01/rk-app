#!/usr/bin/env bash
set -euo pipefail

# 从板子同步 /usr/include 与 /usr/lib 到本机 sysroot，便于编译期头文件/库对齐
# 用法：
#   scripts/deploy/sync_sysroot.sh --host 192.168.1.100 --user root --port 22

HOST=""
USER="root"
PORT="22"
DEST_SYSROOT="sysroot/aarch64-linux-gnu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --user) USER="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --dest) DEST_SYSROOT="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 --host <ip> [--user root] [--port 22] [--dest sysroot/aarch64-linux-gnu]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

[[ -n "$HOST" ]] || { echo "--host required"; exit 2; }

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
mkdir -p "$ROOT/$DEST_SYSROOT/usr/include" "$ROOT/$DEST_SYSROOT/usr/lib"

echo "➡️  Syncing headers to $DEST_SYSROOT/usr/include"
rsync -avz -e "ssh -p $PORT" "${USER}@${HOST}:/usr/include/" "$ROOT/$DEST_SYSROOT/usr/include/"

echo "➡️  Syncing libs to $DEST_SYSROOT/usr/lib"
rsync -avz -e "ssh -p $PORT" "${USER}@${HOST}:/usr/lib/aarch64-linux-gnu/" "$ROOT/$DEST_SYSROOT/usr/lib/"

echo "✅ Done. You can point your toolchain/sysroot to: $ROOT/$DEST_SYSROOT"
