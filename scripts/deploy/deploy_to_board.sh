#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
APP_PATH="$ROOT/out/arm64/bin/rk_app"

HOST=""
USER="root"
PORT="22"
DEST="/opt/rk_app"
ACTION="deploy"   # deploy | run | gdb | kill
GDB_PORT="1234"
LD_LIBRARY_PATH_REMOTE=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --host <ip>            Board IP/hostname (required)
  --user <name>          SSH user (default: root)
  --port <num>           SSH port (default: 22)
  --dest <path>          Remote deploy dir (default: /opt/rk_app)
  --run                  Deploy then run remote binary
  --gdb                  Deploy then run gdbserver :<port>
  --gdb-port <num>       gdbserver port (default: 1234)
  --ld-path <path>       Set LD_LIBRARY_PATH when running on board
  --kill                 Kill remote gdbserver (best-effort)
  -h, --help             Show this help

Examples:
  $0 --host 192.168.1.50 --dest /opt/rk_app          # deploy only
  $0 --host 192.168.1.50 --run                        # deploy and run
  $0 --host 192.168.1.50 --gdb --gdb-port 1234        # deploy and start gdbserver
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --user) USER="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --gdb) ACTION="gdb"; shift;;
    --gdb-port) GDB_PORT="$2"; shift 2;;
    --run) ACTION="run"; shift;;
    --kill) ACTION="kill"; shift;;
    --ld-path) LD_LIBRARY_PATH_REMOTE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "❌ --host is required"; usage; exit 2
fi

if [[ ! -x "$APP_PATH" ]]; then
  echo "❌ 未找到 $APP_PATH"
  echo "👉 先执行：cmake --preset arm64-release && cmake --build --preset arm64 && cmake --install build/arm64"
  exit 1
fi

REMOTE="${USER}@${HOST}"

echo "ℹ️ 目标: $REMOTE (port=$PORT), 部署目录: $DEST"

echo "➡️  创建远端目录: $DEST/bin 和 $DEST/config"
ssh -p "$PORT" "$REMOTE" "mkdir -p '$DEST/bin' '$DEST/config'" 2>/dev/null || true

sync_bin() {
  if command -v rsync >/dev/null 2>&1; then
    echo "➡️  rsync 同步二进制到板子"
    # 优先使用 strip 后的二进制
    TMP_BIN="$ROOT/out/arm64/bin/rk_app"
    if command -v aarch64-linux-gnu-strip >/dev/null 2>&1; then
      echo "➡️  strip 二进制"
      cp "$TMP_BIN" "$TMP_BIN.unstripped"
      aarch64-linux-gnu-strip -S "$TMP_BIN" || mv "$TMP_BIN.unstripped" "$TMP_BIN"
    fi
    rsync -avz -e "ssh -p $PORT" "$ROOT/out/arm64/bin/" "$REMOTE:$DEST/bin/"
  else
    echo "➡️  rsync 不可用，使用 scp 复制"
    scp -P "$PORT" "$ROOT/out/arm64/bin/rk_app" "$REMOTE:$DEST/bin/"
  fi
}

echo "⬆️  部署 rk_app"
sync_bin

case "$ACTION" in
  deploy)
    echo "✅ 部署完成：$REMOTE:$DEST/bin/rk_app"
    ;;
  run)
    echo "🚀 远端运行 rk_app"
    ssh -p "$PORT" "$REMOTE" "cd '$DEST' && chmod +x bin/rk_app && LD_LIBRARY_PATH='$LD_LIBRARY_PATH_REMOTE' ./bin/rk_app --config ./config/app.yaml"
    ;;
  gdb)
    echo "🐞 在板子上启动 gdbserver :$GDB_PORT"
    echo "提示: 在本机 VS Code 选择 'Attach gdbserver (ARM64 board)' 后按 F5。"
    ssh -p "$PORT" "$REMOTE" "cd '$DEST' && chmod +x bin/rk_app && exec env LD_LIBRARY_PATH='$LD_LIBRARY_PATH_REMOTE' gdbserver :$GDB_PORT ./bin/rk_app --config ./config/app.yaml"
    ;;
  kill)
    echo "🧹 结束远端 gdbserver (best-effort)"
    ssh -p "$PORT" "$REMOTE" "pkill -f 'gdbserver.*rk_app'" || true
    ;;
  *)
    echo "内部错误：未知 ACTION=$ACTION"; exit 3
    ;;
esac
