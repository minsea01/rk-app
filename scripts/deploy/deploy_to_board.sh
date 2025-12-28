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

# Security: Input validation functions
validate_path() {
  local path="$1"
  local name="$2"

  # Allow only alphanumeric, forward slash, underscore, hyphen, and dot
  if [[ ! "$path" =~ ^[a-zA-Z0-9/_.-]+$ ]]; then
    echo "❌ Security: Invalid $name path '$path'" >&2
    echo "   Allowed characters: a-z A-Z 0-9 / _ . -" >&2
    exit 1
  fi

  # Prevent directory traversal
  if [[ "$path" == *".."* ]]; then
    echo "❌ Security: Path traversal detected in $name: '$path'" >&2
    exit 1
  fi

  # Must be absolute path for deployment destination
  if [[ "$name" == "destination" && "$path" != /* ]]; then
    echo "❌ Security: Destination must be absolute path, got: '$path'" >&2
    exit 1
  fi
}

validate_port() {
  local port="$1"
  local name="$2"

  # Must be numeric
  if [[ ! "$port" =~ ^[0-9]+$ ]]; then
    echo "❌ Security: Invalid $name port '$port' (must be numeric)" >&2
    exit 1
  fi

  # Valid port range
  if (( port < 1 || port > 65535 )); then
    echo "❌ Security: Invalid $name port $port (must be 1-65535)" >&2
    exit 1
  fi
}

validate_hostname() {
  local host="$1"

  # Allow alphanumeric, dot, hyphen, and colon (for IPv6)
  if [[ ! "$host" =~ ^[a-zA-Z0-9.:_-]+$ ]]; then
    echo "❌ Security: Invalid hostname '$host'" >&2
    echo "   Allowed characters: a-z A-Z 0-9 . : _ -" >&2
    exit 1
  fi
}

validate_username() {
  local user="$1"

  # Standard Unix username rules: alphanumeric, underscore, hyphen
  if [[ ! "$user" =~ ^[a-z_][a-z0-9_-]*\$?$ ]]; then
    echo "❌ Security: Invalid username '$user'" >&2
    echo "   Must start with lowercase letter or underscore" >&2
    exit 1
  fi
}

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

# Security: Validate all user inputs
validate_hostname "$HOST"
validate_username "$USER"
validate_port "$PORT" "SSH"
validate_port "$GDB_PORT" "GDB"
validate_path "$DEST" "destination"

# Validate LD_LIBRARY_PATH if provided
if [[ -n "$LD_LIBRARY_PATH_REMOTE" ]]; then
  # Allow colon-separated paths
  IFS=':' read -ra PATHS <<< "$LD_LIBRARY_PATH_REMOTE"
  for path in "${PATHS[@]}"; do
    validate_path "$path" "LD_LIBRARY_PATH entry"
  done
fi

if [[ ! -x "$APP_PATH" ]]; then
  echo "❌ 未找到 $APP_PATH"
  echo "👉 先执行：cmake --preset arm64-release && cmake --build --preset arm64 && cmake --install build/arm64"
  exit 1
fi

REMOTE="${USER}@${HOST}"

echo "ℹ️ 目标: $REMOTE (port=$PORT), 部署目录: $DEST"

echo "➡️  创建远端目录: $DEST/bin 和 $DEST/config"
# Security: Use printf %q for proper shell escaping
DEST_ESCAPED=$(printf %q "$DEST")
ssh -p "$PORT" "$REMOTE" "mkdir -p ${DEST_ESCAPED}/bin ${DEST_ESCAPED}/config" 2>/dev/null || true

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
    # Security: Use escaped destination path to prevent injection
    rsync -avz -e "ssh -p $PORT" "$ROOT/out/arm64/bin/" "$REMOTE:${DEST_ESCAPED}/bin/"
  else
    echo "➡️  rsync 不可用，使用 scp 复制"
    # Security: Use escaped destination path to prevent injection
    scp -P "$PORT" "$ROOT/out/arm64/bin/rk_app" "$REMOTE:${DEST_ESCAPED}/bin/"
  fi
}

sync_config() {
  echo "➡️  同步配置文件到板子"
  if [[ -d "$ROOT/config" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -avz -e "ssh -p $PORT" "$ROOT/config/" "$REMOTE:${DEST_ESCAPED}/config/"
    else
      scp -r -P "$PORT" "$ROOT/config/"* "$REMOTE:${DEST_ESCAPED}/config/"
    fi
  else
    echo "⚠️  config/ 目录不存在，跳过"
  fi
}

sync_models() {
  echo "➡️  同步模型文件到板子"
  ssh -p "$PORT" "$REMOTE" "mkdir -p ${DEST_ESCAPED}/models" 2>/dev/null || true

  # 同步 .rknn 模型文件
  if ls "$ROOT/artifacts/models/"*.rknn 1>/dev/null 2>&1; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -avz -e "ssh -p $PORT" --include='*.rknn' --include='*.json' --exclude='*' \
        "$ROOT/artifacts/models/" "$REMOTE:${DEST_ESCAPED}/models/"
    else
      scp -P "$PORT" "$ROOT/artifacts/models/"*.rknn "$REMOTE:${DEST_ESCAPED}/models/" 2>/dev/null || true
      scp -P "$PORT" "$ROOT/artifacts/models/"*.json "$REMOTE:${DEST_ESCAPED}/models/" 2>/dev/null || true
    fi
    echo "✅ 模型同步完成"
  else
    echo "⚠️  未找到 .rknn 模型文件，跳过"
  fi
}

sync_rknn_libs() {
  echo "➡️  同步 RKNN SDK 库到板子"
  ssh -p "$PORT" "$REMOTE" "mkdir -p ${DEST_ESCAPED}/lib" 2>/dev/null || true

  # 优先使用 RKNN_HOME 环境变量，否则使用默认路径
  local RKNN_LIB_SRC="${RKNN_HOME:-/opt/rknpu2}/lib"

  if [[ -d "$RKNN_LIB_SRC" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -avz -e "ssh -p $PORT" --include='*.so*' --exclude='*' \
        "$RKNN_LIB_SRC/" "$REMOTE:${DEST_ESCAPED}/lib/"
    else
      scp -P "$PORT" "$RKNN_LIB_SRC/"*.so* "$REMOTE:${DEST_ESCAPED}/lib/" 2>/dev/null || true
    fi
    echo "✅ RKNN SDK 库同步完成"
  else
    echo "⚠️  RKNN SDK 库目录不存在: $RKNN_LIB_SRC"
    echo "   请设置 RKNN_HOME 环境变量或确保 /opt/rknpu2/lib 存在"
    echo "   板子上可能需要手动安装 RKNN SDK: apt install librknpu2"
  fi
}

sync_scripts() {
  echo "➡️  同步运行脚本到板子"
  ssh -p "$PORT" "$REMOTE" "mkdir -p ${DEST_ESCAPED}/scripts" 2>/dev/null || true

  # 同步 rk3588_run.sh
  if [[ -f "$ROOT/scripts/deploy/rk3588_run.sh" ]]; then
    scp -P "$PORT" "$ROOT/scripts/deploy/rk3588_run.sh" "$REMOTE:${DEST_ESCAPED}/scripts/"
    ssh -p "$PORT" "$REMOTE" "chmod +x ${DEST_ESCAPED}/scripts/rk3588_run.sh"
    echo "✅ 运行脚本同步完成"
  fi
}

sync_assets() {
  echo "➡️  同步测试资源到板子"
  ssh -p "$PORT" "$REMOTE" "mkdir -p ${DEST_ESCAPED}/assets" 2>/dev/null || true

  # 同步测试图片（可选）
  if [[ -d "$ROOT/assets" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -avz -e "ssh -p $PORT" --include='*.jpg' --include='*.png' --exclude='*' \
        "$ROOT/assets/" "$REMOTE:${DEST_ESCAPED}/assets/" 2>/dev/null || true
    else
      scp -P "$PORT" "$ROOT/assets/"*.jpg "$REMOTE:${DEST_ESCAPED}/assets/" 2>/dev/null || true
    fi
  fi
}

echo "⬆️  部署 rk_app"
sync_bin
sync_config
sync_models
sync_rknn_libs
sync_scripts
sync_assets

case "$ACTION" in
  deploy)
    echo "✅ 部署完成：$REMOTE:$DEST/{bin,config,models,lib,scripts,assets}"
    echo ""
    echo "📋 板子上运行方式："
    echo "   cd $DEST && LD_LIBRARY_PATH=./lib ./bin/detect_cli --cfg ./config/detect_rknn.yaml"
    echo "   或: $DEST/scripts/rk3588_run.sh"
    ;;
  run)
    echo "🚀 远端运行 detect_cli"
    # 自动添加部署目录的 lib 到 LD_LIBRARY_PATH
    LD_REMOTE="${DEST}/lib"
    if [[ -n "$LD_LIBRARY_PATH_REMOTE" ]]; then
      LD_REMOTE="${LD_REMOTE}:${LD_LIBRARY_PATH_REMOTE}"
    fi
    LD_PATH_ESCAPED=$(printf %q "$LD_REMOTE")
    ssh -p "$PORT" "$REMOTE" "cd ${DEST_ESCAPED} && chmod +x bin/detect_cli && LD_LIBRARY_PATH=${LD_PATH_ESCAPED} ./bin/detect_cli --cfg ./config/detect_rknn.yaml"
    ;;
  gdb)
    echo "🐞 在板子上启动 gdbserver :$GDB_PORT"
    echo "提示: 在本机 VS Code 选择 'Attach gdbserver (ARM64 board)' 后按 F5。"
    # 自动添加部署目录的 lib 到 LD_LIBRARY_PATH
    LD_REMOTE="${DEST}/lib"
    if [[ -n "$LD_LIBRARY_PATH_REMOTE" ]]; then
      LD_REMOTE="${LD_REMOTE}:${LD_LIBRARY_PATH_REMOTE}"
    fi
    LD_PATH_ESCAPED=$(printf %q "$LD_REMOTE")
    GDB_PORT_ESCAPED=$(printf %q "$GDB_PORT")
    ssh -p "$PORT" "$REMOTE" "cd ${DEST_ESCAPED} && chmod +x bin/detect_cli && exec env LD_LIBRARY_PATH=${LD_PATH_ESCAPED} gdbserver :${GDB_PORT_ESCAPED} ./bin/detect_cli --cfg ./config/detect_rknn.yaml"
    ;;
  kill)
    echo "🧹 结束远端 gdbserver (best-effort)"
    # Fixed: Use hardcoded pattern to avoid injection via process name
    ssh -p "$PORT" "$REMOTE" "pkill -f 'gdbserver.*rk_app'" || true
    ;;
  *)
    echo "内部错误：未知 ACTION=$ACTION"; exit 3
    ;;
esac
