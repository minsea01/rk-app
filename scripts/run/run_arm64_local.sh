#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT"

APP_NAME="${APP_NAME:-rk_app}"
APP_PATH="${ROOT}/out/arm64/bin/${APP_NAME}"

if [[ ! -x "$APP_PATH" ]]; then
  echo "❌ 没找到 ${APP_PATH}"
  echo "👉 先执行：cmake --preset arm64-release && cmake --build --preset arm64 && cmake --install build/arm64"
  exit 1
fi

if "$APP_PATH" --config "$ROOT/config/app.yaml" 2>/dev/null; then exit 0; fi

if command -v qemu-aarch64 >/dev/null 2>&1; then
  echo "===== Running on qemu ====="
  echo "ℹ️ 使用 qemu-aarch64 -L /usr/aarch64-linux-gnu"
  exec qemu-aarch64 -L /usr/aarch64-linux-gnu "$APP_PATH" --config "$ROOT/config/app.yaml"
elif command -v qemu-aarch64-static >/dev/null 2>&1; then
  echo "===== Running on qemu ====="
  echo "ℹ️ 使用 qemu-aarch64-static + QEMU_LD_PREFIX"
  exec env QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64-static "$APP_PATH" --config "$ROOT/config/app.yaml"
else
  echo "❌ 未找到 qemu-aarch64 / qemu-aarch64-static"
  exit 1
fi
