#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
APP_PATH="$ROOT/out/x86/bin/rk_app"

if [[ ! -x "$APP_PATH" ]]; then
  echo "❌ 未找到 $APP_PATH"
  echo "👉 先执行：cmake --preset x86-debug && cmake --build --preset x86-debug && cmake --install build/x86-debug"
  exit 1
fi

echo "===== Running x86 rk_app ====="
"$APP_PATH" --config "$ROOT/config/app.yaml"
echo "===== End ====="
