#!/usr/bin/env bash
set -euo pipefail

# Minimal runner for RK3588: prepares LD_LIBRARY_PATH and runs detect_cli (RKNN)
# Fallback to Python RKNNLite runner when CLI binary is unavailable.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT_BIN="$ROOT_DIR/out/arm64/bin/detect_cli"
OUT_LIB="$ROOT_DIR/out/arm64/lib"
CFG="$ROOT_DIR/config/detection/detect_rknn.yaml"
MODEL_DEFAULT="$ROOT_DIR/artifacts/models/best.rknn"
NAMES_DEFAULT="$ROOT_DIR/config/industrial_classes.txt"

usage() {
  echo "Usage: $0 [--cfg <yaml>] [--model <rknn>] [--runner cli|python] [-- core args...]"
  echo "- Defaults: --cfg config/detection/detect_rknn.yaml, --model artifacts/models/best.rknn"
}

RUNNER="cli"
MODEL_OVERRIDE=""
CFG_OVERRIDE=""
if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then usage; exit 0; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg) CFG_OVERRIDE="$2"; shift 2;;
    --model) MODEL_OVERRIDE="$2"; shift 2;;
    --runner) RUNNER="$2"; shift 2;;
    --) shift; break;;
    *) break;;
  esac
done

CFG=${CFG_OVERRIDE:-$CFG}
MODEL=${MODEL_OVERRIDE:-$MODEL_DEFAULT}

# Prefer RKNN_HOME if set, otherwise rely on installed libs or out/arm64/lib
if [[ -d "$OUT_LIB" ]]; then
  export LD_LIBRARY_PATH="$OUT_LIB:${LD_LIBRARY_PATH-}"
fi
if [[ -n "${RKNN_HOME-}" && -d "$RKNN_HOME/lib" ]]; then
  export LD_LIBRARY_PATH="$RKNN_HOME/lib:$LD_LIBRARY_PATH"
fi

echo "[rk3588_run] ROOT=$ROOT_DIR"
echo "[rk3588_run] CFG=$CFG"
echo "[rk3588_run] MODEL=$MODEL"
echo "[rk3588_run] RUNNER=$RUNNER"

run_cli() {
  if [[ ! -x "$OUT_BIN" ]]; then
    echo "[rk3588_run] detect_cli not found: $OUT_BIN" >&2
    return 1
  fi
  echo "[rk3588_run] Running detect_cli..."
  exec "$OUT_BIN" --cfg "$CFG" "$@"
}

run_py() {
  local PY="$ROOT_DIR/apps/yolov8_rknn_infer.py"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found" >&2; return 1; fi
  if [[ ! -f "$PY" ]]; then
    echo "Python runner not found: $PY" >&2; return 1; fi
  echo "[rk3588_run] Running Python RKNNLite runner..."
  # If no explicit --source is passed, use a demo image if present
  local HAS_SOURCE=0
  for a in "$@"; do [[ $a == --source ]] && HAS_SOURCE=1; done
  if [[ $HAS_SOURCE -eq 0 && -f "$ROOT_DIR/assets/test.jpg" ]]; then
    set -- --source "$ROOT_DIR/assets/test.jpg" "$@"
  fi
  exec python3 "$PY" --model "$MODEL" --names "$NAMES_DEFAULT" "$@"
}

cd "$ROOT_DIR"

if [[ "$RUNNER" == "cli" ]]; then
  run_cli "$@" || { echo "[rk3588_run] Falling back to Python runner"; run_py "$@"; }
else
  run_py "$@"
fi
