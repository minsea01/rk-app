#!/usr/bin/env bash
set -euo pipefail

# Build and install a clean RK3588 runtime tree on the board.
# The legacy workspace is left untouched; the destination is atomically swapped.

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

BOARD_HOST="${BOARD_HOST:-192.168.137.56}"
BOARD_USER="${BOARD_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
DEST="${DEST:-/opt/rk_app_current}"
REMOTE_SRC="${REMOTE_SRC:-/tmp/rk_app_clean_src}"
RKNN_HOME_REMOTE="${RKNN_HOME_REMOTE:-/home/RKnpuProjects/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api}"
LEGACY_ROOT="${LEGACY_ROOT:-/root/rk-app-new}"
BUILD_JOBS="${BUILD_JOBS:-}"
GIT_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ "$GIT_COMMIT" != "unknown" ]] && ! git -C "$ROOT" diff --quiet -- . 2>/dev/null; then
  GIT_COMMIT="${GIT_COMMIT}+dirty"
fi

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --host <ip>       Board IP (default: ${BOARD_HOST})
  --user <name>     SSH user (default: ${BOARD_USER})
  --port <n>        SSH port (default: ${SSH_PORT})
  --dest <path>     Runtime destination (default: ${DEST})
  --remote-src <p>  Remote build source dir (default: ${REMOTE_SRC})
  --jobs <n>        Build jobs on board (default: nproc)
  -h, --help        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) BOARD_HOST="$2"; shift 2 ;;
    --user) BOARD_USER="$2"; shift 2 ;;
    --port) SSH_PORT="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --remote-src) REMOTE_SRC="$2"; shift 2 ;;
    --jobs) BUILD_JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required" >&2
  exit 1
fi

SSH=(ssh -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no)
RSYNC_RSH="ssh -p $SSH_PORT -o BatchMode=yes -o StrictHostKeyChecking=no"
REMOTE="${BOARD_USER}@${BOARD_HOST}"

echo "[deploy] target: ${REMOTE}:${DEST}"
echo "[deploy] source commit: ${GIT_COMMIT}"

"${SSH[@]}" "$REMOTE" "rm -rf '$REMOTE_SRC' && mkdir -p '$REMOTE_SRC/artifacts/models'"

rsync -az --delete \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='htmlcov/' \
  --exclude='build/' \
  -e "$RSYNC_RSH" \
  "$ROOT"/apps "$ROOT"/config "$ROOT"/include "$ROOT"/src "$ROOT"/examples "$ROOT"/scripts \
  "$ROOT"/assets "$ROOT"/CMakeLists.txt "$ROOT"/CMakePresets.json \
  "$ROOT"/requirements.txt "$ROOT"/requirements_board.txt "$ROOT"/pyproject.toml "$ROOT"/setup.py \
  "$REMOTE:$REMOTE_SRC/"

rsync -az --delete \
  --include='*.rknn' --exclude='*' \
  -e "$RSYNC_RSH" \
  "$ROOT/artifacts/models/" "$REMOTE:$REMOTE_SRC/artifacts/models/"

"${SSH[@]}" "$REMOTE" "bash -s" -- "$REMOTE_SRC" "$DEST" "$RKNN_HOME_REMOTE" "$LEGACY_ROOT" "$GIT_COMMIT" "$BUILD_JOBS" <<'REMOTE'
set -euo pipefail

REMOTE_SRC="$1"
DEST="$2"
RKNN_HOME_REMOTE="$3"
LEGACY_ROOT="$4"
GIT_COMMIT="$5"
BUILD_JOBS="${6:-}"

cd "$REMOTE_SRC"

mkdir -p artifacts/models
cp -a "$LEGACY_ROOT"/artifacts/models/*.rknn.json artifacts/models/ 2>/dev/null || true
python3 - <<'PY'
import json
from pathlib import Path

models = Path("artifacts/models")
sidecars = {
    "best_person_aug_416_norm_int8.rknn.json": {
        "head": "raw",
        "num_classes": 1,
        "has_objectness": 0,
        "score_is_probability": 1,
        "coords_are_normalized": 1,
        "output_index": 0,
    },
    "yolo11n_coco80_416_int8.rknn.json": {
        "head": "raw",
        "num_classes": 80,
        "has_objectness": 0,
        "score_is_probability": 1,
        "coords_are_normalized": 1,
        "output_index": 0,
    },
}
for name, data in sidecars.items():
    path = models / name
    if not path.exists() and (models / name.removesuffix(".json")).exists():
        path.write_text(json.dumps(data, indent=2) + "\n")
PY

cp -a "$LEGACY_ROOT"/artifacts/fake_camera_416_30fps_60s.avi artifacts/ 2>/dev/null || true

jobs="$BUILD_JOBS"
if [[ -z "$jobs" ]]; then
  jobs="$(nproc)"
fi

cmake -S . -B build/board-clean -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_RKNN=ON \
  -DENABLE_GIGE=ON \
  -DENABLE_CSI=ON \
  -DENABLE_TESTS=OFF \
  -DBUILD_TESTING=OFF \
  -DRKNN_HOME="$RKNN_HOME_REMOTE"
cmake --build build/board-clean -j"$jobs" --target detect_cli bench_e2e_cpp detect_rknn_multicore detect_pipeline

rm -rf "${DEST}.new"
mkdir -p "${DEST}.new"
cmake --install build/board-clean --prefix "${DEST}.new"
rsync -a config apps scripts assets "${DEST}.new/"

mkdir -p "${DEST}.new/artifacts/models"
rsync -a artifacts/models/ "${DEST}.new/artifacts/models/"
find "${DEST}.new/artifacts/models" -maxdepth 1 -type f \
  ! -name best_person_aug_416_norm_int8.rknn \
  ! -name best_person_aug_416_norm_int8.rknn.json \
  ! -name yolo11n_coco80_416_int8.rknn \
  ! -name yolo11n_coco80_416_int8.rknn.json \
  -delete
cp -a artifacts/fake_camera_416_30fps_60s.avi "${DEST}.new/artifacts/" 2>/dev/null || true

ln -sfn artifacts/models "${DEST}.new/models"
mkdir -p "${DEST}.new/build/board"
ln -sfn ../../bin/detect_cli "${DEST}.new/build/board/detect_cli"
cp -a scripts/deploy/rk3588_run.sh "${DEST}.new/scripts/rk3588_run.sh"
: > "${DEST}.new/.rkapp-root"

chmod +x "${DEST}.new"/bin/* "${DEST}.new"/scripts/rk3588_run.sh \
  "${DEST}.new"/scripts/deploy/*.sh "${DEST}.new"/scripts/demo/*.sh 2>/dev/null || true
chown -hR root:root "${DEST}.new"
chmod -R go-w "${DEST}.new"

rm -rf "${DEST}.prev"
if [[ -d "$DEST" ]]; then
  mv "$DEST" "${DEST}.prev"
fi
mv "${DEST}.new" "$DEST"

cat > "$DEST/DEPLOYMENT_MANIFEST.txt" <<MANIFEST
rk-app clean board deployment
source_commit: ${GIT_COMMIT}
created_at: $(date -Is)
layout: clean runtime, legacy ${LEGACY_ROOT} preserved
root_marker: ${DEST}/.rkapp-root
build_flags: Release ENABLE_RKNN=ON ENABLE_GIGE=ON ENABLE_CSI=ON ENABLE_RGA=ON
entry_cli: ${DEST}/bin/detect_cli
entry_runner: ${DEST}/scripts/rk3588_run.sh
compat_cli: ${DEST}/build/board/detect_cli
main_model: ${DEST}/artifacts/models/best_person_aug_416_norm_int8.rknn
coco80_model: ${DEST}/artifacts/models/yolo11n_coco80_416_int8.rknn
owner: root:root
permissions: no group/world writable regular files under ${DEST}
MANIFEST

du -sh "$DEST"
find "$DEST" -type f -perm -002 -print | sed -n '1,20p'
REMOTE

echo "[deploy] done: ${REMOTE}:${DEST}"
