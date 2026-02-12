#!/usr/bin/env bash
# Safe workspace cleanup helper (whitelist only).
# Default mode is dry-run. Use --apply to actually delete files.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=1
if [[ "${1:-}" == "--apply" ]]; then
  DRY_RUN=0
elif [[ "${1:-}" == "--dry-run" || -z "${1:-}" ]]; then
  DRY_RUN=1
else
  echo "Usage: $0 [--dry-run|--apply]"
  exit 1
fi

# Whitelist: only root-level generated junk and accidental model drops.
PATTERNS=(
  "=4.1.0"
  "=7.4.0"
  "check*_*.onnx"
  "bench_dfl_opt"
  "test_bugfix_arm64"
  "yolo11n.onnx"
  "yolo11n.pt"
  "yolov8n.pt"
)

found_any=0
for pattern in "${PATTERNS[@]}"; do
  matches=("$ROOT_DIR"/$pattern)
  for candidate in "${matches[@]}"; do
    if [[ ! -e "$candidate" && ! -L "$candidate" ]]; then
      continue
    fi
    rel="${candidate#$ROOT_DIR/}"
    found_any=1
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY-RUN] would remove: $rel"
    else
      rm -f -- "$candidate"
      echo "[APPLY] removed: $rel"
    fi
  done
done

if [[ "$found_any" -eq 0 ]]; then
  echo "No whitelist targets found in workspace root."
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry-run complete. Re-run with --apply to delete listed files."
else
  echo "Cleanup complete."
fi
