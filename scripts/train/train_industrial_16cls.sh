#!/usr/bin/env bash
set -euo pipefail

# Params
DS_YAML=${1:-/home/minsea01/datasets/coco80.yaml}
EPOCHS=${EPOCHS:-50}
IMG=${IMG:-640}
BATCH=${BATCH:-16}
DEVICE=${DEVICE:-0}
RUN_NAME=${RUN_NAME:-industrial_16cls_coco}

CLASSES="0,1,2,3,5,7,9,11,15,24,26,39,63,67,73,76"

ROOT=~/dev/yolo-projects/coco4cls-vscode
mkdir -p "$ROOT" && cd "$ROOT"

echo "[Train] dataset: $DS_YAML" 
echo "[Train] classes: $CLASSES"

# Preflight: validate dataset YAML presence and contents
if [ ! -f "$DS_YAML" ]; then
  echo "[ERROR] Dataset YAML not found: $DS_YAML" >&2
  echo "        Please point to an existing data.yaml (e.g., /home/minsea01/datasets/coco80.yaml)" >&2
  exit 3
fi

python3 - "$DS_YAML" <<'PY'
import sys, os, yaml, glob
from pathlib import Path

def has_images(p: Path) -> bool:
    if p.is_file():
        try:
            # treat as list file
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if Path(s).exists():
                        return True
            return False
        except Exception:
            return False
    # directory: search common image extensions
    exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
    for root, _, files in os.walk(p):
        if any(Path(fn).suffix.lower() in exts for fn in files):
            return True
    return False

data = Path(sys.argv[1])
try:
    cfg = yaml.safe_load(data.read_text())
except Exception as e:
    print(f"[ERROR] Failed to parse YAML: {e}", file=sys.stderr)
    sys.exit(3)

root = Path(cfg.get('path') or '.')
train = Path(cfg.get('train') or '')
val = Path(cfg.get('val') or '')

def resolve(p: Path) -> Path:
    if not p:
        return p
    # If p is absolute or exists as-is, use it; else join with root
    if p.is_absolute() or p.exists():
        return p
    if root:
        q = (root / p).resolve()
        return q
    return p

train_r = resolve(train)
val_r = resolve(val)

errors = []
if not train_r:
    errors.append('train path missing in YAML')
elif not train_r.exists():
    errors.append(f'train not found: {train_r}')
elif not has_images(train_r):
    errors.append(f'no images found under/listed by: {train_r}')

if not val_r:
    errors.append('val path missing in YAML')
elif not val_r.exists():
    errors.append(f'val not found: {val_r}')
elif not has_images(val_r):
    errors.append(f'no images found under/listed by: {val_r}')

if errors:
    print('[ERROR] Dataset YAML validation failed:')
    for e in errors:
        print('  -', e)
    sys.exit(3)
else:
    print('[OK] Dataset YAML validated.')
PY
if [ $? -ne 0 ]; then
  echo "[ABORT] Fix dataset YAML, then rerun." >&2
  exit 3
fi

# Python env (setup only after dataset is validated)
if [ ! -d "$HOME/yolo-train" ]; then
  python3 -m venv "$HOME/yolo-train"
fi
source "$HOME/yolo-train/bin/activate"
python -m pip install -U pip ultralytics

yolo detect train \
  data="$DS_YAML" \
  model=yolo11s.pt \
  epochs="$EPOCHS" \
  imgsz="$IMG" \
  batch="$BATCH" \
  device="$DEVICE" \
  classes="$CLASSES" \
  name="$RUN_NAME"

BEST_PT=$(ls -t "$ROOT"/runs/detect/${RUN_NAME}*/weights/best.pt | head -1)
echo "Using model: $BEST_PT"

yolo export model="$BEST_PT" format=onnx imgsz="$IMG" simplify=True dynamic=False half=False opset=17

BEST_ONNX="$(dirname "$BEST_PT")/best.onnx"
mkdir -p "$HOME/models"
cp "$BEST_ONNX" "$HOME/models/industrial_16cls_${IMG}.onnx"
echo "ONNX saved to: $HOME/models/industrial_16cls_${IMG}.onnx"

deactivate || true
