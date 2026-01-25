#!/usr/bin/env bash
# Lazy one-click COCO full training (all classes) for AutoDL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/pedestrian_training}"
COCO_ROOT="${COCO_ROOT:-/root/autodl-tmp/coco}"

echo "========================================="
echo "YOLOv8n COCO Full Training (Lazy Script)"
echo "========================================="

# 1) Setup environment and weights
if [ -x "$SCRIPT_DIR/setup_autodl.sh" ]; then
  "$SCRIPT_DIR/setup_autodl.sh"
else
  if ! command -v yolo >/dev/null 2>&1; then
    python3 -m pip install -q "ultralytics>=8.0.0" "albumentations>=1.0.0"
  fi
  mkdir -p "$WORK_DIR"/{datasets,models,outputs}
  if [ ! -f "$WORK_DIR/models/yolov8n.pt" ]; then
    if command -v wget >/dev/null 2>&1; then
      wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt \
        -O "$WORK_DIR/models/yolov8n.pt"
    elif command -v curl >/dev/null 2>&1; then
      curl -L -s https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt \
        -o "$WORK_DIR/models/yolov8n.pt"
    else
      echo "wget/curl not found. Please download yolov8n.pt manually."
      exit 1
    fi
  fi
fi

# 2) Check COCO dataset (YOLO format required)
if [ ! -d "$COCO_ROOT/images/train2017" ] || [ ! -d "$COCO_ROOT/labels/train2017" ]; then
  echo "COCO YOLO-format dataset not found at: $COCO_ROOT"
  echo "Expected: $COCO_ROOT/images/train2017 and $COCO_ROOT/labels/train2017"
  exit 1
fi

# 3) Create COCO YAML (all 80 classes)
mkdir -p "$WORK_DIR/datasets"
COCO_YAML="$WORK_DIR/datasets/coco_full_autodl.yaml"
cat > "$COCO_YAML" <<EOF
path: $COCO_ROOT
train: images/train2017
val: images/val2017

nc: 80
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
EOF

# 4) Train
cd "$WORK_DIR"

EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
CACHE_MODE="${CACHE_MODE:-disk}"
PROJECT="${PROJECT:-outputs}"
NAME="${NAME:-yolov8n_coco_full}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

yolo detect train \
  model="$WORK_DIR/models/yolov8n.pt" \
  data="$COCO_YAML" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device="$DEVICE" \
  project="$PROJECT" \
  name="$NAME" \
  patience=80 \
  save=True \
  save_period=25 \
  val=True \
  plots=True \
  exist_ok=True \
  pretrained=True \
  optimizer=AdamW \
  lr0=0.0005 \
  lrf=0.001 \
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=5 \
  warmup_momentum=0.8 \
  box=7.5 \
  cls=0.5 \
  dfl=1.5 \
  hsv_h=0.02 \
  hsv_s=0.8 \
  hsv_v=0.5 \
  degrees=10.0 \
  translate=0.15 \
  scale=0.6 \
  shear=5.0 \
  perspective=0.0005 \
  flipud=0.0 \
  fliplr=0.5 \
  mosaic=1.0 \
  mixup=0.15 \
  copy_paste=0.1 \
  erasing=0.4 \
  workers="$WORKERS" \
  cache="$CACHE_MODE" \
  amp=True

echo ""
echo "========================================="
echo "Done."
echo "Best model: $PROJECT/$NAME/weights/best.pt"
echo "========================================="
