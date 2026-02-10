#!/bin/bash
# Fine-tune YOLO11n on CityPersons for Pedestrian Detection
# Target: >= 90% mAP@0.5 on COCO person validation set

set -e

echo "============================================="
echo "YOLO11n CityPersons Fine-tuning"
echo "============================================="

# Configuration
MODEL="yolo11n.pt"
DATA="datasets/citypersons/yolo/citypersons.yaml"
EPOCHS=50
IMGSZ=640
BATCH=16
DEVICE="0"  # Use GPU 0 (change to "cpu" if no GPU)
PROJECT="runs/citypersons_finetune"
NAME="yolo11n_citypersons"

# Advanced settings
PATIENCE=10  # Early stopping patience
LR0=0.01     # Initial learning rate
WARMUP_EPOCHS=3

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATA"
echo "  Epochs: $EPOCHS"
echo "  Image size: ${IMGSZ}x${IMGSZ}"
echo "  Batch size: $BATCH"
echo "  Device: $DEVICE"
echo ""

# Check if dataset exists
if [ ! -f "$DATA" ]; then
    echo "❌ Dataset YAML not found: $DATA"
    echo ""
    echo "Please prepare the dataset first:"
    echo "  1. bash scripts/datasets/download_citypersons.sh"
    echo "  2. python scripts/datasets/prepare_citypersons.py"
    exit 1
fi

# Activate virtual environment
source ~/yolo_env/bin/activate

echo "Starting training..."
echo ""

# Train with Ultralytics YOLO
yolo train \
    model=$MODEL \
    data=$DATA \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    batch=$BATCH \
    device=$DEVICE \
    project=$PROJECT \
    name=$NAME \
    patience=$PATIENCE \
    lr0=$LR0 \
    warmup_epochs=$WARMUP_EPOCHS \
    save=True \
    save_period=5 \
    plots=True \
    verbose=True \
    exist_ok=False

echo ""
echo "============================================="
echo "Training Complete!"
echo "============================================="

# Find best weights
BEST_WEIGHTS="$PROJECT/$NAME/weights/best.pt"

if [ -f "$BEST_WEIGHTS" ]; then
    echo ""
    echo "✓ Best weights saved: $BEST_WEIGHTS"
    echo ""
    echo "Next steps:"
    echo "  1. Validate on COCO person:"
    echo "     python scripts/evaluation/official_yolo_map.py \\"
    echo "       --model $BEST_WEIGHTS \\"
    echo "       --annotations datasets/coco/annotations/person_val2017.json \\"
    echo "       --images-dir datasets/coco/val2017 \\"
    echo "       --output artifacts/yolo11n_citypersons_finetuned_map.json"
    echo ""
    echo "  2. Export to ONNX:"
    echo "     yolo export model=$BEST_WEIGHTS format=onnx opset=12 simplify=True"
    echo ""
    echo "  3. Convert to RKNN:"
    echo "     python tools/convert_onnx_to_rknn.py \\"
    echo "       --onnx ${BEST_WEIGHTS%.pt}.onnx \\"
    echo "       --out artifacts/models/yolo11n_citypersons.rknn"
else
    echo "❌ Training failed - weights not found"
    exit 1
fi
