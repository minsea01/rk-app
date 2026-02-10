#!/usr/bin/env bash
# Unified training runner for cloud_training scripts.
#
# Purpose:
#   - Keep YOLO train/export/report logic in one place.
#   - Let scenario scripts only handle dataset/environment preparation.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_runner.sh --model <pt> --data <yaml> --epochs <n> --imgsz <n> --batch <n> [options]

Core options:
  --profile <name>            One of: none, baseline, map90, extreme_stage1, extreme_stage2
  --workdir <path>            cd into this directory before running
  --model <path>              Model weights or model yaml
  --data <path>               Dataset yaml
  --epochs <n>
  --imgsz <n>
  --batch <n|auto|float>
  --device <id>               Default: 0
  --project <dir>             Default: outputs
  --name <run_name>           Default: exp

Training options:
  --patience <n>
  --save-period <n>
  --workers <n>
  --cache <mode>
  --classes <ids>
  --optimizer <name>
  --lr0 <float>
  --lrf <float>
  --momentum <float>
  --weight-decay <float>
  --warmup-epochs <float>
  --warmup-momentum <float>
  --save <True|False>         Default: True
  --val <True|False>          Default: True
  --plots <True|False>        Default: True
  --exist-ok <True|False>     Default: True
  --pretrained <True|False>   Default: True
  --amp <True|False>          Default: True
  --extra <k=v>               Repeatable. Passed to yolo train after profile args.

Export/report options:
  --no-export                 Disable ONNX export
  --export-imgsz <n>          Default: same as --imgsz
  --export-opset <n>          Default: 12
  --no-summary                Disable csv/model-size summary

Example:
  ./train_runner.sh \
    --profile baseline \
    --workdir /root/autodl-tmp/citypersons \
    --model yolov8n.pt \
    --data datasets/citypersons_yolo/citypersons.yaml \
    --epochs 150 --imgsz 640 --batch 64 \
    --name yolov8n_citypersons \
    --cache disk --workers 8
EOF
}

# Defaults
PROFILE="baseline"
WORKDIR=""
MODEL=""
DATA=""
EPOCHS=""
IMGSZ=""
BATCH=""
DEVICE="0"
PROJECT="outputs"
NAME="exp"

PATIENCE=""
SAVE_PERIOD=""
WORKERS=""
CACHE=""
CLASSES=""

OPTIMIZER=""
LR0=""
LRF=""
MOMENTUM=""
WEIGHT_DECAY=""
WARMUP_EPOCHS=""
WARMUP_MOMENTUM=""

SAVE="True"
VAL="True"
PLOTS="True"
EXIST_OK="True"
PRETRAINED="True"
AMP="True"

DO_EXPORT="true"
EXPORT_IMGSZ=""
EXPORT_OPSET="12"
DO_SUMMARY="true"

declare -a EXTRA_ARGS=()
declare -a PROFILE_ARGS=()

# First pass: pick profile regardless of argument order.
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--profile" ]] && ((i + 1 < ${#ARGS[@]})); then
    PROFILE="${ARGS[$((i + 1))]}"
  fi
done

apply_profile_defaults() {
  PROFILE_ARGS=()
  case "$PROFILE" in
    none)
      :
      ;;
    baseline)
      PATIENCE="${PATIENCE:-50}"
      SAVE_PERIOD="${SAVE_PERIOD:-10}"
      OPTIMIZER="${OPTIMIZER:-AdamW}"
      LR0="${LR0:-0.001}"
      LRF="${LRF:-0.01}"
      PROFILE_ARGS=(
        "mosaic=1.0"
        "mixup=0.15"
        "copy_paste=0.1"
        "degrees=5.0"
        "translate=0.1"
        "scale=0.5"
        "fliplr=0.5"
      )
      ;;
    map90)
      PATIENCE="${PATIENCE:-80}"
      SAVE_PERIOD="${SAVE_PERIOD:-25}"
      OPTIMIZER="${OPTIMIZER:-AdamW}"
      LR0="${LR0:-0.0005}"
      LRF="${LRF:-0.001}"
      MOMENTUM="${MOMENTUM:-0.937}"
      WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
      WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
      WARMUP_MOMENTUM="${WARMUP_MOMENTUM:-0.8}"
      PROFILE_ARGS=(
        "box=7.5"
        "cls=0.5"
        "dfl=1.5"
        "hsv_h=0.02"
        "hsv_s=0.8"
        "hsv_v=0.5"
        "degrees=10.0"
        "translate=0.15"
        "scale=0.6"
        "shear=5.0"
        "perspective=0.0005"
        "flipud=0.0"
        "fliplr=0.5"
        "mosaic=1.0"
        "mixup=0.15"
        "copy_paste=0.1"
        "erasing=0.4"
      )
      ;;
    extreme_stage1)
      PATIENCE="${PATIENCE:-80}"
      SAVE_PERIOD="${SAVE_PERIOD:-25}"
      OPTIMIZER="${OPTIMIZER:-AdamW}"
      LR0="${LR0:-0.0001}"
      LRF="${LRF:-0.0001}"
      MOMENTUM="${MOMENTUM:-0.937}"
      WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
      WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
      PROFILE_ARGS=(
        "box=7.5"
        "cls=0.5"
        "dfl=1.5"
        "hsv_h=0.02"
        "hsv_s=0.8"
        "hsv_v=0.5"
        "degrees=10.0"
        "translate=0.2"
        "scale=0.7"
        "shear=5.0"
        "perspective=0.001"
        "flipud=0.0"
        "fliplr=0.5"
        "mosaic=1.0"
        "mixup=0.2"
        "copy_paste=0.15"
        "erasing=0.5"
      )
      ;;
    extreme_stage2)
      PATIENCE="${PATIENCE:-100}"
      SAVE_PERIOD="${SAVE_PERIOD:-30}"
      OPTIMIZER="${OPTIMIZER:-SGD}"
      LR0="${LR0:-0.001}"
      LRF="${LRF:-0.0001}"
      MOMENTUM="${MOMENTUM:-0.937}"
      WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
      WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
      PROFILE_ARGS=(
        "box=7.5"
        "cls=0.5"
        "dfl=1.5"
        "hsv_h=0.03"
        "hsv_s=0.9"
        "hsv_v=0.6"
        "degrees=15.0"
        "translate=0.25"
        "scale=0.8"
        "shear=10.0"
        "perspective=0.001"
        "flipud=0.0"
        "fliplr=0.5"
        "mosaic=1.0"
        "mixup=0.3"
        "copy_paste=0.2"
        "erasing=0.6"
      )
      ;;
    *)
      echo "[ERROR] Unknown profile: $PROFILE" >&2
      echo "        Allowed: none, baseline, map90, extreme_stage1, extreme_stage2" >&2
      exit 2
      ;;
  esac
}

apply_profile_defaults

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --save-period) SAVE_PERIOD="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --lr0) LR0="$2"; shift 2 ;;
    --lrf) LRF="$2"; shift 2 ;;
    --momentum) MOMENTUM="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --warmup-epochs) WARMUP_EPOCHS="$2"; shift 2 ;;
    --warmup-momentum) WARMUP_MOMENTUM="$2"; shift 2 ;;
    --save) SAVE="$2"; shift 2 ;;
    --val) VAL="$2"; shift 2 ;;
    --plots) PLOTS="$2"; shift 2 ;;
    --exist-ok) EXIST_OK="$2"; shift 2 ;;
    --pretrained) PRETRAINED="$2"; shift 2 ;;
    --amp) AMP="$2"; shift 2 ;;
    --extra) EXTRA_ARGS+=("$2"); shift 2 ;;
    --no-export) DO_EXPORT="false"; shift ;;
    --export-imgsz) EXPORT_IMGSZ="$2"; shift 2 ;;
    --export-opset) EXPORT_OPSET="$2"; shift 2 ;;
    --no-summary) DO_SUMMARY="false"; shift ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$DATA" || -z "$EPOCHS" || -z "$IMGSZ" || -z "$BATCH" ]]; then
  echo "[ERROR] Missing required args: --model --data --epochs --imgsz --batch" >&2
  usage
  exit 2
fi

if [[ -z "$PATIENCE" ]]; then PATIENCE=50; fi
if [[ -z "$SAVE_PERIOD" ]]; then SAVE_PERIOD=10; fi
if [[ -z "$EXPORT_IMGSZ" ]]; then EXPORT_IMGSZ="$IMGSZ"; fi

if [[ -n "$WORKDIR" ]]; then
  mkdir -p "$WORKDIR"
  cd "$WORKDIR"
fi

if ! command -v yolo >/dev/null 2>&1; then
  echo "[ERROR] 'yolo' command not found. Install ultralytics first." >&2
  exit 127
fi

TRAIN_CMD=(
  yolo detect train
  "model=$MODEL"
  "data=$DATA"
  "epochs=$EPOCHS"
  "imgsz=$IMGSZ"
  "batch=$BATCH"
  "device=$DEVICE"
  "project=$PROJECT"
  "name=$NAME"
  "patience=$PATIENCE"
  "save=$SAVE"
  "save_period=$SAVE_PERIOD"
  "val=$VAL"
  "plots=$PLOTS"
  "exist_ok=$EXIST_OK"
  "pretrained=$PRETRAINED"
  "amp=$AMP"
)

if [[ -n "$WORKERS" ]]; then TRAIN_CMD+=("workers=$WORKERS"); fi
if [[ -n "$CACHE" ]]; then TRAIN_CMD+=("cache=$CACHE"); fi
if [[ -n "$CLASSES" ]]; then TRAIN_CMD+=("classes=$CLASSES"); fi
if [[ -n "$OPTIMIZER" ]]; then TRAIN_CMD+=("optimizer=$OPTIMIZER"); fi
if [[ -n "$LR0" ]]; then TRAIN_CMD+=("lr0=$LR0"); fi
if [[ -n "$LRF" ]]; then TRAIN_CMD+=("lrf=$LRF"); fi
if [[ -n "$MOMENTUM" ]]; then TRAIN_CMD+=("momentum=$MOMENTUM"); fi
if [[ -n "$WEIGHT_DECAY" ]]; then TRAIN_CMD+=("weight_decay=$WEIGHT_DECAY"); fi
if [[ -n "$WARMUP_EPOCHS" ]]; then TRAIN_CMD+=("warmup_epochs=$WARMUP_EPOCHS"); fi
if [[ -n "$WARMUP_MOMENTUM" ]]; then TRAIN_CMD+=("warmup_momentum=$WARMUP_MOMENTUM"); fi

for kv in "${PROFILE_ARGS[@]}"; do
  TRAIN_CMD+=("$kv")
done

for kv in "${EXTRA_ARGS[@]}"; do
  TRAIN_CMD+=("$kv")
done

echo "========================================="
echo "[train_runner] Profile: $PROFILE"
echo "[train_runner] Model:   $MODEL"
echo "[train_runner] Data:    $DATA"
echo "[train_runner] Run:     $PROJECT/$NAME"
echo "========================================="

"${TRAIN_CMD[@]}"

BEST_PT="$PROJECT/$NAME/weights/best.pt"
if [[ ! -f "$BEST_PT" ]]; then
  echo "[ERROR] Training finished but best weights not found: $BEST_PT" >&2
  exit 1
fi

if [[ "$DO_EXPORT" == "true" ]]; then
  echo "[train_runner] Exporting ONNX (imgsz=$EXPORT_IMGSZ, opset=$EXPORT_OPSET)"
  yolo export model="$BEST_PT" format=onnx opset="$EXPORT_OPSET" simplify=True imgsz="$EXPORT_IMGSZ"
fi

if [[ "$DO_SUMMARY" == "true" ]]; then
  RESULTS_CSV="$PROJECT/$NAME/results.csv"
  ONNX_PATH="${BEST_PT%.pt}.onnx"
  export RESULTS_CSV BEST_PT ONNX_PATH
  python3 <<'PYEOF'
import csv
import os

results_csv = os.getenv("RESULTS_CSV", "")
best_pt = os.getenv("BEST_PT", "")
onnx_path = os.getenv("ONNX_PATH", "")

print("\n[train_runner] Summary")
if os.path.exists(results_csv):
    with open(results_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows:
        last = rows[-1]
        map50 = None
        for key, value in last.items():
            if "mAP50" in key and "mAP50-95" not in key:
                try:
                    map50 = float(str(value).strip())
                except Exception:
                    map50 = None
                break
        if map50 is not None:
            print(f"  Final mAP@0.5: {map50 * 100:.2f}%")
else:
    print(f"  results.csv not found: {results_csv}")

if os.path.exists(best_pt):
    pt_mb = os.path.getsize(best_pt) / 1024 / 1024
    print(f"  Best PT size: {pt_mb:.2f} MB")

if os.path.exists(onnx_path):
    onnx_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  Best ONNX size: {onnx_mb:.2f} MB")
PYEOF
fi

echo "[train_runner] Done: $BEST_PT"
