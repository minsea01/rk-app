#!/usr/bin/env bash
# Unified training runner for cloud_training scripts.
#
# Purpose:
#   - Keep YOLO train/export/report logic in one place.
#   - Let scenario scripts only handle dataset/environment preparation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  train.sh --model <pt> --data <yaml> --epochs <n> --imgsz <n> --batch <n> [options]

Core options:
  --task <name>               YOLO task: detect, pose, segment (default: detect)
  --exp <yaml>                Experiment YAML (config/experiments/*.yaml)
  --set <k=v>                 Override key/value from --exp (repeatable)
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
  ./scripts/train.sh \
    --exp config/experiments/exp.yaml \
    --set epochs=150 --set batch=32 \
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
TASK="detect"
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
declare -a SET_OVERRIDES=()
declare -a IGNORED_EXP_KEYS=()

EXP_FILE=""

is_allowed_extra_key() {
  case "$1" in
    # Common Ultralytics train hyper-params
    box|cls|dfl|hsv_h|hsv_s|hsv_v|degrees|translate|scale|shear|perspective|flipud|fliplr|mosaic|mixup|copy_paste|erasing|multi_scale|cos_lr|max_det|verbose|profile|rect|close_mosaic|overlap_mask|dropout|mask_ratio|single_cls|freeze|deterministic|seed)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

remove_key_from_array() {
  local array_name="$1"
  local key="$2"
  local -n arr_ref="$array_name"
  local -a filtered=()
  local item=""
  for item in "${arr_ref[@]}"; do
    if [[ "${item%%=*}" == "$key" ]]; then
      continue
    fi
    filtered+=("$item")
  done
  arr_ref=("${filtered[@]}")
}

upsert_extra_arg() {
  local kv="$1"
  local key="${kv%%=*}"
  remove_key_from_array EXTRA_ARGS "$key"
  remove_key_from_array PROFILE_ARGS "$key"
  EXTRA_ARGS+=("$kv")
}

apply_key_value() {
  local key="$1"
  local value="$2"
  local source="$3"  # exp|set

  case "$key" in
    task) TASK="$value" ;;
    profile) PROFILE="$value" ;;
    workdir) WORKDIR="$value" ;;
    model) MODEL="$value" ;;
    data|dataset_yaml) DATA="$value" ;;
    epochs) EPOCHS="$value" ;;
    imgsz) IMGSZ="$value" ;;
    batch) BATCH="$value" ;;
    device) DEVICE="$value" ;;
    project) PROJECT="$value" ;;
    name|run_name) NAME="$value" ;;
    patience) PATIENCE="$value" ;;
    save_period|save-period) SAVE_PERIOD="$value" ;;
    workers) WORKERS="$value" ;;
    cache) CACHE="$value" ;;
    classes) CLASSES="$value" ;;
    optimizer) OPTIMIZER="$value" ;;
    lr0) LR0="$value" ;;
    lrf) LRF="$value" ;;
    momentum) MOMENTUM="$value" ;;
    weight_decay|weight-decay) WEIGHT_DECAY="$value" ;;
    warmup_epochs|warmup-epochs) WARMUP_EPOCHS="$value" ;;
    warmup_momentum|warmup-momentum) WARMUP_MOMENTUM="$value" ;;
    save) SAVE="$value" ;;
    val) VAL="$value" ;;
    plots) PLOTS="$value" ;;
    exist_ok|exist-ok) EXIST_OK="$value" ;;
    pretrained) PRETRAINED="$value" ;;
    amp) AMP="$value" ;;
    export_imgsz|export-imgsz) EXPORT_IMGSZ="$value" ;;
    export_opset|export-opset|opset) EXPORT_OPSET="$value" ;;
    no_export|no-export)
      if [[ "$value" == "True" || "$value" == "true" || "$value" == "1" ]]; then
        DO_EXPORT="false"
      fi
      ;;
    no_summary|no-summary)
      if [[ "$value" == "True" || "$value" == "true" || "$value" == "1" ]]; then
        DO_SUMMARY="false"
      fi
      ;;
    _extra)
      upsert_extra_arg "$value"
      ;;
    *)
      if [[ "$source" == "set" ]]; then
        upsert_extra_arg "${key}=${value}"
      elif is_allowed_extra_key "$key"; then
        upsert_extra_arg "${key}=${value}"
      else
        IGNORED_EXP_KEYS+=("${key}=${value}")
      fi
      ;;
  esac
}

load_experiment_entries() {
  local exp_path="$1"
  python3 - "$exp_path" <<'PYEOF'
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(f"PyYAML is required for --exp: {exc}") from exc

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Experiment file not found: {path}")

data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
if not isinstance(data, dict):
    raise SystemExit(f"Experiment YAML must be a mapping: {path}")

def to_scalar(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float, str)):
        return str(value)
    return None

def emit(key, value):
    scalar = to_scalar(value)
    if scalar is None:
        return
    print(f"{key}\t{scalar}")

def walk_mapping(node):
    for key, value in node.items():
        if key in {"train", "training", "experiment"} and isinstance(value, dict):
            walk_mapping(value)
            continue
        if key == "extra":
            if isinstance(value, dict):
                for k, v in value.items():
                    scalar = to_scalar(v)
                    if scalar is not None:
                        print(f"{k}\t{scalar}")
            elif isinstance(value, list):
                for item in value:
                    scalar = to_scalar(item)
                    if scalar and "=" in scalar:
                        print(f"_extra\t{scalar}")
            continue
        emit(key, value)

walk_mapping(data)
PYEOF
}

apply_experiment_file() {
  local exp_path="$1"
  while IFS=$'\t' read -r key value; do
    [[ -z "${key:-}" ]] && continue
    apply_key_value "$key" "$value" "exp"
  done < <(load_experiment_entries "$exp_path")
}

apply_set_overrides() {
  local item=""
  local key=""
  local value=""
  for item in "${SET_OVERRIDES[@]}"; do
    if [[ "$item" != *=* ]]; then
      echo "[ERROR] Invalid --set override (expected k=v): $item" >&2
      exit 2
    fi
    key="${item%%=*}"
    value="${item#*=}"
    if [[ -z "$key" ]]; then
      echo "[ERROR] Invalid --set override with empty key: $item" >&2
      exit 2
    fi
    apply_key_value "$key" "$value" "set"
  done
}

dedupe_profile_args_with_extra() {
  local item=""
  local key=""
  for item in "${EXTRA_ARGS[@]}"; do
    key="${item%%=*}"
    [[ -z "$key" ]] && continue
    remove_key_from_array PROFILE_ARGS "$key"
  done
}

# First pass: pick profile regardless of argument order.
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--profile" ]] && ((i + 1 < ${#ARGS[@]})); then
    PROFILE="${ARGS[$((i + 1))]}"
  fi
  if [[ "${ARGS[$i]}" == "--exp" ]] && ((i + 1 < ${#ARGS[@]})); then
    EXP_FILE="${ARGS[$((i + 1))]}"
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

if [[ -n "$EXP_FILE" ]]; then
  apply_experiment_file "$EXP_FILE"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --exp) EXP_FILE="$2"; shift 2 ;;
    --set) SET_OVERRIDES+=("$2"); shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
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

apply_set_overrides
apply_profile_defaults
dedupe_profile_args_with_extra

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
  yolo "$TASK" train
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
echo "[train.sh] Task:    $TASK"
echo "[train.sh] Profile: $PROFILE"
if [[ -n "$EXP_FILE" ]]; then
  echo "[train.sh] Exp:     $EXP_FILE"
fi
echo "[train.sh] Model:   $MODEL"
echo "[train.sh] Data:    $DATA"
echo "[train.sh] Run:     $PROJECT/$NAME"
echo "========================================="

if [[ ${#IGNORED_EXP_KEYS[@]} -gt 0 ]]; then
  echo "[train.sh] Ignored non-training keys from --exp:" >&2
  printf '  - %s\n' "${IGNORED_EXP_KEYS[@]}" >&2
fi

"${TRAIN_CMD[@]}"

BEST_PT="$PROJECT/$NAME/weights/best.pt"
if [[ ! -f "$BEST_PT" ]]; then
  echo "[ERROR] Training finished but best weights not found: $BEST_PT" >&2
  exit 1
fi

if [[ "$DO_EXPORT" == "true" ]]; then
  echo "[train.sh] Exporting ONNX (imgsz=$EXPORT_IMGSZ, opset=$EXPORT_OPSET)"
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

print("\n[train.sh] Summary")
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

echo "[train.sh] Done: $BEST_PT"
