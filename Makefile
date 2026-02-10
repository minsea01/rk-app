# Simple orchestration for training, export, conversion, and evidence collection

# Defaults (override with env, e.g., make DATA_YAML=...)
EXP_YAML  ?= configs/experiments/exp.yaml
DATA_YAML ?= $(shell yq -r '.dataset_yaml' $(EXP_YAML) 2>/dev/null || echo $(HOME)/datasets/coco80_yolo/data.yaml)
RUN_NAME  ?= $(shell yq -r '.run_name' $(EXP_YAML) 2>/dev/null || echo coco80_y8s)
IMG_SIZE  ?= $(shell yq -r '.imgsz' $(EXP_YAML) 2>/dev/null || echo 640)
EPOCHS    ?= $(shell yq -r '.epochs' $(EXP_YAML) 2>/dev/null || echo 100)
BATCH     ?= $(shell yq -r '.batch' $(EXP_YAML) 2>/dev/null || echo auto)
DEVICE    ?= $(shell yq -r '.device' $(EXP_YAML) 2>/dev/null || echo 0)
WORKERS   ?= $(shell yq -r '.workers' $(EXP_YAML) 2>/dev/null || echo 8)
OPSET     ?= $(shell yq -r '.opset' $(EXP_YAML) 2>/dev/null || echo 12)
CALIB_DIR ?= $(shell yq -r '.calib_dir' $(EXP_YAML) 2>/dev/null || echo datasets/calib)

# Containers
TRAIN_IMG ?= yolov8-train:cu121
BUILD_IMG ?= rknn-build:1.7.5

# Model artifacts (prefix can be overridden)
MODEL_PREFIX ?= yolo11n
BEST_PT      ?= runs/train/$(RUN_NAME)/weights/best.pt
ONNX_OUT     ?= artifacts/models/$(MODEL_PREFIX).onnx
RKNN_INT8    ?= artifacts/models/$(MODEL_PREFIX)_int8.rknn
RKNN_FP16    ?= artifacts/models/$(MODEL_PREFIX)_fp16.rknn

.PHONY: train export convert-int8 convert-fp16 all collect validate

train:
	@echo "[TRAIN] data=$(DATA_YAML) run=$(RUN_NAME)" 
	docker run --rm -it --gpus all -v "$$PWD":/work -w /work \
	  -v $(HOME)/datasets:$(HOME)/datasets $(TRAIN_IMG) \
	  python tools/train_yolov8.py --data $(DATA_YAML) --model yolov8s.pt \
	  --imgsz $(IMG_SIZE) --epochs $(EPOCHS) --batch $(BATCH) --device $(DEVICE) \
	  --workers $(WORKERS) --project runs/train --name $(RUN_NAME)

export:
	@echo "[EXPORT] weights=$(BEST_PT) -> $(ONNX_OUT)"
	docker run --rm -it -v "$$PWD":/work -w /work $(BUILD_IMG) \
	  python tools/export_yolov8_to_onnx.py --weights $(BEST_PT) --imgsz $(IMG_SIZE) \
	  --opset $(OPSET) --simplify --outdir artifacts/models --outfile $$(basename $(ONNX_OUT))

convert-int8:
	@echo "[CONVERT INT8] onnx=$(ONNX_OUT) -> $(RKNN_INT8) calib=$(CALIB_DIR)"
	docker run --rm -it -v "$$PWD":/work -w /work $(BUILD_IMG) \
	  python tools/convert_onnx_to_rknn.py --onnx $(ONNX_OUT) --out $(RKNN_INT8) --calib $(CALIB_DIR)

convert-fp16:
	@echo "[CONVERT FP16] onnx=$(ONNX_OUT) -> $(RKNN_FP16)"
	docker run --rm -it -v "$$PWD":/work -w /work $(BUILD_IMG) \
	  python tools/convert_onnx_to_rknn.py --onnx $(ONNX_OUT) --out $(RKNN_FP16) --no-quant

all: train export convert-int8
	@echo "If INT8 failed, run: make convert-fp16"

collect:
	bash scripts/collect_evidence.sh $(RUN_NAME)

.PHONY: compare
COMPARE_IMG ?=
compare:
	@echo "[COMPARE] ONNX vs RKNN(sim)"
	python tools/pc_compare.py --onnx $(ONNX_OUT) $(if $(COMPARE_IMG),--img $(COMPARE_IMG),) --imgsz $(IMG_SIZE)

VALIDATE_IMG ?= assets/test.jpg
validate:
	@echo "[VALIDATE] onnx=$(ONNX_OUT) rknn=$(RKNN_INT8) img=$(VALIDATE_IMG)"
	python scripts/validate_models.py --onnx $(ONNX_OUT) --rknn $(RKNN_INT8) --image $(VALIDATE_IMG) --imgsz $(IMG_SIZE)

.PHONY: calib
CALIB_SRC ?=
CALIB_OUT ?= datasets/calib
CALIB_N ?= 300
calib:
	@if [ -z "$(CALIB_SRC)" ]; then echo "Usage: make calib CALIB_SRC=/path/to/data.yaml [CALIB_OUT=datasets/calib] [CALIB_N=300]"; exit 1; fi
	python tools/make_calib_set.py --data $(CALIB_SRC) --output $(CALIB_OUT) --num $(CALIB_N)

.PHONY: vis
IMG ?=
vis:
	@if [ -z "$(IMG)" ]; then echo "Usage: make vis IMG=/absolute/path/to/image.jpg"; exit 1; fi
	python tools/visualize_inference.py --onnx $(ONNX_OUT) --img $(IMG) --imgsz $(IMG_SIZE) --names data/coco80.names --out artifacts/vis/out.jpg
