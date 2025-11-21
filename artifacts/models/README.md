# Model Artifacts

This directory contains trained models and conversion artifacts.

## Directory Structure

```
artifacts/models/
├── yolo11n.pt              # PyTorch trained or baseline weights (optional, not tracked)
├── yolo11n.onnx            # Exported ONNX model for RKNN conversion (alias: best.onnx)
├── yolo11n_int8.rknn       # Quantized RKNN model for RK3588 NPU (alias: best.rknn)
├── yolo11n_fp16.rknn       # (Optional) FP16 RKNN fallback
├── yolo11n_416.rknn        # (Optional) 416×416 variant for low-latency profiles
└── README.md       # This file
```

## Export Pipeline

### 1. Export ONNX from PyTorch
```bash
yolo export model=runs/train/${RUN_NAME}/weights/best.pt \
     format=onnx imgsz=640 \
     opset=12 simplify \
     --device 0 \
     --project artifacts/models --name yolo11n
```

### 2. Convert ONNX to RKNN
```bash
python tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/yolo11n.onnx \
  --out artifacts/models/yolo11n_int8.rknn \
  --target rk3588 \
  --dtype asymmetric_quantized-u8
```

### 3. Compare Models (PC Simulation)
```bash
python tools/pc_compare.py \
  --onnx artifacts/models/best.onnx \
  --img ~/datasets/coco4cls_yolo/images/val/000000031296.jpg \
  --imgsz 640 \
  --outdir artifacts/logs/eq
```

## Current Status (2025-xx)

### Inference Benchmarks (PC, ONNX, imgsz=416)
- Command: `python tools/bench_onnx_latency.py --model artifacts/models/best.onnx --image assets/test.jpg --imgsz 416 --runs 30 --warmup 5 --providers cpu --out artifacts/bench_summary.json`
- Result: mean 18.83 ms (fps ≈ 53.1), p95 24.38 ms (fps_p95 ≈ 41.0)
- Notes: Model expects 416×416 input. Adjust `--imgsz` accordingly.

### Evaluation (YOLOv8 refresh pending)
- Latest available: YOLO11n on COCO person subset mAP@0.5 ≈ 0.616 (see `artifacts/yolo11n_official_full_map.json`).
- To refresh with YOLOv8: `yolo detect val model=artifacts/models/best.onnx data=<your_dataset.yaml> imgsz=416` and record mAP@0.5 / mAP@0.5:0.95 here.
