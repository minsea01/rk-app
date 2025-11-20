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
├── yolo11n.rknn.meta.json  # Model metadata (reg_max, strides, head_type)
└── README.md               # This file
```

## Resolution Selection Guide

### 640×640 (Standard)
**Use when:**
- Maximum accuracy is priority
- Running on PC with ONNX GPU inference (8.6ms @ RTX 3060)
- Evaluating mAP on validation datasets

**Performance:**
- ONNX GPU: 8.6ms @ RTX 3060
- RKNN NPU: 20-40ms (INT8 quantized)
- **⚠️ CAUTION**: Output shape (1, 84, 8400) with 4×8400=33,600 elements **exceeds RK3588 Transpose NPU limit (16,384)**
  - Results in **CPU fallback** for Transpose operation
  - Significantly impacts latency on board

### 416×416 (Optimized) ✅ **RECOMMENDED for Production**
**Use when:**
- Real-time performance is critical (>30 FPS target)
- Deploying on RK3588 NPU
- Minimizing latency and power consumption

**Performance:**
- ONNX GPU: 8.6ms @ RTX 3060 (416×416)
- RKNN NPU: 25-35ms estimated (INT8 quantized)
- **✅ ADVANTAGE**: Output shape (1, 84, 3549) with 4×3549=14,196 elements **fits within NPU Transpose limit**
  - Full NPU execution without CPU fallback
  - Optimal latency and power efficiency

### Critical Parameter Tuning

**Confidence Threshold Impact:**
- ❌ `conf=0.25` (default): 3135ms postprocessing → 0.3 FPS (NMS bottleneck)
- ✅ `conf=0.5` (optimized): 5.2ms postprocessing → 60+ FPS (production ready)

**Recommendation**: Use `conf≥0.5` for industrial applications to avoid excessive false positives and NMS overhead.

## Export Pipeline

### 1. Export ONNX from PyTorch

**Standard 640×640 model:**
```bash
yolo export model=runs/train/${RUN_NAME}/weights/best.pt \
     format=onnx imgsz=640 \
     opset=12 simplify \
     --device 0 \
     --project artifacts/models --name yolo11n
```

**Optimized 416×416 model (RECOMMENDED for RK3588 NPU):**
```bash
yolo export model=runs/train/${RUN_NAME}/weights/best.pt \
     format=onnx imgsz=416 \
     opset=12 simplify \
     --device 0 \
     --project artifacts/models --name yolo11n_416
```

### 2. Convert ONNX to RKNN

**Convert 640×640 model:**
```bash
python tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/yolo11n.onnx \
  --out artifacts/models/yolo11n_int8.rknn \
  --calib datasets/coco/calib_images/calib.txt \
  --target rk3588 \
  --do-quant
```

**Convert 416×416 model (RECOMMENDED for production):**
```bash
python tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/yolo11n_416.onnx \
  --out artifacts/models/yolo11n_416_int8.rknn \
  --calib datasets/coco/calib_images/calib.txt \
  --target rk3588 \
  --do-quant
```

**Note:** Use absolute paths in `calib.txt` to avoid path duplication errors:
```bash
find datasets/coco/calib_images -name "*.jpg" -exec realpath {} \; > datasets/coco/calib_images/calib.txt
```

### 3. Compare Models (PC Simulation)
```bash
python tools/pc_compare.py \
  --onnx artifacts/models/best.onnx \
  --img ~/datasets/coco4cls_yolo/images/val/000000031296.jpg \
  --imgsz 640 \
  --outdir artifacts/logs/eq
```
