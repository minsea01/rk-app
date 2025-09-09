# Model Artifacts

This directory contains trained models and conversion artifacts.

## Directory Structure

```
artifacts/models/
├── best.pt         # PyTorch trained weights
├── best.onnx       # Exported ONNX model for inference  
├── best.rknn       # RKNN quantized model for RK3588 NPU
└── README.md       # This file
```

## Export Pipeline

### 1. Export ONNX from PyTorch
```bash
yolo export model=runs/train/your10c_y8s/weights/best.pt \
     format=onnx imgsz=640 \
     opset=12 simplify \
     --device 0 \
     --project artifacts/models --name best
```

### 2. Convert ONNX to RKNN
```bash
python tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/best.onnx \
  --out artifacts/models/best.rknn \
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