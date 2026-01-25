# YOLOv8n Full Pipeline Report

**Generated:** 2026-01-05
**Model:** YOLOv8n (COCO pretrained)

## Pipeline Summary

| Step | Status | Details |
|------|--------|---------|
| Prerequisites | ✅ Pass | yolo_env, calibration dataset (300 images) |
| ONNX Export | ✅ Skip | Already exists (12.8MB) |
| RKNN Conversion | ✅ Pass | INT8 quantized, w8a8 |
| PC Simulator | ✅ Pass | Inference successful |

## Model Metrics

| Metric | Value | Requirement | Status |
|--------|-------|-------------|--------|
| PyTorch Size | 6.5 MB | - | - |
| ONNX Size | 12.8 MB | - | - |
| **RKNN Size** | **4.8 MB** | < 5 MB | ✅ |
| Input Size | 640×640 | - | - |
| Quantization | INT8 (w8a8) | - | - |

## PC Simulator Results

- **Latency:** 711.51 ms (PC simulation, not representative of board performance)
- **Output Shape:** (1, 84, 8400)
- **Output Dtype:** float32
- **Output Range:** [0.0, 638.0]
- **Output Mean:** 8.7360

## Conversion Details

- **RKNN Toolkit Version:** 2.3.2
- **Target Platform:** RK3588
- **Quantization Type:** w8a8 (INT8 weights, INT8 activations)
- **Calibration Dataset:** 300 images from COCO

### Quantization Warnings

Found outlier values in weights (may affect accuracy):
- `model.0.conv.weight`: outlier -17.494
- `model.22.cv3.2.1.conv.weight`: outlier -10.215
- `model.22.cv3.1.1.conv.weight`: outliers 13.361, 13.317
- `model.22.cv3.0.1.conv.weight`: outlier -11.216

## Files Generated

| File | Path | Size |
|------|------|------|
| PyTorch | `artifacts/models/yolov8n.pt` | 6.5 MB |
| ONNX | `artifacts/models/yolov8n.onnx` | 12.8 MB |
| RKNN (INT8) | `artifacts/models/yolov8n_int8.rknn` | 4.8 MB |

## Next Steps

1. **mAP Evaluation:** Run accuracy comparison between ONNX and RKNN
   ```bash
   python scripts/compare_onnx_rknn.py
   ```

2. **CityPersons Fine-tuning:** To achieve ≥90% mAP on pedestrian detection
   ```bash
   bash scripts/train/train_citypersons.sh
   ```

3. **Board Deployment:** When RK3588 hardware is available
   ```bash
   scripts/deploy/rk3588_run.sh --model artifacts/models/yolov8n_int8.rknn
   ```

## Conclusion

YOLOv8n model successfully converted through the full pipeline:
- ✅ Model size 4.8 MB meets <5 MB requirement
- ✅ INT8 quantization completed with w8a8
- ✅ PC simulator inference validated
- ⏸️ Board deployment pending hardware availability
