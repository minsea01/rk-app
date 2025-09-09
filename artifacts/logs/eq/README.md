# Comparison Logs

This directory contains equivalence test results between ONNX and RKNN models.

## Log Files

- `compare_YYYYMMDD-HHMMSS.json` - Detailed metrics in JSON format
- `compare_YYYYMMDD-HHMMSS.txt` - Human-readable summary

## Metrics Explained

### Numerical Accuracy
- **MSE**: Mean Squared Error between ONNX and RKNN outputs
- **MAE**: Mean Absolute Error 
- **Max Abs**: Maximum absolute difference

### Post-Processing Accuracy  
- **Post-NMS IoU**: Average IoU between top-100 detection boxes
- **Mean |Δconf|**: Average confidence score difference

### Performance
- **ONNX Infer**: ONNXRuntime inference time (ms)
- **RKNN Build**: RKNN model compilation time (ms)
- **RKNN Init**: RKNN runtime initialization (ms)
- **RKNN Infer**: RKNN simulator inference time (ms)

## Acceptance Criteria

**Good INT8 Quantization**:
- MSE < 0.001
- Post-NMS IoU > 0.95
- Mean |Δconf| < 0.05

**Performance Target**:
- RKNN inference < 42ms for 640×640 input on RK3588