# Full Pipeline - 完整模型转换和验证流程

Execute the complete model conversion pipeline from PyTorch/ONNX to RKNN with validation.

## What this skill does

1. **Check prerequisites**: Verify yolo_env, model files, and calibration dataset
2. **Export to ONNX**: Convert PyTorch model to ONNX format (if .pt file exists)
3. **Convert to RKNN**: Apply INT8 quantization with calibration dataset
4. **PC Simulator Validation**: Run inference on PC simulator to verify functionality
5. **Performance Benchmark**: Generate performance metrics and comparison report

## Parameters

- `model_name` (optional): Model filename without extension (default: "best")
- `imgsz` (optional): Input image size, 416 or 640 (default: 640)
- `skip_export` (optional): Skip PyTorch to ONNX export if ONNX already exists

## Expected Output

- `artifacts/models/{model_name}.onnx` - ONNX model
- `artifacts/models/{model_name}.rknn` or `{model_name}_int8.rknn` - RKNN model
- `artifacts/pipeline_report.md` - Complete pipeline report with:
  - Model sizes
  - Conversion status
  - PC simulator inference results
  - Performance metrics

## Usage

Invoke this skill when you need to:
- Convert a new model for RK3588 deployment
- Validate model conversion pipeline
- Generate performance baseline for thesis documentation

## Steps

1. Activate Python environment and check dependencies
2. Locate or export ONNX model from PyTorch
3. Verify calibration dataset exists (datasets/coco/calib_images/calib.txt)
4. Run RKNN conversion with INT8 quantization
5. Execute PC simulator inference test
6. Generate comprehensive report with all metrics
7. Save artifacts to artifacts/models/

## Success Criteria

- ✅ ONNX model created/verified
- ✅ RKNN model size <5MB (graduation requirement)
- ✅ PC simulator inference completes without errors
- ✅ Output shape matches expected (1, 84, N)
- ✅ Report generated with all performance metrics
