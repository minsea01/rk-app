Execute the complete model conversion pipeline from PyTorch/ONNX to RKNN with validation.

## Task

Run the full model conversion and validation workflow:

1. Check prerequisites (yolo_env, models, calibration dataset)
2. Export PyTorch model to ONNX (if .pt file exists and ONNX doesn't)
3. Convert ONNX to RKNN with INT8 quantization using calibration dataset
4. Run PC simulator inference validation
5. Generate comprehensive performance report

## Expected Actions

- Activate Python environment: `source ~/yolo_env/bin/activate`
- Export ONNX: `python tools/export_yolov8_to_onnx.py --weights {model}.pt --imgsz 640 --outdir artifacts/models`
- Convert to RKNN: `python tools/convert_onnx_to_rknn.py --onnx artifacts/models/{model}.onnx --out artifacts/models/{model}.rknn --calib datasets/coco/calib_images/calib.txt --target rk3588 --do-quant`
- Validate: `python scripts/run_rknn_sim.py`
- Generate report in `artifacts/pipeline_report.md`

## Success Criteria

- ONNX model created/exists
- RKNN model <5MB (graduation requirement)
- PC simulator inference completes successfully
- Report generated with all metrics

Ask user for model name if not specified (default: "best").
