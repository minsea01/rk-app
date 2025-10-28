Validate model accuracy and compare ONNX vs RKNN inference results with visualizations.

## Task

Execute comprehensive model validation:

1. Run ONNX inference on test images (baseline)
2. Run RKNN inference on same images (target)
3. Compare numerical outputs (MAE, max error, relative error)
4. Compare detection-level results (boxes, confidence, classes)
5. Generate visual comparisons (side-by-side images)
6. Calculate accuracy metrics (mAP if ground truth available)

## Parameters

Ask user (or use defaults):
- Model name: "best" (default)
- Test images: "assets/test.jpg" (default) or specify folder
- Ground truth: Optional path for mAP calculation
- Visualize: true (generate comparison images)

## Expected Actions

### 1. ONNX Inference
```bash
source ~/yolo_env/bin/activate
export PYTHONPATH=/home/minsea/rk-app
yolo predict model=artifacts/models/best.onnx source=assets/test.jpg save=true
```

### 2. RKNN Inference
```bash
python scripts/run_rknn_sim.py
```

### 3. Output Comparison
- Calculate tensor-level differences:
  - Mean Absolute Error (MAE)
  - Maximum Absolute Error
  - Relative Error (%)
- Compare detections:
  - Number of detections
  - Bounding box IoU
  - Confidence score differences
  - Class agreement

### 4. Visual Validation
Generate images:
- `{image}_onnx.jpg` - ONNX detections (green boxes)
- `{image}_rknn.jpg` - RKNN detections (blue boxes)
- `{image}_comparison.jpg` - Side-by-side with metrics

### 5. Accuracy Metrics (if ground truth provided)
- mAP@0.5 (graduation requirement: >90%)
- mAP@0.5:0.95
- Precision, Recall, F1

## Expected Results (Reference)

From CLAUDE.md:
- Mean absolute difference: ~0.01 (1%)
- Max relative error: <5%
- These indicate good INT8 quantization quality

## Output Files

- `artifacts/validation_report_{timestamp}.md` - Full report
- `artifacts/onnx_vs_rknn_comparison.json` - Numerical data
- `artifacts/visualizations/` - Comparison images

## Report Should Include

- Executive summary (conversion quality)
- Numerical comparison table
- Detection-level results table
- Visual validation images
- Accuracy metrics (if available)
- Conclusions and recommendations

## Success Criteria

- ONNX and RKNN produce consistent results
- Numerical differences <5% (acceptable for INT8)
- Detection agreement >90%
- Visual validation confirms correctness
- Optional: mAP meets >90% requirement

## Graduation Requirement Check

✅ Model size: 4.7MB (<5MB requirement)
⏸️ mAP@0.5: >90% (needs ground truth dataset)
