# Model Validate - 模型精度验证和对比

Validate model accuracy and compare ONNX vs RKNN inference results.

## What this skill does

1. **ONNX Inference**: Run inference on test images with ONNX model
2. **RKNN Inference**: Run same images through RKNN model
3. **Output Comparison**: Calculate numerical differences between outputs
4. **Visual Validation**: Generate side-by-side detection visualizations
5. **Accuracy Metrics**: Compute mAP, precision, recall (if ground truth available)
6. **Generate Validation Report**: Document accuracy and conversion quality

## Parameters

- `model_name` (optional): Model to validate (default: "best")
- `test_images` (optional): Path to test images or folder (default: "assets/test.jpg")
- `ground_truth` (optional): Path to ground truth annotations for mAP calculation
- `visualize` (optional): Generate visualization images (default: true)

## Expected Output

- `artifacts/validation_report_{timestamp}.md` - Validation report
- `artifacts/onnx_vs_rknn_comparison.json` - Numerical comparison metrics
- `artifacts/visualizations/` - Side-by-side detection images (if enabled)
  - `{image_name}_onnx.jpg` - ONNX detections
  - `{image_name}_rknn.jpg` - RKNN detections
  - `{image_name}_comparison.jpg` - Side-by-side

## Usage

Invoke this skill when:
- Validating INT8 quantization accuracy
- Verifying ONNX→RKNN conversion correctness
- Preparing accuracy metrics for thesis
- Debugging detection quality issues
- Creating visual results for defense presentation

## Validation Process

### 1. ONNX Inference (Baseline)

**Setup:**
- Load ONNX model with onnxruntime-gpu
- Use CUDAExecutionProvider for speed
- Apply same preprocessing as RKNN

**Inference:**
- Run on all test images
- Extract bounding boxes, confidence scores, class IDs
- Save raw outputs for comparison

### 2. RKNN Inference (Target)

**Setup:**
- Load RKNN model in PC simulator mode
- Use NHWC input format
- Keep uint8 inputs (no normalization)

**Inference:**
- Run on same test images
- Extract detections using same postprocessing
- Save raw outputs

### 3. Numerical Comparison

**Output Tensor Comparison:**
- Shape matching: Verify (1, 84, N) format
- Raw output differences:
  - Mean Absolute Error (MAE)
  - Maximum Absolute Error
  - Relative Error (%)
- Distribution analysis

**Detection-level Comparison:**
- Number of detections: ONNX vs RKNN
- Bounding box differences (IoU-based matching)
- Confidence score differences
- Class prediction agreements

**Expected Results (from CLAUDE.md):**
- Mean absolute difference: ~0.01 (1%)
- Max relative error: <5%
- These are reference values

### 4. Visual Validation

Generate visualizations:
1. **ONNX detections**: Green boxes
2. **RKNN detections**: Blue boxes
3. **Comparison view**: Side-by-side with metrics overlay

Metrics displayed:
- Detection counts
- Average confidence
- IoU agreement
- Processing time

### 5. Accuracy Metrics (if ground truth provided)

**Metrics calculated:**
- **mAP@0.5**: Main thesis requirement (target: >90%)
- **mAP@0.5:0.95**: COCO standard metric
- Precision, Recall per class
- F1 score

**Dataset options:**
- COCO validation set (person class)
- Custom pedestrian dataset
- Industrial scene test set

## Validation Report Structure

```markdown
# Model Validation Report
Date: YYYY-MM-DD
Model: best.onnx / best.rknn
Test Images: 10 images

## Executive Summary
- ✅ ONNX→RKNN conversion successful
- ✅ Output numerical difference: MAE=0.008 (<1%)
- ✅ Detection agreement: 95% IoU>0.7
- ⏸️ mAP validation: Pending ground truth

## Numerical Comparison

### Raw Output Tensor
- ONNX shape: (1, 84, 3549)
- RKNN shape: (1, 84, 3549) ✅ Match
- Mean Absolute Error: 0.0082 (0.82%)
- Max Absolute Error: 0.156 (15.6% at low values)
- Relative Error: 1.2% (acceptable for INT8)

### Detection-level Results

Test Image: test.jpg
| Metric | ONNX | RKNN | Difference |
|--------|------|------|------------|
| Detections | 1 person | 1 person | ✅ Match |
| Avg Confidence | 0.87 | 0.85 | -0.02 |
| Box IoU | - | - | 0.96 |

## Visual Validation

### Image 1: test.jpg
- ONNX: 1 person detected (conf=0.87)
- RKNN: 1 person detected (conf=0.85)
- Agreement: ✅ Same detection
- Visualization: artifacts/visualizations/test_comparison.jpg

[Include visualization images]

## Accuracy Metrics (if available)

### mAP Results
- mAP@0.5: XX.X% (Requirement: >90%)
- mAP@0.5:0.95: XX.X%
- Precision: XX.X%
- Recall: XX.X%

Dataset: [COCO val / Custom dataset]
Classes evaluated: [person / all]

## Conclusions

1. **Conversion Quality**: ✅ Excellent
   - Numerical differences <1% (INT8 quantization acceptable)
   - Detection agreement >95%

2. **Inference Correctness**: ✅ Verified
   - Both models detect same objects
   - Confidence scores within acceptable range
   - Bounding boxes closely aligned

3. **Thesis Compliance**: [Status]
   - Model size: 4.7MB ✅ (<5MB)
   - Accuracy: [If measured] vs >90% requirement

4. **Recommendations**:
   - INT8 quantization maintains accuracy
   - Ready for board deployment
   - Consider testing on full validation dataset

## Next Steps
- [ ] Run on larger test set for statistical significance
- [ ] Compute mAP with ground truth annotations
- [ ] Test on industrial/pedestrian-specific dataset
- [ ] Validate on RK3588 board (when available)
```

## Success Criteria

- ✅ ONNX and RKNN produce consistent results
- ✅ Numerical differences within expected range (<5%)
- ✅ Detection agreement >90%
- ✅ Visual validation confirms correctness
- ✅ (Optional) mAP meets >90% requirement

## Tools Used

- `scripts/compare_onnx_rknn.py` - Automated comparison script
- `apps/yolov8_rknn_infer.py` - RKNN inference
- `ultralytics` - ONNX inference and mAP calculation
- `opencv` - Visualization

## Notes

- INT8 quantization inherently introduces small errors (~1%)
- Errors <5% are acceptable for embedded deployment
- mAP calculation requires properly formatted ground truth
- Visual inspection is important for qualitative validation
- Some detection differences are due to NMS randomness
