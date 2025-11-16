# mAP@0.5 Evaluation Report

**Model:** best.onnx
**Input Size:** 416×416
**Confidence Threshold:** 0.5

## Detection Statistics

| Metric | Value |
|--------|-------|
| Total Images | 300 |
| Images with Detections | 0 |
| Detection Rate | 0.0% |
| Total Detections | 0 |
| Avg Detections/Image | 0.00 |

## Notes

⚠️ Full mAP@0.5 requires ground truth annotations. This report shows detection statistics.

## Full mAP@0.5 Calculation

To compute mAP@0.5 on real pedestrian detection dataset:

1. **Prepare dataset with annotations:**
   - Download COCO pedestrian (person) subset
   - Or use custom pedestrian dataset with COCO format annotations

2. **Run mAP evaluation with pycocotools:**
   ```bash
   python3 scripts/evaluate_map.py \
     --onnx artifacts/models/best.onnx \
     --dataset path/to/coco/val2017 \
     --annotations path/to/instances_val2017.json
   ```

3. **Expected target:**
   - mAP@0.5: >90% (for pedestrian detection)

