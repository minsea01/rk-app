# mAP Evaluation Root Cause Analysis & Solution

**Date:** 2025-11-16
**Status:** RESOLVED

## Problem Summary

Initial mAP measurement showed 0.4% mAP@0.5, which was 154x lower than expected baseline (60-85%).

## Root Cause

### Issue 1: Incorrect Postprocessing

**File:** apps/utils/yolo_post.py

**Problem:**
- Custom postprocess_yolov8() function designed for YOLOv8 DFL head
- YOLO11 has slightly different output format
- Caused bbox decoding errors:
  - Official YOLO: 1 person, conf=0.7155, IoU=0.7162
  - Our decoder: 8 persons, max conf=0.5743, only 1 IoU >= 0.5
  - Bbox offset: [396, 157, 456, 277] vs official [426, 156, 464, 297]

## Solution

**New Script:** scripts/evaluation/official_yolo_map.py

**Approach:**
1. Use ultralytics.YOLO.predict() for inference (official implementation)
2. Extract bbox directly from result.boxes.xyxy
3. Manual mAP calculation with precision-recall curve

## Results

### Official Ultralytics mAP (100 images):

- mAP@0.5: 61.93%
- Precision: 87.06%
- Recall: 64.20%
- TP: 269, FP: 40

### Previous Broken Method (100 images):

- mAP@0.5: 0.40%
- Precision: 3.63%
- Recall: 9.79%
- TP: 41, FP: 1090

**Improvement:** 154x better mAP!

## Next Steps

1. Complete full 2693-image baseline (running in background)
2. Fine-tune YOLO11n on pedestrian dataset to reach 90% requirement
3. Fix postprocess_yolov8() for ONNX/RKNN compatibility
