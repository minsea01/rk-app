#!/usr/bin/env python3
"""Test with official Ultralytics YOLO inference"""

from ultralytics import YOLO
import json

# Load YOLO model
model = YOLO('yolo11n.pt')

# Run inference
results = model.predict(
    source='datasets/coco/val2017/000000000139.jpg',
    imgsz=640,
    conf=0.25,
    iou=0.45,
    verbose=False
)

result = results[0]

print(f"Total detections: {len(result.boxes)}")

# Show person detections
person_boxes = []
for box in result.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].cpu().numpy()

    if cls_id == 0:  # person
        person_boxes.append({
            'bbox': xyxy.tolist(),
            'conf': conf
        })
        print(f"Person: conf={conf:.4f}, bbox={xyxy}")

print(f"\nTotal person detections: {len(person_boxes)}")

# Load GT
with open('datasets/coco/annotations/person_val2017.json', 'r') as f:
    coco_data = json.load(f)

gt_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == 139]
print(f"\nGround truth: {len(gt_annotations)} persons")
for ann in gt_annotations:
    x, y, w, h = ann['bbox']
    print(f"  GT: bbox=[{x:.1f}, {y:.1f}, {x+w:.1f}, {y+h:.1f}] (xyxy)")

# Calculate IoU
import numpy as np

def calc_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

if len(person_boxes) > 0:
    print("\nIoU with GT:")
    for i, pred in enumerate(person_boxes):
        print(f"\nPrediction {i} (conf={pred['conf']:.4f}):")
        for j, ann in enumerate(gt_annotations):
            x, y, w, h = ann['bbox']
            gt_box = [x, y, x+w, y+h]
            iou = calc_iou(pred['bbox'], gt_box)
            print(f"  vs GT {j}: IoU={iou:.4f} {'✓' if iou >= 0.5 else ''}")
