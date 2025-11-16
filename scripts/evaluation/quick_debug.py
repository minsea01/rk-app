#!/usr/bin/env python3
"""Quick debug script to diagnose mAP evaluation issues"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    # Load model
    import onnxruntime as ort
    model_path = "artifacts/models/yolo11n.onnx"

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Check model info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"\nModel Info:")
    print(f"  Input: {input_info.name}, shape={input_info.shape}, dtype={input_info.type}")
    print(f"  Output: {output_info.name}, shape={output_info.shape}, dtype={output_info.type}")

    # Load test image
    img_path = "datasets/coco/val2017/000000000139.jpg"
    print(f"\nLoading image: {img_path}")
    img = cv2.imread(img_path)
    print(f"  Original shape: {img.shape}")

    # Preprocess
    from apps.utils.yolo_post import letterbox
    img_resized, ratio, (dw, dh) = letterbox(img, 640)
    print(f"  Resized shape: {img_resized.shape}, ratio={ratio}, pad=({dw}, {dh})")

    # Convert to model input format
    img_input = img_resized[:, :, ::-1]  # BGR -> RGB
    img_input = img_input.transpose(2, 0, 1)  # HWC -> CHW
    img_input = img_input.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    print(f"  Input tensor shape: {img_input.shape}")

    # Run inference
    print("\nRunning inference...")
    output = session.run(None, {input_info.name: img_input})[0]
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Decode predictions
    print("\nDecoding predictions...")
    from apps.yolov8_rknn_infer import decode_predictions

    orig_shape = img.shape[:2]
    boxes, confs, class_ids = decode_predictions(
        output, 640, 0.25, 0.45,  # conf_thres, iou_thres
        ratio_pad=(ratio, (dw, dh)), orig_shape=orig_shape
    )

    print(f"  Total detections: {len(boxes)}")

    if len(boxes) > 0:
        print(f"\nAll detections:")
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, class_ids)):
            print(f"    {i}: class={int(cls_id)}, conf={conf:.4f}, bbox={box}")
            if i >= 10:
                print(f"    ... ({len(boxes) - 10} more)")
                break

        # Count person detections (class 0)
        person_mask = class_ids == 0
        person_count = np.sum(person_mask)
        print(f"\n  Person detections (class 0): {person_count}/{len(boxes)}")

        if person_count > 0:
            print(f"\n  Person detections only:")
            person_boxes = boxes[person_mask]
            person_confs = confs[person_mask]
            for i, (box, conf) in enumerate(zip(person_boxes, person_confs)):
                print(f"    {i}: conf={conf:.4f}, bbox={box}")
    else:
        print("  No detections found!")

    # Load ground truth
    print("\nLoading ground truth annotations...")
    ann_file = "datasets/coco/annotations/person_val2017.json"
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # Find annotations for this image
    img_id = 139  # From filename 000000000139.jpg
    gt_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

    print(f"  Ground truth boxes for image {img_id}: {len(gt_annotations)}")
    gt_boxes_xyxy = []
    for i, ann in enumerate(gt_annotations):
        x, y, w, h = ann['bbox']
        x2, y2 = x + w, y + h
        gt_boxes_xyxy.append([x, y, x2, y2])
        print(f"    {i}: bbox=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}] (COCO x,y,w,h)")
        print(f"        -> xyxy=[{x:.1f}, {y:.1f}, {x2:.1f}, {y2:.1f}]")

    # Calculate IoU between detections and ground truth
    if person_count > 0 and len(gt_boxes_xyxy) > 0:
        print(f"\n  IoU between person detections and GT:")
        for i, (pred_box, conf) in enumerate(zip(person_boxes, person_confs)):
            print(f"\n    Detection {i} (conf={conf:.4f}): {pred_box}")
            for j, gt_box in enumerate(gt_boxes_xyxy):
                # Calculate IoU
                x1_min, y1_min, x1_max, y1_max = pred_box
                x2_min, y2_min, x2_max, y2_max = gt_box

                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)

                if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
                    iou = 0.0
                else:
                    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0

                print(f"      vs GT {j}: IoU={iou:.4f} {'✓' if iou >= 0.5 else ''}")

if __name__ == '__main__':
    main()
