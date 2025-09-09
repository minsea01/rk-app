#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import numpy as np
import cv2

# Ensure repo root is importable when running this script directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from apps.utils.yolo_post import letterbox, postprocess_yolov8, sigmoid, nms


def unify_pred(pred):
    # Normalize to (1, N, C) where N=anchors, C=attrs
    if pred.ndim == 2:
        pred = pred[None, ...]
    # If second dim (N) is larger/equal than third (C), already (1,N,C)
    if pred.shape[1] >= pred.shape[2]:
        return pred
    # Else it's (1,C,N), transpose
    return pred.transpose(0, 2, 1)


def decode_raw(pred_nc, imgsz, conf_thres=0.25, iou_thres=0.45):
    # pred_nc: (1, N, C) where C>=5 -> [cx,cy,w,h,obj,cls...]
    p = pred_nc[0]
    C = p.shape[1]
    if C < 5:
        return np.empty((0, 4)), np.array([]), np.array([])
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    obj = sigmoid(p[:, 4])
    if C > 5:
        cls_scores = sigmoid(p[:, 5:])
        cls_ids = cls_scores.argmax(axis=1)
        cls_conf = cls_scores.max(axis=1)
    else:
        cls_ids = np.zeros_like(obj, dtype=np.int64)
        cls_conf = np.ones_like(obj)
    conf = obj * cls_conf
    m = conf >= conf_thres
    if not np.any(m):
        return np.empty((0, 4)), np.array([]), np.array([])
    scale_needed = (np.percentile(w[m], 95) < 1.0) or (np.percentile(h[m], 95) < 1.0)
    s = float(imgsz) if scale_needed else 1.0
    cx *= s; cy *= s; w *= s; h *= s
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)[m]
    conf = conf[m]
    cls_ids = cls_ids[m]
    keep = nms(boxes, conf, iou_thres=iou_thres)
    return boxes[keep], conf[keep], cls_ids[keep]


def annotate(img, boxes, confs, cls_ids, names=None):
    for (x1, y1, x2, y2), c, ci in zip(boxes.astype(int), confs, cls_ids):
        label = f"{int(ci)}:{c:.2f}"
        if names and 0 <= int(ci) < len(names):
            label = f"{names[int(ci)]}:{c:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description='Visualize ONNX inference and save annotated image')
    ap.add_argument('--onnx', type=Path, required=True, help='path to ONNX')
    ap.add_argument('--img', type=Path, required=False, default=Path(''), help='input image path (optional)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--names', type=Path, default=None, help='names file (optional)')
    ap.add_argument('--out', type=Path, default=Path('artifacts/vis/out.jpg'))
    args = ap.parse_args()

    import onnxruntime as ort

    names = None
    if args.names and args.names.exists():
        names = [x.strip() for x in args.names.read_text().splitlines() if x.strip()]

    img0 = None
    if args.img and args.img.exists():
        img0 = cv2.imread(str(args.img))
    if img0 is None:
        # build a synthetic image if not provided
        sz = args.imgsz
        img0 = (np.random.RandomState(0).rand(sz, sz, 3) * 255).astype(np.uint8)
        cv2.putText(img0, 'SYNTH', (10, sz//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 3)
    img, r, d = letterbox(img0, args.imgsz)

    sess = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
    y = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0})[0]
    pred = unify_pred(y)
    C = pred.shape[2]

    if C >= 64:
        boxes, confs, cls_ids = postprocess_yolov8(pred, args.imgsz, img0.shape[:2], (r, d), args.conf, args.iou)
    else:
        boxes, confs, cls_ids = decode_raw(pred, args.imgsz, args.conf, args.iou)
        # scale back to original
        boxes[:, [0, 2]] -= d[0]
        boxes[:, [1, 3]] -= d[1]
        boxes /= r
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, img0.shape[1] - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, img0.shape[0] - 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vis = annotate(img0.copy(), boxes, confs, cls_ids, names)
    cv2.imwrite(str(args.out), vis)
    print(f'Saved visualization: {args.out}')


if __name__ == '__main__':
    main()
