#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import cv2
import numpy as np

from apps.utils.yolo_post import letterbox, postprocess_yolov8


def decode_predictions(pred, imgsz, conf_thres, iou_thres, head='auto', ratio_pad=(1.0, (0.0, 0.0)), orig_shape=None):
    # pred: (1, N, C) or (1, C, N) or (N, C)
    from apps.utils.yolo_post import sigmoid, nms
    # unify to (1, N, C)
    if pred.ndim == 2:
        pred = pred[None, ...]
    # Normalize to (1, N, C)
    if pred.shape[1] >= pred.shape[2]:
        pred_nc = pred
    else:
        pred_nc = pred.transpose(0, 2, 1)
    C = pred_nc.shape[2]
    if head == 'auto':
        head = 'dfl' if C >= 64 else 'raw'

    if head == 'dfl':
        boxes, confs, cls_ids = postprocess_yolov8(
            pred_nc, imgsz, orig_shape or (imgsz, imgsz), ratio_pad, conf_thres, iou_thres
        )
        return boxes, confs, cls_ids

    # raw path as heuristic decode: [cx,cy,w,h,obj,cls...]
    p = pred_nc[0]
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
    boxes = boxes[keep]; conf = conf[keep]; cls_ids = cls_ids[keep]

    # Scale back to original coordinates if provided
    r, (dw, dh) = ratio_pad
    if orig_shape is not None and r is not None:
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= r
        h0, w0 = orig_shape
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0 - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0 - 1)
    return boxes, conf, cls_ids


def load_labels(names_path: Path):
    if names_path and names_path.exists():
        return [x.strip() for x in names_path.read_text().splitlines() if x.strip()]
    return None


def draw_boxes(img, boxes, confs, class_ids, names=None):
    for (x1, y1, x2, y2), c, cls in zip(boxes.astype(int), confs, class_ids):
        label = f"{int(cls)}:{c:.2f}"
        if names and 0 <= int(cls) < len(names):
            label = f"{names[int(cls)]}:{c:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description='YOLOv8 RKNNLite inference on RK3588')
    ap.add_argument('--model', type=Path, required=True, help='path to .rknn model file')
    ap.add_argument('--source', type=Path, help='image path; if omitted, opens /dev/video0')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--names', type=Path, default=None, help='names.txt (one class per line)')
    ap.add_argument('--head', type=str, choices=['auto', 'dfl', 'raw'], default='auto', help='head decode type')
    ap.add_argument('--core-mask', type=lambda x: int(x, 0), default=0x7, help='NPU core mask (e.g., 0x7 for 3 cores)')
    ap.add_argument('--save', type=Path, default=None, help='save annotated result image')
    args = ap.parse_args()

    try:
        from rknnlite.api import RKNNLite
    except Exception as e:
        raise SystemExit(f"rknn-toolkit-lite2 not installed on device. pip install rknn-toolkit-lite2. Error: {e}")

    rknn = RKNNLite()
    print(f'Loading RKNN: {args.model}')
    ret = rknn.load_rknn(str(args.model))
    if ret != 0:
        raise SystemExit('load_rknn failed')

    print(f'Initializing runtime, core_mask={hex(args.core_mask)}')
    ret = rknn.init_runtime(core_mask=args.core_mask)
    if ret != 0:
        raise SystemExit('init_runtime failed')

    class_names = load_labels(args.names)

    if args.source and args.source.exists():
        img0 = cv2.imread(str(args.source))
        if img0 is None:
            raise SystemExit(f'Failed to read image: {args.source}')
        img, r, d = letterbox(img0, args.imgsz)
        # RKNN preproc set to BGR->RGB and /255 in conversion; feed BGR uint8
        input_data = img
        t0 = time.time()
        outputs = rknn.inference(inputs=[input_data])
        dt = (time.time() - t0) * 1000
        print(f'Inference time: {dt:.2f} ms')
        pred = outputs[0]
        if pred.ndim == 2:
            pred = pred[None, ...]
        boxes, confs, cls_ids = decode_predictions(pred, args.imgsz, args.conf, args.iou, args.head, (r, d), img0.shape[:2])
        print(f'Detections: {len(boxes)}')
        vis = draw_boxes(img0.copy(), boxes, confs, cls_ids, class_names)
        if args.save:
            cv2.imwrite(str(args.save), vis)
            print(f'Saved: {args.save}')
        else:
            cv2.imshow('result', vis)
            cv2.waitKey(0)
        rknn.release()
        return

    # Fallback to camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit('Failed to open camera (/dev/video0)')
    fps_hist = []
    try:
        while True:
            ret, img0 = cap.read()
            if not ret:
                break
            img, r, d = letterbox(img0, args.imgsz)
            t0 = time.time()
            outputs = rknn.inference(inputs=[img])
            pred = outputs[0]
            if pred.ndim == 2:
                pred = pred[None, ...]
            boxes, confs, cls_ids = decode_predictions(pred, args.imgsz, args.conf, args.iou, args.head, (r, d), img0.shape[:2])
            t1 = time.time()
            fps = 1.0 / max(1e-6, (t1 - t0))
            fps_hist.append(fps)
            vis = draw_boxes(img0.copy(), boxes, confs, cls_ids, class_names)
            cv2.putText(vis, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow('result', vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        rknn.release()
        cv2.destroyAllWindows()
        if fps_hist:
            print(f'Avg FPS: {np.mean(fps_hist):.2f}, P90: {np.percentile(fps_hist, 90):.2f}')


if __name__ == '__main__':
    main()

def decode_predictions(pred, args):
    # pred: (1, N, C) or (1, C, N) or (N, C)
    import numpy as np
    from apps.utils.yolo_post import postprocess_yolov8 as post_dfl
    # unify to (1, N, C)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if pred.shape[1] < pred.shape[2]:
        pred_nc = pred
    else:
        pred_nc = pred.transpose(0, 2, 1)
    N, C = pred_nc.shape[1], pred_nc.shape[2]
    head = args.head
    # Auto-detect: if C >= 64 assume DFL
    if head == 'auto':
        head = 'dfl' if C >= 64 else 'raw'
    if head == 'dfl':
        return post_dfl(pred_nc, args.imgsz, args.source and cv2.imread(str(args.source)).shape[:2] if args.source else (args.imgsz, args.imgsz), (1.0, (0.0, 0.0)), args.conf, args.iou)
    # raw path as heuristic decode: [cx,cy,w,h,obj,cls...]
    # Construct boxes/confs on the resized image scale; final scaling to original is done similarly to DFL path
    from apps.utils.yolo_post import sigmoid, nms
    p = pred_nc[0]
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
    m = conf >= args.conf
    if not np.any(m):
        return np.empty((0, 4)), np.array([]), np.array([])
    scale_needed = (np.percentile(w[m], 95) < 1.0) or (np.percentile(h[m], 95) < 1.0)
    s = float(args.imgsz) if scale_needed else 1.0
    cx *= s; cy *= s; w *= s; h *= s
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)[m]
    conf = conf[m]
    cls_ids = cls_ids[m]
    keep = nms(boxes, conf, iou_thres=args.iou)
    # Note: scaling back to original image is handled in DFL path; for raw we assume direct resized image scale
    return boxes[keep], conf[keep], cls_ids[keep]
