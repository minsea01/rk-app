#!/usr/bin/env python3
# Evaluate mAP (YOLO-style) from detect_cli JSONL outputs against YOLO txt labels
# - Predictions: out/result.jsonl (produced by detect_cli via --json)
# - Ground truth: YOLO txt files under labels/* with normalized cx,cy,w,h
# - Images folder: taken from JSONL 'source_uri' (must be a folder source)

import argparse
import json
import os
import glob
from collections import defaultdict
from pathlib import Path

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0

def compute_ap(rec, prec):
    # VOC-style integral: numerical integration over precision envelope
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    # Make precision non-increasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Integrate AP
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap

def load_class_names(path):
    names = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names

def list_images(folder):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = []
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
    return files

def yolo_txt_to_xyxy(txt_path, width, height):
    gts = []
    if not os.path.exists(txt_path):
        return gts
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5 and len(parts) != 6:
                # allow optional conf in label
                continue
            cls = int(parts[0])
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            w  = float(parts[3]) * width
            h  = float(parts[4]) * height
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            gts.append((cls, [x1, y1, x2, y2]))
    return gts

def main():
    ap = argparse.ArgumentParser(description='Evaluate mAP from detect_cli JSONL vs YOLO labels')
    ap.add_argument('--pred', required=True, help='result.jsonl from detect_cli')
    ap.add_argument('--names', required=True, help='classes.txt path')
    ap.add_argument('--labels-root', default=None, help='labels root (default: infer from images path)')
    ap.add_argument('--img-root', default=None, help='images root (default: take from JSON)')
    ap.add_argument('--iou', default='0.50:0.95:0.05', help='IoU thresholds as start:end:step or single, e.g., 0.5 or 0.50:0.95:0.05')
    args = ap.parse_args()

    names = load_class_names(args.names)
    nc = len(names)

    # Read predictions
    preds_by_frame = {}
    src_dirs = set()
    with open(args.pred, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = obj.get('frame_id')
            w = obj.get('width')
            h = obj.get('height')
            src = obj.get('source_uri', '')
            src_dirs.add(src)
            dets = []
            for d in obj.get('detections', []):
                x1 = float(d['x'])
                y1 = float(d['y'])
                x2 = x1 + float(d['w'])
                y2 = y1 + float(d['h'])
                conf = float(d.get('confidence', 1.0))
                cid = int(d.get('class_id', -1))
                if cid < 0:
                    continue
                dets.append((cid, conf, [x1, y1, x2, y2]))
            preds_by_frame[fid] = {'wh': (w, h), 'src': src, 'dets': dets}

    if args.img_root:
        img_root = args.img_root
    else:
        if len(src_dirs) != 1:
            raise RuntimeError(f'Cannot infer a unique images folder from JSON (got {src_dirs}). Use --img-root')
        img_root = list(src_dirs)[0]

    # Build sorted image list to map frame_id -> image path
    images = list_images(img_root)
    if not images:
        raise RuntimeError(f'No images found under {img_root}')

    # Derive labels root if not provided
    if args.labels_root:
        labels_root = args.labels_root
    else:
        # replace /images/ with /labels/
        if '/images/' in img_root:
            labels_root = img_root.replace('/images/', '/labels/')
        else:
            labels_root = img_root.replace('/images', '/labels')

    # Prepare GT and prediction pools per class
    gts_per_cls = defaultdict(list)    # cls -> list of (img_id, bbox, matched=False)
    preds_per_cls = defaultdict(list)  # cls -> list of (img_id, conf, bbox)

    for fid, entry in preds_by_frame.items():
        if fid is None:
            continue
        if fid < 0 or fid >= len(images):
            continue
        img_path = images[fid]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        w, h = entry['wh']
        # Load GTs
        # Construct YOLO label path: labels_root/<subdirs_of_img_root...>/<img_name>.txt
        # Preserve substructure after the first occurrence of img_root's suffix path
        # Simplify: assume parallel structure labels/* mirrors images/*
        rel = os.path.relpath(img_path, img_root)
        label_path = os.path.join(labels_root, os.path.splitext(rel)[0] + '.txt')
        gts = yolo_txt_to_xyxy(label_path, w, h)
        for cls, bbox in gts:
            gts_per_cls[cls].append([fid, bbox, False])
        # Predictions
        for cls, conf, bbox in entry['dets']:
            preds_per_cls[cls].append((fid, conf, bbox))

    # IoU thresholds
    iou_list = []
    if ':' in args.iou:
        s, e, st = args.iou.split(':')
        a = float(s); b = float(e); step = float(st)
        t = a
        while t <= b + 1e-9:
            iou_list.append(round(t, 2))
            t += step
    else:
        iou_list = [float(args.iou)]

    # Compute AP per class and per IoU
    aps_50_95 = []
    ap50s = []
    per_class_output = []
    for cls in range(nc):
        aps_th = []
        for thr in iou_list:
            preds = sorted(preds_per_cls.get(cls, []), key=lambda x: -x[1])
            gts_cls = gts_per_cls.get(cls, [])
            npos = len(gts_cls)
            tp = []
            fp = []
            # Reset matched flags per threshold
            gmatched = [False] * len(gts_cls)
            for (img_id, conf, pb) in preds:
                # find best match in GTs of same image
                best_iou = 0.0
                best_j = -1
                for j, (gt_img, gb, m) in enumerate(gts_cls):
                    if gmatched[j] or gt_img != img_id:
                        continue
                    iou = iou_xyxy(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= thr and best_j >= 0:
                    gmatched[best_j] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            if npos == 0:
                aps_th.append(0.0)
                continue
            # Precision-Recall
            import itertools
            cum_tp = list(itertools.accumulate(tp))
            cum_fp = list(itertools.accumulate(fp))
            rec = [ct / npos for ct in cum_tp]
            prec = [ (ct / (ct + cf)) if (ct + cf) > 0 else 0.0 for ct, cf in zip(cum_tp, cum_fp) ]
            ap = compute_ap(rec, prec)
            aps_th.append(ap)
        # save outputs
        if aps_th:
            ap50 = aps_th[0] if abs(iou_list[0] - 0.5) < 1e-3 else None
            ap_mean = sum(aps_th) / len(aps_th)
            per_class_output.append((cls, names[cls] if cls < len(names) else str(cls), ap50, ap_mean))
            if ap50 is not None:
                ap50s.append(ap50)
            aps_50_95.append(ap_mean)

    # Print summary
    print('Classes:', len(names))
    print('IoU thresholds:', iou_list)
    print('Per-class AP:')
    for cls, cname, ap50, apm in per_class_output:
        if ap50 is not None:
            print(f'  {cls:2d} {cname:20s}  AP50={ap50*100:.2f}  AP@[.50:.95]={apm*100:.2f}')
        else:
            print(f'  {cls:2d} {cname:20s}  AP={apm*100:.2f}')
    if ap50s:
        print(f'Overall: mAP@0.50={sum(ap50s)/len(ap50s)*100:.2f}%')
    if aps_50_95:
        print(f'Overall: mAP@0.50:0.95={sum(aps_50_95)/len(aps_50_95)*100:.2f}%')

if __name__ == '__main__':
    main()

