#!/usr/bin/env python3
"""Shared prediction decoding for RKNN inference modules."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np

from apps.utils.decode_meta import (
    normalize_decode_meta,
    resolve_dfl_layout,
    resolve_head,
    resolve_raw_layout,
)
from apps.utils.yolo_post import nms, postprocess_yolov8, sigmoid


def _empty_decode_result() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 4), dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.int64),
    )


def decode_predictions(
    pred: np.ndarray,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    head: str = "auto",
    ratio_pad: Tuple[float, Tuple[float, float]] = (1.0, (0.0, 0.0)),
    orig_shape: Optional[Tuple[int, int]] = None,
    decode_meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode model predictions to boxes/scores/class ids.

    Modes:
      - `orig_shape is None`: return boxes in resized-input coordinates.
      - `orig_shape is not None`: map boxes back to original image coordinates.
    """
    # Normalize input to (1, N, C)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if pred.shape[1] >= pred.shape[2]:
        pred_nc = pred
    else:
        pred_nc = pred.transpose(0, 2, 1)

    c = pred_nc.shape[2]
    decode_meta = normalize_decode_meta(decode_meta)
    resolved_head = resolve_head(head, c, decode_meta)
    if resolved_head is None:
        return _empty_decode_result()

    if resolved_head == "dfl":
        dfl_layout = resolve_dfl_layout(c, decode_meta)
        if dfl_layout is None:
            return _empty_decode_result()
        reg_max, strides = dfl_layout

        # Stream mode keeps resize-space coordinates; infer mode maps to original space.
        if orig_shape is None:
            decode_orig_shape = (imgsz, imgsz)
            decode_ratio_pad = (1.0, (0.0, 0.0))
        else:
            decode_orig_shape = orig_shape
            decode_ratio_pad = ratio_pad

        try:
            boxes, confs, cls_ids = postprocess_yolov8(
                pred_nc,
                imgsz,
                decode_orig_shape,
                decode_ratio_pad,
                conf_thres,
                iou_thres,
                reg_max=reg_max,
                strides=tuple(strides),
            )
        except ValueError:
            return _empty_decode_result()
        return boxes, confs, cls_ids

    raw_layout = resolve_raw_layout(c, decode_meta)
    if raw_layout is None:
        return _empty_decode_result()
    has_objness, num_classes = raw_layout

    # raw path: [cx, cy, w, h, obj, cls...]
    p = pred_nc[0]
    if c < 5:
        return _empty_decode_result()

    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    obj = sigmoid(p[:, 4]) if has_objness else np.ones_like(cx, dtype=np.float32)
    cls_offset = 5 if has_objness else 4
    if num_classes > 0:
        cls_scores = sigmoid(p[:, cls_offset:cls_offset + num_classes])
        cls_ids = cls_scores.argmax(axis=1)
        cls_conf = cls_scores.max(axis=1)
    else:
        cls_ids = np.zeros(cx.shape[0], dtype=np.int64)
        cls_conf = np.ones_like(obj)

    conf = obj * cls_conf
    mask = conf >= conf_thres
    if not np.any(mask):
        return _empty_decode_result()

    # Heuristic:
    # If width/height 95th percentile is < 1.0, outputs are likely normalized to [0,1].
    # In that case scale by imgsz before converting to xyxy.
    scale_needed = (np.percentile(w[mask], 95) < 1.0) or (np.percentile(h[mask], 95) < 1.0)
    scale = float(imgsz) if scale_needed else 1.0
    cx *= scale
    cy *= scale
    w *= scale
    h *= scale

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)[mask]
    conf = conf[mask]
    cls_ids = cls_ids[mask]

    keep = nms(boxes, conf, iou_thres=iou_thres)
    boxes = boxes[keep]
    conf = conf[keep]
    cls_ids = cls_ids[keep]

    if orig_shape is not None:
        r, (dw, dh) = ratio_pad
        if r is not None:
            boxes[:, [0, 2]] -= dw
            boxes[:, [1, 3]] -= dh
            boxes /= r
            h0, w0 = orig_shape
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0 - 1)
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0 - 1)

    return boxes, conf, cls_ids

