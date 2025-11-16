import math
from typing import Tuple, List, Union
import numpy as np
import cv2


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Numerically stable sigmoid function.

    Avoids overflow for large negative values and division by near-zero for large positive values.
    Uses the identity: sigmoid(x) = exp(x) / (1 + exp(x)) for x < 0

    Args:
        x: Input array

    Returns:
        Sigmoid activation: 1 / (1 + exp(-x))
    """
    # For x >= 0: use standard formula to avoid exp overflow
    # For x < 0: use exp(x) / (1 + exp(x)) to avoid division issues
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def letterbox(
    im: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints.

    Args:
        im: Input image array (H, W, C)
        new_shape: Target shape (single int for square or (H, W) tuple)
        color: Padding color (R, G, B)

    Returns:
        Tuple of (resized_image, ratio, (pad_w, pad_h))
    """
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def make_anchors(strides: List[int], img_size: int) -> np.ndarray:
    anchors = []
    for s in strides:
        fm_w = img_size // s
        fm_h = img_size // s
        grid_y, grid_x = np.meshgrid(np.arange(fm_h), np.arange(fm_w), indexing='ij')
        cx = (grid_x.reshape(-1, 1) + 0.5) * s
        cy = (grid_y.reshape(-1, 1) + 0.5) * s
        anchors.append(np.hstack([cx, cy]))
    return np.vstack(anchors).astype(np.float32)  # (N, 2)


def dfl_decode(d: np.ndarray, reg_max: int = 16) -> np.ndarray:
    # d: (N, 4*reg_max) -> (N, 4)
    d = d.reshape(-1, 4, reg_max)
    # softmax along reg_max
    d = np.exp(d - d.max(axis=2, keepdims=True))
    d = d / d.sum(axis=2, keepdims=True)
    project = np.arange(reg_max, dtype=np.float32)
    out = (d * project).sum(axis=2)
    return out  # l, t, r, b in grid units


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45, topk: int = 300) -> List[int]:
    """Non-Maximum Suppression for object detection.

    Args:
        boxes: Bounding boxes in xyxy format, shape (M, 4)
        scores: Confidence scores, shape (M,)
        iou_thres: IoU threshold for suppression
        topk: Maximum number of boxes to keep before NMS

    Returns:
        List of indices to keep

    Note:
        Uses floating-point coordinate system (modern YOLO).
        Removed legacy +1 correction for integer coordinates.
    """
    # boxes: (M, 4) xyxy, scores: (M,)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # Fixed: Removed +1 for floating-point coordinates
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    if topk is not None:
        order = order[:topk]
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # Fixed: Removed +1 for floating-point intersection area
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

# Expose sigmoid for reuse by app
__all__ = ['letterbox', 'postprocess_yolov8', 'nms', 'sigmoid']


def postprocess_yolov8(
    preds: np.ndarray,  # expected (1, N, 64+nc)
    img_size: int,
    orig_shape: Tuple[int, int],
    ratio_pad: Tuple[float, Tuple[float, float]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    reg_max: int = 16,
    strides: List[int] = (8, 16, 32),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-process YOLOv8 model outputs.

    Args:
        preds: Model predictions, shape (1, N, 64+nc)
        img_size: Input image size used for inference
        orig_shape: Original image shape (H, W) before letterbox
        ratio_pad: Tuple of (ratio, (pad_w, pad_h)) from letterbox
        conf_thres: Confidence threshold for filtering
        iou_thres: IoU threshold for NMS
        reg_max: DFL regression maximum value (default: 16)
        strides: Feature map strides (default: [8, 16, 32])

    Returns:
        Tuple of (boxes, confidences, class_ids)
        - boxes: (M, 4) in xyxy format, scaled to original image
        - confidences: (M,) confidence scores
        - class_ids: (M,) class indices
    """
    assert preds.ndim == 3 and preds.shape[0] == 1
    pred = preds[0]
    n, c = pred.shape
    if c < 64:
        raise ValueError(f'Unexpected YOLOv8 head dims: {pred.shape}. Need (N, 64+nc)')
    nc = c - 4 * reg_max
    # Split
    raw_box = pred[:, : 4 * reg_max]
    cls_logits = pred[:, 4 * reg_max : 4 * reg_max + nc]
    # Decode DFL distances
    dfl = dfl_decode(raw_box, reg_max=reg_max)  # (N, 4)
    # Anchors
    anchors = make_anchors(list(strides), img_size)  # (N,2)
    if anchors.shape[0] != n:
        # Try alternate order: sometimes concat is in reverse
        alt = make_anchors(list(strides)[::-1], img_size)
        if alt.shape[0] == n:
            anchors = alt
    # l,t,r,b to xyxy
    cx, cy = anchors[:, 0], anchors[:, 1]
    l, t, r, b = dfl[:, 0], dfl[:, 1], dfl[:, 2], dfl[:, 3]
    # distances are in grid units, multiply by stride per level; approximate by nearest stride via anchor center
    # Determine stride per anchor by nearest fm step
    s_map = np.zeros(n, dtype=np.float32)
    idx = 0
    for s in strides:
        fm = img_size // s
        count = fm * fm
        s_map[idx : idx + count] = s
        idx += count
    if idx == n:
        l *= s_map
        t *= s_map
        r *= s_map
        b *= s_map
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b

    # Class scores
    scores = sigmoid(cls_logits)
    class_ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)
    mask = confs >= conf_thres
    boxes = np.stack([x1, y1, x2, y2], axis=1)[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    # NMS per-image
    keep = nms(boxes, confs, iou_thres=iou_thres)
    boxes, confs, class_ids = boxes[keep], confs[keep], class_ids[keep]

    # Scale boxes back to original image
    r, (dw, dh) = ratio_pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= r
    h0, w0 = orig_shape
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0 - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0 - 1)

    return boxes, confs, class_ids
