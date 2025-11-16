import math
from typing import Tuple, List, Union, Optional
import numpy as np
import cv2

# Constants
PADDING_ROUNDING_EPSILON = 0.1  # Small epsilon for padding rounding to avoid floating-point errors
MIN_VALID_DIMENSION = 1  # Minimum valid image dimension
MIN_VALID_RATIO = 1e-6  # Minimum valid scale ratio to prevent division issues


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

    Raises:
        ValueError: If image dimensions are invalid or image is not 3D
    """
    # Validate input
    if im.ndim != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got shape {im.shape}")

    shape = im.shape[:2]  # current shape [h, w]

    # Validate image dimensions
    if shape[0] < MIN_VALID_DIMENSION or shape[1] < MIN_VALID_DIMENSION:
        raise ValueError(
            f"Invalid image dimensions {shape}. "
            f"Both height and width must be >= {MIN_VALID_DIMENSION}"
        )

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calculate resize ratio with safety check
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Ensure ratio is not too small to prevent numerical issues
    if r < MIN_VALID_RATIO:
        raise ValueError(
            f"Resize ratio {r} too small. Image dimensions {shape} vs target {new_shape}"
        )
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # Use epsilon for symmetric rounding to avoid floating-point precision issues
    top = int(round(dh - PADDING_ROUNDING_EPSILON))
    bottom = int(round(dh + PADDING_ROUNDING_EPSILON))
    left = int(round(dw - PADDING_ROUNDING_EPSILON))
    right = int(round(dw + PADDING_ROUNDING_EPSILON))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


# Cache for anchors to avoid recomputation
_anchor_cache = {}

def make_anchors(strides: List[int], img_size: int) -> np.ndarray:
    """Generate anchor grid points for YOLO detection layers.

    Args:
        strides: List of feature map strides (e.g., [8, 16, 32])
        img_size: Input image size

    Returns:
        Anchor points array of shape (N, 2) where N is total number of anchors.
        Each row contains (center_x, center_y) coordinates.

    Note:
        Results are cached based on (img_size, strides) for performance.
    """
    # Use cache to avoid recomputation
    cache_key = (img_size, tuple(strides))
    if cache_key in _anchor_cache:
        return _anchor_cache[cache_key]

    anchors = []
    for s in strides:
        fm = img_size // s  # Feature map size
        grid_y, grid_x = np.meshgrid(np.arange(fm), np.arange(fm), indexing='ij')
        # Calculate center coordinates: (grid_idx + 0.5) * stride
        cx = (grid_x.ravel() + 0.5) * s
        cy = (grid_y.ravel() + 0.5) * s
        anchors.append(np.stack([cx, cy], axis=1))

    result = np.vstack(anchors).astype(np.float32)
    _anchor_cache[cache_key] = result
    return result


def dfl_decode(d: np.ndarray, reg_max: int = 16) -> np.ndarray:
    """Decode Distribution Focal Loss (DFL) predictions to bounding box distances.

    Args:
        d: DFL predictions, shape (N, 4*reg_max)
        reg_max: Maximum regression value (default: 16)

    Returns:
        Decoded distances (left, top, right, bottom), shape (N, 4)

    Note:
        Uses softmax to convert distributions to expected values.
        Handles numerical stability for extreme input values.
    """
    # d: (N, 4*reg_max) -> (N, 4, reg_max)
    d = d.reshape(-1, 4, reg_max)

    # Numerically stable softmax along reg_max dimension
    d_max = d.max(axis=2, keepdims=True)

    # Check for all -inf case (would cause nan)
    if np.any(np.isinf(d_max)):
        # Replace -inf with large negative value
        d = np.where(np.isinf(d), -1e10, d)
        d_max = d.max(axis=2, keepdims=True)

    d = np.exp(d - d_max)
    d_sum = d.sum(axis=2, keepdims=True)

    # Avoid division by zero (shouldn't happen with exp, but be safe)
    d_sum = np.maximum(d_sum, 1e-10)
    d = d / d_sum

    # Calculate expected value
    project = np.arange(reg_max, dtype=np.float32)
    out = (d * project).sum(axis=2)

    return out  # l, t, r, b in grid units


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45, topk: Optional[int] = 300) -> List[int]:
    """Non-Maximum Suppression for object detection.

    Args:
        boxes: Bounding boxes in xyxy format, shape (M, 4)
        scores: Confidence scores, shape (M,)
        iou_thres: IoU threshold for suppression
        topk: Maximum number of boxes to keep before NMS (None for no limit)

    Returns:
        List of indices to keep

    Raises:
        ValueError: If boxes/scores are invalid or have mismatched lengths

    Note:
        Uses floating-point coordinate system (modern YOLO).
        Removed legacy +1 correction for integer coordinates.
    """
    # Early return for empty input
    if len(boxes) == 0:
        return []

    # Validate input shapes
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes shape (M, 4), got {boxes.shape}")

    if len(scores) != len(boxes):
        raise ValueError(
            f"Boxes and scores length mismatch: {len(boxes)} vs {len(scores)}"
        )

    # boxes: (M, 4) xyxy, scores: (M,)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Validate box coordinates: x2 >= x1 and y2 >= y1
    invalid_boxes = (x2 < x1) | (y2 < y1)
    if np.any(invalid_boxes):
        # Swap coordinates if reversed
        x1_tmp, x2_tmp = np.minimum(x1, x2), np.maximum(x1, x2)
        y1_tmp, y2_tmp = np.minimum(y1, y2), np.maximum(y1, y2)
        x1, x2 = x1_tmp, x2_tmp
        y1, y2 = y1_tmp, y2_tmp

    # Calculate areas (now guaranteed non-negative)
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

# Cache for stride maps to avoid recomputation
_stride_map_cache = {}

def _get_stride_map(n: int, strides: Tuple[int, ...], img_size: int) -> np.ndarray:
    """Get or create cached stride map.

    Args:
        n: Total number of anchors
        strides: Feature map strides
        img_size: Input image size

    Returns:
        Stride map array of shape (n,)
    """
    cache_key = (n, strides, img_size)
    if cache_key in _stride_map_cache:
        return _stride_map_cache[cache_key]

    s_map = np.zeros(n, dtype=np.float32)
    idx = 0
    for s in strides:
        fm = img_size // s
        count = fm * fm
        if idx + count <= n:
            s_map[idx : idx + count] = s
            idx += count

    if idx == n:
        _stride_map_cache[cache_key] = s_map
        return s_map
    else:
        # Mismatch, don't cache
        return np.ones(n, dtype=np.float32)


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
    # distances are in grid units, multiply by stride per level
    # Use cached stride map for performance
    s_map = _get_stride_map(n, tuple(strides), img_size)
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

    # Early return if no detections passed threshold
    if len(boxes) == 0:
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_confs = np.array([], dtype=np.float32)
        empty_ids = np.array([], dtype=np.int64)
        return empty_boxes, empty_confs, empty_ids

    # NMS per-image
    try:
        keep = nms(boxes, confs, iou_thres=iou_thres)
        boxes, confs, class_ids = boxes[keep], confs[keep], class_ids[keep]
    except ValueError as e:
        # NMS validation failed, return empty
        import warnings
        warnings.warn(f"NMS failed: {e}. Returning empty detections.")
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_confs = np.array([], dtype=np.float32)
        empty_ids = np.array([], dtype=np.int64)
        return empty_boxes, empty_confs, empty_ids

    # Scale boxes back to original image
    r, (dw, dh) = ratio_pad

    # Validate ratio to prevent division by near-zero
    if r < MIN_VALID_RATIO:
        raise ValueError(
            f"Invalid scale ratio {r}. Check letterbox output."
        )

    # Validate original shape
    h0, w0 = orig_shape
    if h0 < MIN_VALID_DIMENSION or w0 < MIN_VALID_DIMENSION:
        raise ValueError(
            f"Invalid original shape {orig_shape}. "
            f"Dimensions must be >= {MIN_VALID_DIMENSION}"
        )

    # Remove padding and scale to original coordinates
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= r

    # Clip to image boundaries
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0 - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0 - 1)

    return boxes, confs, class_ids
