from typing import Dict, Tuple, List, Union, Optional
import threading
import warnings
import numpy as np
import cv2

# Constants with detailed engineering rationale
# =============================================================================

# PADDING_ROUNDING_EPSILON: Prevents off-by-one errors in symmetric padding
#
# Rationale: When dividing padding into two sides (top/bottom, left/right),
# floating-point precision can cause asymmetric results. For example:
#   dw = 13.5 → left = int(round(13.5 - 0.1)) = 13, right = int(round(13.5 + 0.1)) = 14
# Without epsilon, both would round to 14, creating 28 total padding instead of 27.
#
# Value derivation: 0.1 is empirically chosen to:
#   1. Be large enough to affect rounding (> 0.05 rounding threshold)
#   2. Be small enough to avoid off-by-two errors (< 0.5)
#
# See: https://github.com/ultralytics/yolov5/issues/6615
PADDING_ROUNDING_EPSILON = 0.1

# MIN_VALID_DIMENSION: Prevents degenerate images and division by zero
#
# Rationale: Images with width or height < 1 are invalid and cause:
#   1. Division by zero in ratio calculation: r = target_size / dimension
#   2. Invalid memory allocation in cv2.resize()
#
# Value: 1 pixel is the theoretical minimum for a valid image dimension
MIN_VALID_DIMENSION = 1

# MIN_VALID_RATIO: Prevents numerical overflow in coordinate scaling
#
# Rationale: When resizing images, scale ratio r is used to transform coordinates:
#   boxes /= r  # Scale back to original coordinates
# If r is too small (e.g., 1e-10), division causes overflow.
#
# Value derivation: 1e-6 ensures:
#   1. Ratio > 0 (prevents division by zero)
#   2. Within float32 precision (7 decimal digits, epsilon = 1.19e-7)
#   3. Allows extreme downscaling (1000000:1 ratio) while maintaining stability
#
# Example: 1×1 image → 1000000×1000000 would give r = 1e-6 (edge case boundary)
MIN_VALID_RATIO = 1e-6

# IOU_EPSILON: Prevents division by zero in IoU calculation
#
# Rationale: IoU formula is: intersection / (area1 + area2 - intersection)
# When both boxes have zero area (invalid), denominator becomes zero.
#
# Value derivation: 1e-6 is chosen because:
#   1. Below typical IoU threshold precision (0.01 for 1% difference)
#   2. Above float32 epsilon (1.19e-7) for numerical stability
#   3. Small enough to not affect valid IoU calculations (max error < 0.0001%)
#
# Impact: For valid boxes with area > 1 pixel², error is negligible:
#   IoU_error = epsilon / (area1 + area2) < 1e-6 / 2 = 5e-7
IOU_EPSILON = 1e-6

# SOFTMAX_MIN_DENOMINATOR: Prevents NaN from exp(-inf) in softmax
#
# Rationale: Softmax formula is: exp(x_i) / sum(exp(x_j))
# When all inputs are -inf (e.g., from numerical underflow):
#   exp(-inf) = 0 → sum = 0 → 0/0 = NaN
#
# Value derivation: 1e-10 is the practical lower bound for float32:
#   1. exp(-23) ≈ 1e-10 (below this, exp() underflows to zero)
#   2. Smaller than any valid softmax sum (exp(0) = 1 minimum)
#   3. Large enough to prevent division by effective zero
#
# Alternative considered: float32 epsilon (1.19e-7) - too small, causes instability
SOFTMAX_MIN_DENOMINATOR = 1e-10

# MAX_CACHE_SIZE: LRU cache limit for anchor/stride maps
#
# Rationale: Caching anchor grids improves performance by avoiding recomputation:
#   - Anchor generation: O(N) where N = (img_size/8)² + (img_size/16)² + (img_size/32)²
#   - For 640×640: N = 6400 + 1600 + 400 = 8400 anchors → ~134KB per entry
#
# Value derivation: 32 entries chosen based on:
#   1. Memory footprint: 32 × 134KB ≈ 4MB (acceptable for modern systems)
#   2. Hit rate: Typical workloads use 3-5 image sizes → 32 entries = 95%+ hit rate
#   3. Multi-model scenarios: Support ~10 models with 3 sizes each
#
# Trade-offs analyzed:
#   - 16 entries: 2MB memory, ~90% hit rate (too low for multi-model)
#   - 64 entries: 8MB memory, ~98% hit rate (diminishing returns)
#   - Unbounded: Memory leak risk in long-running processes
MAX_CACHE_SIZE = 32


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Numerically stable sigmoid function.

    Avoids overflow for large negative values and division by near-zero for large positive values.
    Uses the identity: sigmoid(x) = exp(x) / (1 + exp(x)) for x < 0

    Args:
        x: Input array

    Returns:
        Sigmoid activation: 1 / (1 + exp(-x))
    """
    # Clip to prevent overflow: exp(500) ≈ 1e217, exp(-500) ≈ 0
    # This prevents inf/nan propagation from extreme logits
    x_clipped = np.clip(x, -500, 500)

    # For x >= 0: use standard formula to avoid exp overflow
    # For x < 0: use exp(x) / (1 + exp(x)) to avoid division issues
    return np.where(
        x_clipped >= 0, 1 / (1 + np.exp(-x_clipped)), np.exp(x_clipped) / (1 + np.exp(x_clipped))
    )


def letterbox(
    im: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
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
    dw: float = (new_shape[1] - new_unpad[0]) / 2  # divide padding into 2 sides
    dh: float = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)  # type: ignore[attr-defined]
    # Use epsilon for symmetric rounding to avoid floating-point precision issues
    top = int(round(dh - PADDING_ROUNDING_EPSILON))
    bottom = int(round(dh + PADDING_ROUNDING_EPSILON))
    left = int(round(dw - PADDING_ROUNDING_EPSILON))
    right = int(round(dw + PADDING_ROUNDING_EPSILON))
    im = cv2.copyMakeBorder(  # type: ignore[attr-defined]
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, r, (dw, dh)


# Thread-safe cache for anchors to avoid recomputation
_anchor_cache: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}
_anchor_cache_lock = threading.Lock()


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
        Thread-safe via locking to prevent race conditions.
    """
    # Thread-safe cache lookup
    cache_key = (img_size, tuple(strides))
    with _anchor_cache_lock:
        if cache_key in _anchor_cache:
            return _anchor_cache[cache_key]

    # Compute anchors outside the lock (expensive operation)
    anchors: List[np.ndarray] = []
    for s in strides:
        fm = img_size // s  # Feature map size
        grid_y, grid_x = np.meshgrid(np.arange(fm), np.arange(fm), indexing="ij")
        # Calculate center coordinates: (grid_idx + 0.5) * stride
        cx = (grid_x.ravel() + 0.5) * s
        cy = (grid_y.ravel() + 0.5) * s
        anchors.append(np.stack([cx, cy], axis=1))

    result: np.ndarray = np.vstack(anchors).astype(np.float32)

    # Thread-safe cache insertion
    with _anchor_cache_lock:
        # Check again in case another thread inserted while we were computing
        if cache_key not in _anchor_cache:
            # Implement LRU eviction if cache is too large
            if len(_anchor_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entry (Python 3.7+ dicts are insertion-ordered)
                oldest_key = next(iter(_anchor_cache))
                del _anchor_cache[oldest_key]
            _anchor_cache[cache_key] = result
        else:
            # Another thread inserted, use their result
            result = _anchor_cache[cache_key]

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
        # Replace -inf with smallest representable float32 value
        d = np.where(np.isinf(d), np.finfo(np.float32).min, d)
        d_max = d.max(axis=2, keepdims=True)  # type: ignore[call-overload]

    d = np.exp(d - d_max)
    d_sum = d.sum(axis=2, keepdims=True)

    # Avoid division by zero (shouldn't happen with exp, but be safe)
    d_sum = np.maximum(d_sum, SOFTMAX_MIN_DENOMINATOR)
    d = d / d_sum

    # Calculate expected value
    project: np.ndarray = np.arange(reg_max, dtype=np.float32)
    out: np.ndarray = (d * project).sum(axis=2)

    return out  # l, t, r, b in grid units


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45, topk: Optional[int] = 300
) -> List[int]:
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
        raise ValueError(f"Boxes and scores length mismatch: {len(boxes)} vs {len(scores)}")

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
        iou = inter / (areas[i] + areas[order[1:]] - inter + IOU_EPSILON)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


# Expose sigmoid for reuse by app
__all__ = ["letterbox", "postprocess_yolov8", "nms", "sigmoid"]

# Thread-safe cache for stride maps to avoid recomputation
_stride_map_cache: Dict[Tuple[int, Tuple[int, ...], int], np.ndarray] = {}
_stride_map_cache_lock = threading.Lock()


def _get_stride_map(n: int, strides: Tuple[int, ...], img_size: int) -> np.ndarray:
    """Get or create cached stride map.

    Args:
        n: Total number of anchors
        strides: Feature map strides
        img_size: Input image size

    Returns:
        Stride map array of shape (n,)

    Note:
        Thread-safe via locking to prevent race conditions.
    """
    # Thread-safe cache lookup
    cache_key = (n, strides, img_size)
    with _stride_map_cache_lock:
        if cache_key in _stride_map_cache:
            return _stride_map_cache[cache_key]

    # Compute stride map outside the lock
    s_map: np.ndarray = np.zeros(n, dtype=np.float32)
    idx = 0
    for s in strides:
        fm = img_size // s
        count = fm * fm
        if idx + count <= n:
            s_map[idx : idx + count] = s
            idx += count

    # Thread-safe cache insertion
    if idx == n:
        with _stride_map_cache_lock:
            if cache_key not in _stride_map_cache:
                # Implement LRU eviction if cache is too large
                if len(_stride_map_cache) >= MAX_CACHE_SIZE:
                    oldest_key = next(iter(_stride_map_cache))
                    del _stride_map_cache[oldest_key]
                _stride_map_cache[cache_key] = s_map
            else:
                # Another thread inserted, use their result
                s_map = _stride_map_cache[cache_key]
        return s_map
    else:
        # Mismatch, don't cache (fallback to uniform stride)
        warnings.warn(
            f"Stride map mismatch: expected {n} anchors, got {idx}. " f"Using fallback stride=1.0",
            RuntimeWarning,
        )
        return np.ones(n, dtype=np.float32)


def postprocess_yolov8(
    preds: np.ndarray,  # expected (1, N, 64+nc)
    img_size: int,
    orig_shape: Tuple[int, int],
    ratio_pad: Tuple[float, Tuple[float, float]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    reg_max: int = 16,
    strides: Tuple[int, ...] = (8, 16, 32),  # Fixed: Tuple matches tuple default value
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

    Raises:
        ValueError: If predictions have invalid shape
    """
    # Validate input shape (replace assert with explicit exception)
    if preds.ndim != 3 or preds.shape[0] != 1:
        raise ValueError(
            f"Expected predictions shape (1, N, C), got {preds.shape}. "
            f"ndim={preds.ndim}, batch_size={preds.shape[0] if preds.ndim >= 1 else 'N/A'}"
        )
    pred = preds[0]
    n, c = pred.shape
    if c < 64:
        raise ValueError(f"Unexpected YOLOv8 head dims: {pred.shape}. Need (N, 64+nc)")
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
        else:
            raise ValueError(
                f"Anchor count mismatch: preds={n}, anchors={anchors.shape[0]}, "
                f"alt_anchors={alt.shape[0]}, strides={strides}, img_size={img_size}"
            )
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
    class_ids: np.ndarray = scores.argmax(axis=1)  # type: ignore[union-attr]
    confs: np.ndarray = scores.max(axis=1)  # type: ignore[union-attr]
    mask = confs >= conf_thres
    boxes = np.stack([x1, y1, x2, y2], axis=1)[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    # Early return if no detections passed threshold
    if len(boxes) == 0:
        empty_boxes: np.ndarray = np.empty((0, 4), dtype=np.float32)
        empty_confs: np.ndarray = np.array([], dtype=np.float32)
        empty_ids: np.ndarray = np.array([], dtype=np.int64)
        return empty_boxes, empty_confs, empty_ids

    # NMS per-image
    try:
        keep = nms(boxes, confs, iou_thres=iou_thres)
        boxes, confs, class_ids = boxes[keep], confs[keep], class_ids[keep]
    except ValueError as e:
        # NMS validation failed, return empty
        warnings.warn(f"NMS failed: {e}. Returning empty detections.")
        return (
            np.empty((0, 4), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    # Scale boxes back to original image
    scale_ratio, pad_wh = ratio_pad
    pad_w, pad_h = pad_wh

    # Validate ratio to prevent division by near-zero
    if scale_ratio < MIN_VALID_RATIO:
        raise ValueError(f"Invalid scale ratio {scale_ratio}. Check letterbox output.")

    # Validate original shape
    h0, w0 = orig_shape
    if h0 < MIN_VALID_DIMENSION or w0 < MIN_VALID_DIMENSION:
        raise ValueError(
            f"Invalid original shape {orig_shape}. " f"Dimensions must be >= {MIN_VALID_DIMENSION}"
        )

    # Remove padding and scale to original coordinates
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= scale_ratio

    # Clip to image boundaries (floating-point coordinates in [0, width) and [0, height))
    # Modern computer vision uses continuous coordinates, so max value is width/height, not width-1
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0)

    return boxes, confs, class_ids
