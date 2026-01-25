#!/usr/bin/env python3
"""快速性能测试 - 优化版 vs 原版对比

单文件版本，方便手动复制到板端
"""
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, '/root/rk-app')
from apps.utils.yolo_post import letterbox, postprocess_yolov8
from rknnlite.api import RKNNLite

# 简化的优化版DFL解码（纯numpy，无需numba）
def dfl_decode_fast(d, reg_max=16):
    """优化的DFL解码 - 减少内存分配"""
    d = d.reshape(-1, 4, reg_max)
    # 使用原地操作减少内存拷贝
    d_max = d.max(axis=2, keepdims=True)
    np.subtract(d, d_max, out=d)
    np.exp(d, out=d)
    d_sum = d.sum(axis=2, keepdims=True)
    np.maximum(d_sum, 1e-10, out=d_sum)
    d /= d_sum
    project = np.arange(reg_max, dtype=np.float32)
    return (d * project).sum(axis=2)

def sigmoid_fast(x):
    """优化的sigmoid - 减少临时数组"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def postprocess_fast(preds, img_size, orig_shape, ratio_pad, conf_thres=0.5, iou_thres=0.45):
    """快速后处理 - 内存优化版本"""
    from apps.utils.yolo_post import make_anchors, _get_stride_map, nms

    pred = preds[0]
    n, c = pred.shape
    nc = c - 64

    # 使用切片视图避免拷贝
    raw_box = pred[:, :64]
    cls_logits = pred[:, 64:64+nc]

    # 优化的DFL解码
    dfl = dfl_decode_fast(raw_box, 16)

    # 生成anchors
    anchors = make_anchors([8, 16, 32], img_size)
    if anchors.shape[0] != n:
        anchors = make_anchors([32, 16, 8], img_size)

    # 转换为boxes
    cx, cy = anchors[:, 0], anchors[:, 1]
    l, t, r, b = dfl[:, 0], dfl[:, 1], dfl[:, 2], dfl[:, 3]

    s_map = _get_stride_map(n, (8, 16, 32), img_size)
    l *= s_map
    t *= s_map
    r *= s_map
    b *= s_map

    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b

    # 优化的sigmoid
    scores = sigmoid_fast(cls_logits)
    class_ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)

    # 置信度过滤
    mask = confs >= conf_thres
    boxes = np.stack([x1, y1, x2, y2], axis=1)[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])

    # NMS
    keep = nms(boxes, confs, iou_thres)
    boxes = boxes[keep]
    confs = confs[keep]
    class_ids = class_ids[keep]

    # 缩放回原图
    scale_ratio, (pad_w, pad_h) = ratio_pad
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= scale_ratio

    h0, w0 = orig_shape
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0)

    return boxes, confs, class_ids

def benchmark(use_optimized=False, iterations=50):
    """性能测试"""
    print(f"\n{'='*70}")
    print(f"测试版本: {'优化版' if use_optimized else '原版'}")
    print(f"{'='*70}")

    # 加载模型
    rknn = RKNNLite()
    rknn.load_rknn('artifacts/models/yolo11n_416.rknn')
    rknn.init_runtime(core_mask=0x7)

    # 加载图片
    img = cv2.imread('assets/test.jpg')
    h, w = img.shape[:2]
    img_input, ratio, (dw, dh) = letterbox(img, 416)
    img_input = img_input[np.newaxis, ...]

    # 预热
    for _ in range(3):
        outputs = rknn.inference(inputs=[img_input])

    # 测试
    infer_times = []
    post_times = []
    total_times = []

    postprocess = postprocess_fast if use_optimized else postprocess_yolov8

    for i in range(iterations):
        t0 = time.perf_counter()

        # 推理
        t_infer_start = time.perf_counter()
        outputs = rknn.inference(inputs=[img_input])
        infer_times.append((time.perf_counter() - t_infer_start) * 1000)

        # 后处理
        pred = outputs[0]
        if pred.ndim == 2:
            pred = pred[None, ...]
        if pred.shape[1] < pred.shape[2]:
            pred = pred.transpose(0, 2, 1)

        t_post_start = time.perf_counter()
        boxes, confs, cls_ids = postprocess(pred, 416, (h, w), (ratio, (dw, dh)), 0.5, 0.45)
        post_times.append((time.perf_counter() - t_post_start) * 1000)

        total_times.append((time.perf_counter() - t0) * 1000)

        if (i+1) % 10 == 0:
            print(f"  迭代 {i+1:3d}/{iterations}: 后处理 {np.mean(post_times[-10:]):.2f}ms")

    rknn.release()

    # 统计
    infer_mean = np.mean(infer_times)
    post_mean = np.mean(post_times)
    total_mean = np.mean(total_times)

    print(f"\n结果:")
    print(f"  推理: {infer_mean:.2f}ms")
    print(f"  后处理: {post_mean:.2f}ms ({post_mean/total_mean*100:.1f}%)")
    print(f"  总延迟: {total_mean:.2f}ms")

    return {'infer': infer_mean, 'post': post_mean, 'total': total_mean}

if __name__ == '__main__':
    print("\n" + "="*70)
    print("端到端性能对比测试")
    print("="*70)

    # 测试原版
    result_orig = benchmark(use_optimized=False, iterations=50)

    print("\n" + "-"*70 + "\n")

    # 测试优化版
    result_opt = benchmark(use_optimized=True, iterations=50)

    # 对比
    print("\n" + "="*70)
    print("性能对比")
    print("="*70)
    print(f"后处理: {result_orig['post']:.2f}ms → {result_opt['post']:.2f}ms "
          f"(加速 {result_orig['post']/result_opt['post']:.2f}x)")
    print(f"总延迟: {result_orig['total']:.2f}ms → {result_opt['total']:.2f}ms "
          f"(减少 {result_orig['total']-result_opt['total']:.2f}ms)")
    print(f"任务书: {'✅ 合规' if result_opt['total'] <= 45 else '⚠️ 超出 ' + str(round(result_opt['total']-45, 2)) + 'ms'}")
    print("="*70 + "\n")
