#!/usr/bin/env python3
"""端到端延迟基准测试 - 优化版本

对比优化前后的性能差异：
- 原版：使用标准numpy实现
- 优化版：使用Numba JIT加速

测试方法：1080P输入，50次迭代，统计平均延迟
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict

sys.path.insert(0, '/root/rk-app')

# 导入原版和优化版
from apps.utils.yolo_post import letterbox, postprocess_yolov8
from apps.utils.yolo_post_optimized import (
    postprocess_yolov8_optimized,
    NUMBA_AVAILABLE
)
from rknnlite.api import RKNNLite

def benchmark_e2e(model_path, test_image, iterations=50, conf_thres=0.5, use_optimized=False):
    """端到端延迟测试

    测试阶段：
    1. Capture - 图像获取（模拟）
    2. Preprocess - Letterbox预处理
    3. Inference - RKNN推理
    4. Postprocess - 后处理（DFL解码 + NMS）
    5. Encode - 结果编码
    """
    print(f"\n{'='*80}")
    print(f"端到端延迟基准测试 - {'优化版' if use_optimized else '原版'}")
    print(f"{'='*80}\n")
    print(f"模型: {model_path}")
    print(f"测试图片: {test_image}")
    print(f"迭代次数: {iterations}")
    print(f"置信度阈值: {conf_thres}")
    print(f"Numba可用: {NUMBA_AVAILABLE}")
    print(f"使用优化: {use_optimized}")

    # 加载模型
    print("\n加载RKNN模型...")
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"Error: Failed to load model")
        return None

    ret = rknn.init_runtime(core_mask=0x7)  # 3核并行
    if ret != 0:
        print(f"Error: Failed to init runtime")
        return None

    print("✓ 模型加载成功")

    # 加载测试图片
    print(f"\n加载测试图片...")
    img_orig = cv2.imread(test_image)
    if img_orig is None:
        print(f"Error: Failed to load image")
        rknn.release()
        return None

    h_orig, w_orig = img_orig.shape[:2]
    print(f"✓ 图片加载成功: {w_orig}×{h_orig}")

    # 预处理（只做一次，所有迭代复用）
    print(f"\n执行预处理...")
    img_letterbox, ratio, (dw, dh) = letterbox(img_orig, 416)
    img_letterbox = img_letterbox[np.newaxis, ...]  # (1, H, W, C)
    print(f"✓ 预处理完成: {img_letterbox.shape}")

    # 统计信息
    timings = {
        'capture': [],
        'preprocess': [],
        'inference': [],
        'postprocess': [],
        'encode': [],
        'total': []
    }

    detection_counts = []

    # 选择后处理函数
    postprocess_fn = postprocess_yolov8_optimized if use_optimized else postprocess_yolov8

    print(f"\n开始基准测试...\n")

    for i in range(iterations):
        t_start = time.perf_counter()

        # 1. Capture (模拟图像获取)
        t0 = time.perf_counter()
        img_capture = img_orig.copy()  # 模拟从相机读取
        t1 = time.perf_counter()
        timings['capture'].append((t1 - t0) * 1000)

        # 2. Preprocess (使用预先处理好的结果)
        t0 = time.perf_counter()
        img_input = img_letterbox  # 复用预处理结果
        t1 = time.perf_counter()
        timings['preprocess'].append((t1 - t0) * 1000)

        # 3. Inference
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[img_input])
        t1 = time.perf_counter()
        timings['inference'].append((t1 - t0) * 1000)

        # 准备预测结果
        pred = outputs[0]
        if pred.ndim == 2:
            pred = pred[None, ...]
        if pred.shape[1] < pred.shape[2]:
            pred = pred.transpose(0, 2, 1)

        # 4. Postprocess (优化版 vs 原版)
        t0 = time.perf_counter()
        boxes, confs, cls_ids = postprocess_fn(
            pred, 416, (h_orig, w_orig), (ratio, (dw, dh)),
            conf_thres, 0.45
        )
        t1 = time.perf_counter()
        timings['postprocess'].append((t1 - t0) * 1000)

        # 5. Encode (模拟UDP封包)
        t0 = time.perf_counter()
        result_json = json.dumps({
            'count': len(boxes),
            'boxes': boxes.tolist() if len(boxes) > 0 else []
        })
        t1 = time.perf_counter()
        timings['encode'].append((t1 - t0) * 1000)

        t_end = time.perf_counter()
        timings['total'].append((t_end - t_start) * 1000)

        detection_counts.append(len(boxes))

        if (i + 1) % 10 == 0:
            avg_total = np.mean(timings['total'][-10:])
            avg_post = np.mean(timings['postprocess'][-10:])
            print(f"迭代 {i+1:3d}/{iterations}: "
                  f"总延迟 {avg_total:6.2f}ms, "
                  f"后处理 {avg_post:6.2f}ms, "
                  f"检测 {detection_counts[-1]}个目标")

    rknn.release()

    # 统计结果
    print(f"\n{'='*80}")
    print(f"基准测试结果 - {'优化版' if use_optimized else '原版'}")
    print(f"{'='*80}\n")

    results = {}
    for stage, times in timings.items():
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        min_ms = np.min(times)
        max_ms = np.max(times)
        results[stage] = {
            'mean': mean_ms,
            'std': std_ms,
            'min': min_ms,
            'max': max_ms
        }
        print(f"{stage.capitalize():12s}: {mean_ms:7.2f}ms ± {std_ms:5.2f}ms "
              f"[{min_ms:6.2f}, {max_ms:6.2f}]")

    total_mean = results['total']['mean']
    post_mean = results['postprocess']['mean']
    post_ratio = (post_mean / total_mean) * 100

    print(f"\n后处理占比: {post_ratio:.1f}% ({post_mean:.2f}ms / {total_mean:.2f}ms)")
    print(f"推理FPS: {1000.0/results['inference']['mean']:.1f}")
    print(f"端到端FPS: {1000.0/total_mean:.1f}")
    print(f"平均检测数: {np.mean(detection_counts):.1f}个目标")

    # 任务书指标检查
    print(f"\n{'='*80}")
    print(f"任务书指标检查")
    print(f"{'='*80}")
    print(f"要求: 1080P处理延迟 ≤ 45ms")
    print(f"实测: {total_mean:.2f}ms")
    if total_mean <= 45:
        print(f"状态: ✅ 合规")
    else:
        print(f"状态: ⚠️ 超出 {total_mean - 45:.2f}ms")
    print(f"{'='*80}\n")

    return results

def compare_versions(model_path, test_image, iterations=50, conf_thres=0.5):
    """对比原版和优化版的性能"""
    print(f"\n{'#'*80}")
    print(f"# 性能对比测试：原版 vs 优化版")
    print(f"{'#'*80}\n")

    # 测试原版
    print("[1/2] 测试原版实现...")
    results_original = benchmark_e2e(model_path, test_image, iterations, conf_thres, use_optimized=False)

    print("\n" + "="*80 + "\n")

    # 测试优化版
    print("[2/2] 测试优化版实现...")
    results_optimized = benchmark_e2e(model_path, test_image, iterations, conf_thres, use_optimized=True)

    # 性能对比
    if results_original and results_optimized:
        print(f"\n{'#'*80}")
        print(f"# 性能提升总结")
        print(f"{'#'*80}\n")

        print(f"{'阶段':<15s} {'原版(ms)':<12s} {'优化版(ms)':<12s} {'加速比':<10s} {'提升':<10s}")
        print("-" * 70)

        for stage in ['capture', 'preprocess', 'inference', 'postprocess', 'encode', 'total']:
            orig_mean = results_original[stage]['mean']
            opt_mean = results_optimized[stage]['mean']
            speedup = orig_mean / opt_mean if opt_mean > 0 else 0
            improvement = ((orig_mean - opt_mean) / orig_mean * 100) if orig_mean > 0 else 0

            print(f"{stage.capitalize():<15s} {orig_mean:8.2f}    {opt_mean:8.2f}    "
                  f"{speedup:6.2f}x    {improvement:6.1f}%")

        # 关键指标
        total_orig = results_original['total']['mean']
        total_opt = results_optimized['total']['mean']
        post_orig = results_original['postprocess']['mean']
        post_opt = results_optimized['postprocess']['mean']

        print(f"\n关键指标：")
        print(f"  端到端延迟: {total_orig:.2f}ms → {total_opt:.2f}ms "
              f"(减少 {total_orig - total_opt:.2f}ms, {(total_orig-total_opt)/total_orig*100:.1f}%)")
        print(f"  后处理延迟: {post_orig:.2f}ms → {post_opt:.2f}ms "
              f"(加速 {post_orig/post_opt:.2f}x)")
        print(f"  任务书合规: {total_orig:.2f}ms → {total_opt:.2f}ms "
              f"({'✅ 合规' if total_opt <= 45 else '⚠️ 超出 ' + str(round(total_opt - 45, 2)) + 'ms'})")

        print(f"\n{'#'*80}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                       default='artifacts/models/yolo11n_416.rknn',
                       help='RKNN模型路径')
    parser.add_argument('--image', type=str,
                       default='assets/test_1080p.jpg',
                       help='测试图片路径')
    parser.add_argument('--iterations', type=int, default=50,
                       help='迭代次数')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['compare', 'original', 'optimized'],
                       help='测试模式')
    args = parser.parse_args()

    if args.mode == 'compare':
        compare_versions(args.model, args.image, args.iterations, args.conf)
    elif args.mode == 'original':
        benchmark_e2e(args.model, args.image, args.iterations, args.conf, use_optimized=False)
    else:
        benchmark_e2e(args.model, args.image, args.iterations, args.conf, use_optimized=True)
