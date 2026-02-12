#!/usr/bin/env python3
"""端到端延时基准测试脚本（模拟1080P工业相机输入）

测试完整流水线：
1. 图像采集（从文件/模拟1080P输入）
2. 预处理（letterbox resize）
3. RKNN推理
4. 后处理（解码+NMS）
5. 结果编码（模拟网络上传准备）
"""
import argparse
import time
import json
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apps.utils.yolo_post import letterbox, postprocess_yolov8


def benchmark_e2e(args):
    """运行端到端延时基准测试"""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("Error: rknn-toolkit-lite2 not installed")
        return 1

    # 1. 加载测试图像
    print(f"Loading test image: {args.image}")
    img_orig = cv2.imread(str(args.image))
    if img_orig is None:
        print(f"Error: Failed to read image {args.image}")
        return 1

    orig_h, orig_w = img_orig.shape[:2]
    print(f"Original image size: {orig_w}×{orig_h}")

    # 如果需要模拟1080P输入，resize到1920×1080
    if args.simulate_1080p:
        print("Simulating 1080P camera input (resize to 1920×1080)")
        img_orig = cv2.resize(img_orig, (1920, 1080))
        orig_h, orig_w = 1080, 1920

    # 2. 加载RKNN模型
    print(f"\nLoading RKNN model: {args.model}")
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(args.model))
    if ret != 0:
        print("Error: load_rknn failed")
        return 1

    ret = rknn.init_runtime(core_mask=args.core_mask)
    if ret != 0:
        print("Error: init_runtime failed")
        return 1

    print(f"RKNN runtime initialized (core_mask={hex(args.core_mask)})")

    # 3. Warmup
    print(f"\nWarming up ({args.warmup} iterations)...")
    img_letterbox, ratio, (dw, dh) = letterbox(img_orig, args.imgsz)
    for i in range(args.warmup):
        rknn.inference(inputs=[img_letterbox])

    # 4. 基准测试
    print(f"\nRunning benchmark ({args.runs} iterations)...")
    print(f"Pipeline: 1080P({orig_w}×{orig_h}) → Preprocess → RKNN({args.imgsz}×{args.imgsz}) → Postprocess → Encode")
    print("-" * 80)

    postprocess_fn = postprocess_yolov8
    if args.optimized:
        try:
            from apps.utils.yolo_post_optimized import NUMBA_AVAILABLE, postprocess_yolov8_optimized

            postprocess_fn = postprocess_yolov8_optimized
            print(f"Postprocess mode: optimized (NUMBA_AVAILABLE={NUMBA_AVAILABLE})")
        except ImportError:
            print("Postprocess mode: optimized requested but unavailable, fallback to standard")
    else:
        print("Postprocess mode: standard")

    timings = {
        'capture': [],      # 模拟相机采集
        'preprocess': [],   # letterbox resize
        'inference': [],    # RKNN推理
        'postprocess': [],  # 解码+NMS
        'encode': [],       # 结果编码
        'total': []         # 端到端总延时
    }

    for run_idx in range(args.runs):
        t_start = time.perf_counter()

        # Stage 1: 模拟相机采集（从内存读取）
        t0 = time.perf_counter()
        frame = img_orig.copy()
        t1 = time.perf_counter()
        timings['capture'].append(t1 - t0)

        # Stage 2: 预处理（letterbox resize）
        t2 = time.perf_counter()
        img_letterbox, ratio, (dw, dh) = letterbox(frame, args.imgsz)
        t3 = time.perf_counter()
        timings['preprocess'].append(t3 - t2)

        # Stage 3: RKNN推理
        t4 = time.perf_counter()
        outputs = rknn.inference(inputs=[img_letterbox])
        t5 = time.perf_counter()
        timings['inference'].append(t5 - t4)

        # Stage 4: 后处理（解码+NMS）
        t6 = time.perf_counter()
        pred = outputs[0]
        if pred.ndim == 2:
            pred = pred[None, ...]
        if pred.shape[1] < pred.shape[2]:
            pred = pred.transpose(0, 2, 1)

        boxes, confs, cls_ids = postprocess_fn(
            pred, args.imgsz, (args.imgsz, args.imgsz),
            (ratio, (dw, dh)), args.conf, args.iou
        )
        t7 = time.perf_counter()
        timings['postprocess'].append(t7 - t6)

        # Stage 5: 结果编码（模拟网络上传准备）
        t8 = time.perf_counter()
        detections = [
            {
                'xyxy': [int(x1), int(y1), int(x2), int(y2)],
                'conf': float(c),
                'cls': int(ci)
            }
            for (x1, y1, x2, y2), c, ci in zip(boxes, confs, cls_ids)
        ]
        payload = json.dumps({'detections': detections})
        t9 = time.perf_counter()
        timings['encode'].append(t9 - t8)

        # 总延时
        t_end = time.perf_counter()
        timings['total'].append(t_end - t_start)

        if (run_idx + 1) % 10 == 0 or run_idx == 0:
            print(f"Run {run_idx+1}/{args.runs}: Total={timings['total'][-1]*1000:.2f}ms, "
                  f"Infer={timings['inference'][-1]*1000:.2f}ms, "
                  f"Detections={len(detections)}")

    # 5. 统计结果
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    def print_stats(name, values):
        values_ms = [v * 1000 for v in values]
        avg = np.mean(values_ms)
        std = np.std(values_ms)
        min_val = np.min(values_ms)
        max_val = np.max(values_ms)
        p50 = np.percentile(values_ms, 50)
        p95 = np.percentile(values_ms, 95)
        p99 = np.percentile(values_ms, 99)
        print(f"{name:15s}: avg={avg:6.2f}ms  std={std:5.2f}ms  "
              f"min={min_val:6.2f}ms  max={max_val:6.2f}ms  "
              f"p50={p50:6.2f}ms  p95={p95:6.2f}ms  p99={p99:6.2f}ms")

    print_stats("Capture", timings['capture'])
    print_stats("Preprocess", timings['preprocess'])
    print_stats("Inference", timings['inference'])
    print_stats("Postprocess", timings['postprocess'])
    print_stats("Encode", timings['encode'])
    print("-" * 80)
    print_stats("TOTAL (E2E)", timings['total'])

    # 计算FPS
    avg_total_ms = np.mean(timings['total']) * 1000
    fps = 1000.0 / avg_total_ms
    print(f"\nThroughput: {fps:.1f} FPS (avg latency={avg_total_ms:.2f}ms)")

    # 任务书合规性检查
    print("\n" + "=" * 80)
    print("任务书合规性检查")
    print("=" * 80)
    print(f"任务书要求: 1080P图像处理延时 ≤45ms")
    print(f"实测端到端延时: {avg_total_ms:.2f}ms")
    if avg_total_ms <= 45:
        print(f"✅ 合规 (余量={45-avg_total_ms:.2f}ms)")
    else:
        print(f"❌ 不合规 (超出={avg_total_ms-45:.2f}ms)")

    # 保存详细结果
    if args.output:
        results = {
            'config': {
                'model': str(args.model),
                'image': str(args.image),
                'original_size': f"{orig_w}×{orig_h}",
                'input_size': f"{args.imgsz}×{args.imgsz}",
                'core_mask': hex(args.core_mask),
                'conf_threshold': args.conf,
                'iou_threshold': args.iou,
                'runs': args.runs
            },
            'timings': {k: [v*1000 for v in vals] for k, vals in timings.items()},
            'statistics': {
                k: {
                    'avg_ms': float(np.mean([v*1000 for v in vals])),
                    'std_ms': float(np.std([v*1000 for v in vals])),
                    'min_ms': float(np.min([v*1000 for v in vals])),
                    'max_ms': float(np.max([v*1000 for v in vals])),
                    'p50_ms': float(np.percentile([v*1000 for v in vals], 50)),
                    'p95_ms': float(np.percentile([v*1000 for v in vals], 95)),
                    'p99_ms': float(np.percentile([v*1000 for v in vals], 99))
                }
                for k, vals in timings.items()
            },
            'compliance': {
                'requirement': '1080P处理延时 ≤45ms',
                'measured_ms': float(avg_total_ms),
                'passed': bool(avg_total_ms <= 45),
                'margin_ms': float(45 - avg_total_ms)
            }
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    rknn.release()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='端到端延时基准测试（1080P工业相机模拟）'
    )
    parser.add_argument('--model', type=Path, required=True,
                        help='RKNN模型路径')
    parser.add_argument('--image', type=Path, required=True,
                        help='测试图像路径')
    parser.add_argument('--imgsz', type=int, default=416,
                        help='模型输入尺寸 (default: 416)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值 (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU阈值 (default: 0.45)')
    parser.add_argument('--core-mask', type=lambda x: int(x, 0), default=0x7,
                        help='NPU核心掩码 (default: 0x7=3核)')
    parser.add_argument('--simulate-1080p', action='store_true',
                        help='将输入图像resize到1920×1080模拟1080P相机')
    parser.add_argument('--runs', type=int, default=50,
                        help='测试运行次数 (default: 50)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热次数 (default: 10)')
    parser.add_argument('--output', type=Path, default=None,
                        help='保存详细结果到JSON文件')
    parser.add_argument('--optimized', action='store_true',
                        help='启用优化后处理实现（如果可用）')

    args = parser.parse_args(argv)
    return benchmark_e2e(args)


if __name__ == '__main__':
    exit(main())
