#!/usr/bin/env python3
"""视频多类别检测测试"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from collections import Counter

# COCO 80类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

sys.path.insert(0, '/root/rk-app')
from apps.utils.yolo_post import letterbox, postprocess_yolov8
from rknnlite.api import RKNNLite

def test_video(model_path, video_path, max_frames=100, conf_thres=0.3):
    """测试视频的多类别检测"""
    print(f"\n{'='*80}")
    print(f"视频多类别检测测试")
    print(f"{'='*80}\n")
    print(f"视频文件: {video_path}")
    print(f"最大帧数: {max_frames}")
    print(f"置信度阈值: {conf_thres}")

    # 加载模型
    print("\n加载RKNN模型...")
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"Error: Failed to load model")
        return

    ret = rknn.init_runtime(core_mask=0x7)
    if ret != 0:
        print(f"Error: Failed to init runtime")
        return

    # 打开视频
    print(f"打开视频文件...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video")
        rknn.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n视频信息:")
    print(f"  分辨率: {width}×{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  总帧数: {total_frames}")

    # 统计信息
    all_classes = set()
    class_counter_total = Counter()
    frame_count = 0
    inference_times = []
    total_detections = 0

    print(f"\n开始处理...\n")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"视频读取结束（共{frame_count}帧）")
            break

        frame_count += 1

        # 预处理
        img_input, ratio, (dw, dh) = letterbox(frame, 416)
        img_input = img_input[np.newaxis, ...]

        # 推理
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[img_input])
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

        pred = outputs[0]
        if pred.ndim == 2:
            pred = pred[None, ...]
        if pred.shape[1] < pred.shape[2]:
            pred = pred.transpose(0, 2, 1)

        # 后处理
        boxes, confs, cls_ids = postprocess_yolov8(
            pred, 416, (416, 416), (ratio, (dw, dh)), conf_thres, 0.45
        )

        # 统计类别
        if len(cls_ids) > 0:
            all_classes.update(cls_ids)
            class_counter_total.update(cls_ids)
            total_detections += len(cls_ids)

        if frame_count % 10 == 0:
            avg_infer = np.mean(inference_times[-10:])
            print(f"帧 {frame_count}/{min(max_frames, total_frames)}: "
                  f"检测{len(boxes)}个目标, "
                  f"推理{avg_infer:.1f}ms, "
                  f"累计类别{len(all_classes)}种")

    cap.release()
    rknn.release()

    # 输出结果
    print(f"\n{'='*80}")
    print(f"测试结果")
    print(f"{'='*80}\n")
    print(f"处理帧数: {frame_count}")
    print(f"总检测次数: {total_detections}")
    print(f"平均推理时间: {np.mean(inference_times):.2f}ms")
    print(f"推理FPS: {1000.0/np.mean(inference_times):.1f}")
    print(f"\n检测到的类别总数: {len(all_classes)}")

    if len(all_classes) > 0:
        print(f"\n各类别检测次数:")
        print(f"{'类别ID':<10} {'类别名称':<20} {'检测次数':<10}")
        print("-" * 50)

        for cls_id in sorted(all_classes):
            count = class_counter_total[cls_id]
            cls_name = COCO_CLASSES[int(cls_id)] if 0 <= int(cls_id) < len(COCO_CLASSES) else f"class_{cls_id}"
            print(f"{int(cls_id):<10} {cls_name:<20} {count:<10}")

    print(f"\n{'='*80}")
    print(f"任务书合规性检查")
    print(f"{'='*80}")
    print(f"要求: 检测种类 > 10种")
    print(f"实测: {len(all_classes)}种")
    print(f"模型支持: 80种 (COCO数据集)")
    if len(all_classes) > 10:
        print(f"状态: ✅ 合规")
    else:
        print(f"状态: ⚠️ 当前视频仅包含{len(all_classes)}类，但模型支持80类")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='视频文件路径')
    parser.add_argument('--max-frames', type=int, default=100, help='最大处理帧数')
    parser.add_argument('--conf', type=float, default=0.3, help='置信度阈值')
    args = parser.parse_args()

    test_video('artifacts/models/yolo11n_416.rknn', args.video, args.max_frames, args.conf)
