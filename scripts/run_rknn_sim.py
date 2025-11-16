#!/usr/bin/env python3
"""
PC模拟器推理脚本 - 无需板子即可验证RKNN功能
"""
import sys
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rknn.api import RKNN
import cv2
import time

from apps.utils.preprocessing import preprocess_from_array_rknn_sim

def main():
    parser = argparse.ArgumentParser(description='Run RKNN inference on PC simulator')
    parser.add_argument('--model', type=str, default='artifacts/models/yolo11n_416.onnx',
                        help='Path to ONNX model (default: artifacts/models/yolo11n_416.onnx)')
    parser.add_argument('--image', type=str, default='assets/test.jpg',
                        help='Path to test image (default: assets/test.jpg)')
    parser.add_argument('--imgsz', type=int, default=416,
                        help='Input image size (default: 416)')
    args = parser.parse_args()

    rk = RKNN()

    # 配置RKNN（必须在load之前）
    print('Configuring RKNN...')
    ret = rk.config(
        target_platform='rk3588',
        optimization_level=3
    )
    if ret != 0:
        print('Config failed!')
        return

    # PC模拟器需要从ONNX重新加载和build
    print(f'Loading ONNX model for PC simulation: {args.model}')
    ret = rk.load_onnx(model=args.model)
    if ret != 0:
        print('Load ONNX failed!')
        return

    # 在PC上build（用于模拟器）
    print('Building for PC simulator...')
    ret = rk.build(do_quantization=False)  # PC模拟器暂时不量化，快速验证
    if ret != 0:
        print('Build failed!')
        return

    # PC模拟器模式初始化
    print('Init runtime on PC simulator...')
    ret = rk.init_runtime()
    if ret != 0:
        print('Init runtime failed!')
        return

    # 准备测试图片
    print(f'Loading test image: {args.image}')
    img = cv2.imread(args.image)
    if img is None:
        print(f'Failed to load image: {args.image}')
        return

    # 预处理：PC模拟器需要NHWC格式
    inp = preprocess_from_array_rknn_sim(img, target_size=args.imgsz)

    # 推理
    print('Running inference...')
    st = time.time()
    outputs = rk.inference(inputs=[inp], data_format='nhwc')
    dt = (time.time() - st) * 1000

    # 打印结果
    print(f'\n=== PC Simulator Results ===')
    print(f'Latency: {dt:.2f} ms')
    print(f'Output shapes: {[o.shape for o in outputs]}')
    print(f'Output dtypes: {[o.dtype for o in outputs]}')

    # 简单统计输出范围
    for i, out in enumerate(outputs):
        print(f'\nOutput[{i}]:')
        print(f'  Shape: {out.shape}')
        print(f'  Dtype: {out.dtype}')
        print(f'  Range: [{out.min()}, {out.max()}]')
        print(f'  Mean: {out.mean():.4f}')

    rk.release()
    print('\nDone!')

if __name__ == '__main__':
    main()
