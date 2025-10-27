#!/usr/bin/env python3
"""
PC模拟器推理脚本 - 无需板子即可验证RKNN功能
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rknn.api import RKNN
import cv2
import time

from apps.utils.preprocessing import preprocess_from_array_rknn_sim

def main():
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
    print('Loading ONNX model for PC simulation...')
    ret = rk.load_onnx(model='artifacts/models/yolo11n_416.onnx')
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
    img_path = 'assets/test.jpg'
    print(f'Loading test image: {img_path}')
    img = cv2.imread(img_path)
    if img is None:
        print(f'Failed to load image: {img_path}')
        return

    # 预处理：PC模拟器需要NHWC格式 (1, 416, 416, 3)
    inp = preprocess_from_array_rknn_sim(img, target_size=416)

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
