#!/usr/bin/env python3
"""PC模拟器推理脚本 - 无需板子即可验证RKNN功能

This script runs RKNN inference on PC simulator for boardless validation.
It loads ONNX models, builds for RK3588 target, and runs inference without
requiring actual hardware.
"""
import sys
import argparse
from pathlib import Path
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

# Import custom exceptions and logger
from apps.exceptions import ModelLoadError, ConfigurationError, InferenceError, PreprocessError
from apps.logger import setup_logger
from apps.utils.preprocessing import preprocess_from_array_rknn_sim

# Setup logger
logger = setup_logger(__name__, level='INFO')

def main():
    from apps.config import PathConfig

    parser = argparse.ArgumentParser(description='Run RKNN inference on PC simulator')
    parser.add_argument('--model', type=str, default=PathConfig.YOLO11N_ONNX_416,
                        help=f'Path to ONNX model (default: {PathConfig.YOLO11N_ONNX_416})')
    parser.add_argument('--image', type=str, default=PathConfig.TEST_IMAGE,
                        help=f'Path to test image (default: {PathConfig.TEST_IMAGE})')
    parser.add_argument('--imgsz', type=int, default=416,
                        help='Input image size (default: 416)')
    args = parser.parse_args()

    # Import RKNN API with error handling
    try:
        from rknn.api import RKNN
    except ImportError as e:
        raise ConfigurationError(
            f"rknn-toolkit2 not installed. Please run: pip install rknn-toolkit2\nError: {e}"
        ) from e

    # Validate model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {args.model}")

    # Validate image file exists
    image_path = Path(args.image)
    if not image_path.exists():
        raise PreprocessError(f"Image file not found: {args.image}")

    rk = RKNN()

    # 配置RKNN（必须在load之前）
    logger.info('Configuring RKNN for RK3588 platform...')
    ret = rk.config(
        target_platform='rk3588',
        optimization_level=3
    )
    if ret != 0:
        rk.release()
        raise ConfigurationError('RKNN config failed')

    # PC模拟器需要从ONNX重新加载和build
    logger.info(f'Loading ONNX model for PC simulation: {args.model}')
    ret = rk.load_onnx(model=str(args.model))
    if ret != 0:
        rk.release()
        raise ModelLoadError(f'Failed to load ONNX model: {args.model}')

    # 在PC上build（用于模拟器）
    logger.info('Building for PC simulator (no quantization for faster validation)...')
    ret = rk.build(do_quantization=False)
    if ret != 0:
        rk.release()
        raise ModelLoadError('RKNN build failed')

    # PC模拟器模式初始化
    logger.info('Initializing runtime on PC simulator...')
    ret = rk.init_runtime()
    if ret != 0:
        rk.release()
        raise InferenceError('RKNN init runtime failed')

    # 准备测试图片
    logger.info(f'Loading test image: {args.image}')
    img = cv2.imread(str(args.image))
    if img is None:
        rk.release()
        raise PreprocessError(f'Failed to load image: {args.image}')

    # 预处理：PC模拟器需要NHWC格式
    logger.info(f'Preprocessing image to NHWC format (target_size={args.imgsz})')
    try:
        inp = preprocess_from_array_rknn_sim(img, target_size=args.imgsz)
    except Exception as e:
        rk.release()
        raise PreprocessError(f'Image preprocessing failed: {e}') from e

    # 推理
    logger.info('Running inference on PC simulator...')
    try:
        st = time.time()
        outputs = rk.inference(inputs=[inp], data_format='nhwc')
        dt = (time.time() - st) * 1000
    except Exception as e:
        rk.release()
        raise InferenceError(f'RKNN inference failed: {e}') from e

    # 打印结果
    logger.info('\n=== PC Simulator Results ===')
    logger.info(f'Latency: {dt:.2f} ms')
    logger.info(f'Output shapes: {[o.shape for o in outputs]}')
    logger.info(f'Output dtypes: {[o.dtype for o in outputs]}')

    # 简单统计输出范围
    for i, out in enumerate(outputs):
        logger.info(f'\nOutput[{i}]:')
        logger.info(f'  Shape: {out.shape}')
        logger.info(f'  Dtype: {out.dtype}')
        logger.info(f'  Range: [{out.min()}, {out.max()}]')
        logger.info(f'  Mean: {out.mean():.4f}')

    rk.release()
    logger.info('\nPC simulator inference completed successfully!')
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ModelLoadError, ConfigurationError, InferenceError, PreprocessError) as e:
        logger.error(f"PC simulator inference failed: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
