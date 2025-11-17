#!/usr/bin/env python3
"""ONNX vs RKNN 精度对拍脚本

This script compares inference outputs between ONNX Runtime and RKNN PC simulator
to validate accuracy preservation during model conversion.
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json

# Import custom exceptions and logger
from apps.exceptions import ModelLoadError, ConfigurationError, InferenceError, ValidationError
from apps.logger import setup_logger
from apps.utils.preprocessing import preprocess_onnx, preprocess_rknn_sim

# Setup logger
logger = setup_logger(__name__, level='INFO')

def run_onnx_inference(model_path, img_path):
    """Run ONNX inference with error handling."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ConfigurationError(
            f"onnxruntime not installed. Please run: pip install onnxruntime\nError: {e}"
        ) from e

    try:
        sess = ort.InferenceSession(str(model_path))
    except Exception as e:
        raise ModelLoadError(f"Failed to load ONNX model {model_path}: {e}") from e

    try:
        inp = preprocess_onnx(img_path, target_size=416)
        outputs = sess.run(None, {sess.get_inputs()[0].name: inp})
        return outputs[0]
    except Exception as e:
        raise InferenceError(f"ONNX inference failed for {img_path}: {e}") from e

def run_rknn_inference(model_path, img_path):
    """Run RKNN PC simulator inference with error handling."""
    try:
        from rknn.api import RKNN
    except ImportError as e:
        raise ConfigurationError(
            f"rknn-toolkit2 not installed. Please run: pip install rknn-toolkit2\nError: {e}"
        ) from e

    rk = RKNN()

    try:
        # 配置并加载ONNX（用于模拟）
        ret = rk.config(target_platform='rk3588', optimization_level=3)
        if ret != 0:
            rk.release()
            raise ConfigurationError('RKNN config failed')

        ret = rk.load_onnx(model=str(model_path))
        if ret != 0:
            rk.release()
            raise ModelLoadError(f'Failed to load ONNX for RKNN: {model_path}')

        ret = rk.build(do_quantization=False)
        if ret != 0:
            rk.release()
            raise ModelLoadError('RKNN build failed')

        ret = rk.init_runtime()
        if ret != 0:
            rk.release()
            raise InferenceError('RKNN init runtime failed')

        # 推理
        inp = preprocess_rknn_sim(img_path, target_size=416)
        outputs = rk.inference(inputs=[inp], data_format='nhwc')

        rk.release()
        return outputs[0]
    except (ConfigurationError, ModelLoadError, InferenceError):
        raise  # Re-raise custom exceptions
    except Exception as e:
        rk.release()
        raise InferenceError(f"RKNN inference failed for {img_path}: {e}") from e

def compare_outputs(onnx_out, rknn_out):
    """计算输出差异"""
    abs_diff = np.abs(onnx_out - rknn_out)

    stats = {
        'max_abs_diff': float(abs_diff.max()),
        'mean_abs_diff': float(abs_diff.mean()),
        'median_abs_diff': float(np.median(abs_diff)),
        'onnx_range': [float(onnx_out.min()), float(onnx_out.max())],
        'rknn_range': [float(rknn_out.min()), float(rknn_out.max())],
        'onnx_mean': float(onnx_out.mean()),
        'rknn_mean': float(rknn_out.mean()),
    }

    # 计算相对误差
    mask = np.abs(onnx_out) > 1e-6  # 避免除0
    if mask.sum() > 0:
        rel_diff = abs_diff[mask] / (np.abs(onnx_out[mask]) + 1e-10)
        stats['mean_rel_diff'] = float(rel_diff.mean())
        stats['max_rel_diff'] = float(rel_diff.max())

    return stats

def main():
    from apps.config import PathConfig
    from apps.utils.paths import resolve_path

    # Use PathConfig instead of hardcoded paths
    onnx_model = resolve_path(PathConfig.YOLO11N_ONNX_416)

    # Validate model file exists
    if not onnx_model.exists():
        raise ModelLoadError(f"Model file not found: {onnx_model}")

    # Validate calibration directory exists
    calib_dir = resolve_path(PathConfig.COCO_CALIB_DIR)
    if not calib_dir.exists():
        raise ConfigurationError(f"Calibration directory not found: {calib_dir}")

    test_images = list(calib_dir.glob('*.jpg'))[:20]  # 测试20张

    if not test_images:
        raise ValidationError(f"No test images found in {calib_dir}")

    logger.info(f'Testing {len(test_images)} images for ONNX vs RKNN accuracy comparison\n')

    all_stats = []
    for i, img_path in enumerate(test_images, 1):
        logger.info(f'[{i}/{len(test_images)}] Processing {img_path.name}...')

        try:
            # 运行推理
            onnx_out = run_onnx_inference(onnx_model, img_path)
            rknn_out = run_rknn_inference(onnx_model, img_path)

            # 对比
            stats = compare_outputs(onnx_out, rknn_out)
            stats['image'] = img_path.name
            all_stats.append(stats)

            logger.info(f"  Max diff: {stats['max_abs_diff']:.6f}, Mean diff: {stats['mean_abs_diff']:.6f}")
        except Exception as e:
            logger.warning(f"  Failed to process {img_path.name}: {e}")

    if not all_stats:
        raise ValidationError("No successful comparisons - all images failed processing")

    # 汇总统计
    logger.info('\n=== Summary Statistics ===')
    max_diffs = [s['max_abs_diff'] for s in all_stats]
    mean_diffs = [s['mean_abs_diff'] for s in all_stats]

    summary = {
        'num_images': len(all_stats),
        'num_failed': len(test_images) - len(all_stats),
        'max_abs_diff': {
            'min': min(max_diffs),
            'max': max(max_diffs),
            'mean': float(np.mean(max_diffs)),
            'median': float(np.median(max_diffs)),
        },
        'mean_abs_diff': {
            'min': min(mean_diffs),
            'max': max(mean_diffs),
            'mean': float(np.mean(mean_diffs)),
            'median': float(np.median(mean_diffs)),
        }
    }

    logger.info(json.dumps(summary, indent=2))

    # 保存详细结果 - Use PathConfig
    from apps.utils.paths import get_artifact_path
    output_file = get_artifact_path('onnx_rknn_comparison.json')
    try:
        with open(output_file, 'w') as f:
            json.dump({'summary': summary, 'details': all_stats}, f, indent=2)
    except Exception as e:
        raise ConfigurationError(f"Failed to write comparison results: {e}") from e

    logger.info(f'\nDetailed results saved to: {output_file}')
    logger.info('ONNX vs RKNN comparison completed successfully!')
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ModelLoadError, ConfigurationError, InferenceError, ValidationError) as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
