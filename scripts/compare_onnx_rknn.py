#!/usr/bin/env python3
"""
ONNX vs RKNN 精度对拍脚本
对比ONNX和RKNN模拟器的推理输出差异
"""
import onnxruntime as ort
from rknn.api import RKNN
import cv2
import numpy as np
from pathlib import Path
import json

def preprocess_for_onnx(img_path):
    """ONNX预处理: NCHW, float32, 0-255"""
    img = cv2.imread(str(img_path))
    inp = cv2.resize(img, (640, 640))
    inp = inp[..., ::-1]  # BGR -> RGB
    inp = inp.transpose(2, 0, 1)  # HWC -> CHW
    inp = np.expand_dims(inp, axis=0)  # (3,640,640) -> (1,3,640,640)
    return inp.astype(np.float32)

def preprocess_for_rknn(img_path):
    """RKNN预处理: NHWC, float32"""
    img = cv2.imread(str(img_path))
    inp = cv2.resize(img, (640, 640))
    inp = inp[..., ::-1]  # BGR -> RGB
    inp = np.expand_dims(inp, axis=0)  # (640,640,3) -> (1,640,640,3)
    return inp.astype(np.float32)

def run_onnx_inference(model_path, img_path):
    """ONNX推理"""
    sess = ort.InferenceSession(str(model_path))
    inp = preprocess_for_onnx(img_path)
    outputs = sess.run(None, {sess.get_inputs()[0].name: inp})
    return outputs[0]

def run_rknn_inference(model_path, img_path):
    """RKNN PC模拟器推理"""
    rk = RKNN()

    # 配置并加载ONNX（用于模拟）
    rk.config(target_platform='rk3588', optimization_level=3)
    rk.load_onnx(model=str(model_path))
    rk.build(do_quantization=False)
    rk.init_runtime()

    # 推理
    inp = preprocess_for_rknn(img_path)
    outputs = rk.inference(inputs=[inp], data_format='nhwc')

    rk.release()
    return outputs[0]

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
    onnx_model = Path('yolo11n.onnx')
    test_images = list(Path('datasets/coco/calib_images').glob('*.jpg'))[:20]  # 测试20张

    print(f'Testing {len(test_images)} images...\n')

    all_stats = []
    for i, img_path in enumerate(test_images, 1):
        print(f'[{i}/{len(test_images)}] Processing {img_path.name}...')

        # 运行推理
        onnx_out = run_onnx_inference(onnx_model, img_path)
        rknn_out = run_rknn_inference(onnx_model, img_path)

        # 对比
        stats = compare_outputs(onnx_out, rknn_out)
        stats['image'] = img_path.name
        all_stats.append(stats)

        print(f"  Max diff: {stats['max_abs_diff']:.6f}, Mean diff: {stats['mean_abs_diff']:.6f}")

    # 汇总统计
    print('\n=== Summary Statistics ===')
    max_diffs = [s['max_abs_diff'] for s in all_stats]
    mean_diffs = [s['mean_abs_diff'] for s in all_stats]

    summary = {
        'num_images': len(test_images),
        'max_abs_diff': {
            'min': min(max_diffs),
            'max': max(max_diffs),
            'mean': np.mean(max_diffs),
            'median': np.median(max_diffs),
        },
        'mean_abs_diff': {
            'min': min(mean_diffs),
            'max': max(mean_diffs),
            'mean': np.mean(mean_diffs),
            'median': np.median(mean_diffs),
        }
    }

    print(json.dumps(summary, indent=2))

    # 保存详细结果
    output_file = Path('artifacts/onnx_rknn_comparison.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({'summary': summary, 'details': all_stats}, f, indent=2)

    print(f'\nDetailed results saved to: {output_file}')

if __name__ == '__main__':
    main()
