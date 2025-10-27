#!/usr/bin/env python3
"""
从YOLO数据集生成RKNN量化校准集
从训练集中随机抽取N张图像，生成绝对路径列表
"""

import argparse
import random
import yaml
from pathlib import Path


def load_yaml(yaml_path):
    """加载YOLO data.yaml"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_image_paths(data_config):
    """获取训练集图像路径"""
    dataset_root = Path(data_config['path'])
    train_dir = dataset_root / data_config['train']

    # 支持的图像格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

    image_paths = []
    for ext in extensions:
        image_paths.extend(train_dir.glob(ext))

    return [p.absolute() for p in image_paths]


def sample_images(image_paths, num_samples):
    """随机采样图像"""
    if len(image_paths) < num_samples:
        print(f"Warning: 可用图像 ({len(image_paths)}) 少于请求数量 ({num_samples})")
        return image_paths

    return random.sample(image_paths, num_samples)


def main():
    parser = argparse.ArgumentParser(description='Generate RKNN calibration dataset')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to YOLO data.yaml')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for calibration dataset')
    parser.add_argument('--num', type=int, default=300,
                        help='Number of images to sample (default: 300)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 加载数据集配置
    print(f"Loading dataset config from: {args.data}")
    data_config = load_yaml(args.data)

    # 获取图像路径
    print("Scanning training images...")
    image_paths = get_image_paths(data_config)
    print(f"Found {len(image_paths)} training images")

    # 采样
    print(f"Sampling {args.num} images...")
    sampled_paths = sample_images(image_paths, args.num)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写入校准列表（绝对路径）
    calib_file = output_dir / 'calib.txt'
    with open(calib_file, 'w') as f:
        for path in sampled_paths:
            f.write(f"{path}\n")

    print(f"\n✓ Calibration dataset generated!")
    print(f"  Output: {calib_file}")
    print(f"  Images: {len(sampled_paths)}")

    # 验证路径格式
    with open(calib_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('/'):
            print(f"  Format: Absolute paths ✓")
        else:
            print(f"  Warning: Paths are not absolute!")


if __name__ == '__main__':
    main()
