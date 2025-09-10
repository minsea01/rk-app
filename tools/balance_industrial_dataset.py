#!/usr/bin/env python3
"""
工业数据集平衡工具
平衡各类别样本数量，确保训练效果
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np

def analyze_dataset_distribution(dataset_dir):
    """分析数据集类别分布"""
    label_files = list(Path(dataset_dir).glob("**/*.txt"))
    class_counts = Counter()
    
    for label_file in label_files:
        if label_file.name == "classes.txt":
            continue
            
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    
    return class_counts

def balance_classes(source_dirs, output_dir, min_samples_per_class=300, 
                   train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """平衡类别样本数量"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 收集所有样本
    samples_by_class = defaultdict(list)
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        label_files = list(source_path.glob("**/*.txt"))
        
        for label_file in label_files:
            if label_file.name == "classes.txt":
                continue
                
            # 找到对应的图像文件
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_img = label_file.with_suffix(ext)
                if potential_img.exists():
                    img_file = potential_img
                    break
            
            if img_file and img_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            samples_by_class[class_id].append((img_file, label_file))
                            break  # 只记录第一个类别
    
    print("📊 原始类别分布:")
    for class_id, samples in samples_by_class.items():
        print(f"  Class {class_id}: {len(samples)} samples")
    
    # 平衡类别
    balanced_samples = []
    for class_id, samples in samples_by_class.items():
        if len(samples) < min_samples_per_class:
            # 数据增强到最小样本数
            augmented_samples = augment_samples(samples, min_samples_per_class)
            balanced_samples.extend(augmented_samples)
        else:
            # 随机采样到最小样本数
            random.shuffle(samples)
            balanced_samples.extend(samples[:min_samples_per_class])
    
    # 打乱并分割数据集
    random.shuffle(balanced_samples)
    total_samples = len(balanced_samples)
    
    train_split = int(total_samples * train_ratio)
    val_split = int(total_samples * (train_ratio + val_ratio))
    
    train_samples = balanced_samples[:train_split]
    val_samples = balanced_samples[train_split:val_split]
    test_samples = balanced_samples[val_split:]
    
    # 复制文件
    for split_name, samples in [('train', train_samples), 
                                ('val', val_samples), 
                                ('test', test_samples)]:
        print(f"📁 处理 {split_name} 数据集: {len(samples)} samples")
        
        for i, (img_file, label_file) in enumerate(samples):
            # 复制图像文件
            dst_img = output_path / split_name / 'images' / f"{split_name}_{i:06d}{img_file.suffix}"
            shutil.copy2(img_file, dst_img)
            
            # 复制标签文件
            dst_label = output_path / split_name / 'labels' / f"{split_name}_{i:06d}.txt"
            shutil.copy2(label_file, dst_label)
    
    print("✅ 数据集平衡完成!")
    print(f"📈 训练集: {len(train_samples)} samples")
    print(f"📊 验证集: {len(val_samples)} samples")
    print(f"🧪 测试集: {len(test_samples)} samples")

def augment_samples(samples, target_count):
    """数据增强到目标数量"""
    augmented = list(samples)
    
    while len(augmented) < target_count:
        # 随机选择一个样本进行增强
        img_file, label_file = random.choice(samples)
        
        # 简单的水平翻转增强
        img = cv2.imread(str(img_file))
        if img is not None:
            # 水平翻转
            flipped_img = cv2.flip(img, 1)
            
            # 调整标签中的x坐标
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            adjusted_lines = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x, y, w, h = parts[:5]
                        x_new = 1.0 - float(x)  # 水平翻转x坐标
                        adjusted_lines.append(f"{class_id} {x_new:.6f} {y} {w} {h}\n")
            
            # 创建临时增强文件
            aug_img_path = img_file.parent / f"aug_{len(augmented)}_{img_file.name}"
            aug_label_path = label_file.parent / f"aug_{len(augmented)}_{label_file.name}"
            
            cv2.imwrite(str(aug_img_path), flipped_img)
            with open(aug_label_path, 'w') as f:
                f.writelines(adjusted_lines)
            
            augmented.append((aug_img_path, aug_label_path))
    
    return augmented[:target_count]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="平衡工业数据集")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                       help="输入数据集目录")
    parser.add_argument("--output-dir", required=True,
                       help="输出数据集目录")
    parser.add_argument("--min-samples-per-class", type=int, default=300,
                       help="每类最小样本数")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                       help="测试集比例")
    
    args = parser.parse_args()
    
    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"数据集分割比例总和应为1.0，当前为{total_ratio}")
    
    print("🔧 开始平衡工业数据集...")
    balance_classes(
        source_dirs=args.input_dirs,
        output_dir=args.output_dir,
        min_samples_per_class=args.min_samples_per_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    print("🎉 数据集平衡完成！")