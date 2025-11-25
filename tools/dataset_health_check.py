#!/usr/bin/env python3
"""
YOLO数据集质量体检工具
诊断"召回爆表、精度偏低"问题的数据源头
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import yaml
import json
from datetime import datetime

def check_dataset_health(dataset_yaml):
    """完整的数据集健康检查"""
    
    print("🏥 YOLO数据集体检开始...")
    
    # 读取数据集配置
    with open(dataset_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    class_names = config['names']
    num_classes = config['nc']
    
    print(f"📊 数据集: {dataset_path}")
    print(f"🏷️ 类别数: {num_classes}")
    print(f"📝 类别: {class_names}")
    
    results = {}
    
    for split in ['train', 'val', 'test']:
        if split in config:
            print(f"\n🔍 检查 {split} 数据集...")
            
            img_dir = dataset_path / config[split].replace('/images', '').replace('images/', '') / 'images'
            label_dir = dataset_path / config[split].replace('/images', '').replace('images/', '') / 'labels'
            
            split_results = check_split_data(img_dir, label_dir, class_names, split)
            results[split] = split_results
    
    # 生成报告
    generate_health_report(results, dataset_path)
    
    return results

def check_split_data(img_dir, label_dir, class_names, split_name):
    """检查单个数据分割"""
    
    if not img_dir.exists() or not label_dir.exists():
        print(f"❌ {split_name} 目录不存在")
        return {}
    
    # 获取所有图像文件
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        img_files.extend(glob.glob(str(img_dir / ext)))
    
    label_files = list(label_dir.glob('*.txt'))
    
    print(f"📁 图像文件: {len(img_files)}")
    print(f"📄 标签文件: {len(label_files)}")
    
    results = {
        'total_images': len(img_files),
        'total_labels': len(label_files),
        'issues': []
    }
    
    # 1. 检查空标签文件
    empty_labels = []
    for label_file in label_files:
        if os.path.getsize(label_file) == 0:
            empty_labels.append(label_file)
    
    if empty_labels:
        results['issues'].append(f"空标签文件: {len(empty_labels)}个")
        print(f"⚠️ 发现 {len(empty_labels)} 个空标签文件")
        for empty_file in empty_labels[:5]:  # 只显示前5个
            print(f"   - {empty_file}")
    
    # 2. 检查图像-标签对应关系
    img_stems = {Path(f).stem for f in img_files}
    label_stems = {f.stem for f in label_files}
    
    missing_labels = img_stems - label_stems
    missing_images = label_stems - img_stems
    
    if missing_labels:
        results['issues'].append(f"缺失标签: {len(missing_labels)}个")
        print(f"❌ {len(missing_labels)} 个图像缺失标签")
        for missing in list(missing_labels)[:5]:
            print(f"   - {missing}")
    
    if missing_images:
        results['issues'].append(f"缺失图像: {len(missing_images)}个")
        print(f"❌ {len(missing_images)} 个标签缺失图像")
    
    # 3. 类别分布统计
    class_counts = Counter()
    bbox_counts = Counter()  # 每个图像的目标数量统计
    small_objects = 0
    large_objects = 0
    
    valid_label_files = [f for f in label_files if os.path.getsize(f) > 0]
    
    for label_file in valid_label_files:
        bbox_count = 0
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            
                            # 检查是否在合法范围内
                            if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                                if class_id < len(class_names):
                                    class_counts[class_id] += 1
                                    bbox_count += 1
                                    
                                    # 统计目标大小（假设图像640x640）
                                    area = w * h
                                    if area < 0.01:  # 小于1%的图像面积
                                        small_objects += 1
                                    elif area > 0.3:  # 大于30%的图像面积
                                        large_objects += 1
                                else:
                                    results['issues'].append(f"非法类别ID: {class_id}")
                            else:
                                results['issues'].append(f"边界框越界: {label_file}")
        
        except Exception as e:
            results['issues'].append(f"标签文件解析错误: {label_file} - {str(e)}")
        
        bbox_counts[bbox_count] += 1
    
    # 统计结果
    results['class_distribution'] = dict(class_counts)
    results['bbox_per_image'] = dict(bbox_counts)
    results['small_objects'] = small_objects
    results['large_objects'] = large_objects
    
    # 4. 类别不平衡检查
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / max(min_count, 1)
        
        if imbalance_ratio > 10:
            results['issues'].append(f"严重类别不平衡: {imbalance_ratio:.1f}:1")
            print(f"⚠️ 类别不平衡严重: {imbalance_ratio:.1f}:1")
    
    print(f"📊 类别分布:")
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"   {class_name}: {count}")
    
    print(f"🎯 小目标: {small_objects}, 大目标: {large_objects}")
    print(f"📦 平均每图目标数: {sum(class_counts.values()) / max(len(valid_label_files), 1):.2f}")
    
    return results

def generate_health_report(results, dataset_path):
    """生成数据集健康报告"""
    
    report_path = dataset_path / 'dataset_health_report.json'
    
    # 总结问题
    all_issues = []
    for split, split_data in results.items():
        if 'issues' in split_data:
            for issue in split_data['issues']:
                all_issues.append(f"{split}: {issue}")
    
    # 分析召回率高、精确度低的可能原因
    diagnosis = []
    
    # 检查类别分布
    for split, split_data in results.items():
        if 'class_distribution' in split_data:
            class_dist = split_data['class_distribution']
            if len(class_dist) > 0:
                max_count = max(class_dist.values())
                min_count = min(class_dist.values())
                if max_count / max(min_count, 1) > 20:
                    diagnosis.append(f"❗ {split}集类别极度不平衡，可能导致模型偏向多数类")
                
                # 检查是否有太多小目标
                small_ratio = split_data.get('small_objects', 0) / max(sum(class_dist.values()), 1)
                if small_ratio > 0.5:
                    diagnosis.append(f"❗ {split}集小目标过多({small_ratio:.1%})，建议提高分辨率")
    
    # 检查标签质量
    total_empty = sum(1 for split_data in results.values() 
                     for issue in split_data.get('issues', []) 
                     if '空标签' in issue)
    
    if total_empty > 0:
        diagnosis.append(f"❗ 发现{total_empty}个数据分割有空标签，会导致FP增加")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_issues': len(all_issues),
            'diagnosis': diagnosis
        },
        'detailed_results': results,
        'issues': all_issues,
        'recommendations': generate_recommendations(results)
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 健康报告已生成: {report_path}")
    
    # 打印诊断结果
    if diagnosis:
        print("\n🔍 问题诊断:")
        for d in diagnosis:
            print(f"   {d}")
    else:
        print("\n✅ 数据集健康状况良好")

def generate_recommendations(results):
    """生成修复建议"""
    
    recommendations = []
    
    # 基于发现的问题给出建议
    for split, split_data in results.items():
        issues = split_data.get('issues', [])
        
        for issue in issues:
            if '空标签' in issue:
                recommendations.append("删除空标签文件及对应图像")
            elif '类别不平衡' in issue:
                recommendations.append("对少数类进行数据增强或重采样")
            elif '缺失' in issue:
                recommendations.append("修复图像-标签对应关系")
            elif '边界框越界' in issue:
                recommendations.append("修正标签文件中的坐标错误")
    
    # 通用建议
    recommendations.extend([
        "使用更高分辨率训练(imgsz=960)",
        "增加focal loss抑制易样本(fl_gamma=1.5)",
        "开启强数据增强(mosaic=1.0, mixup=0.1)",
        "调整conf阈值进行推理(0.35-0.5)"
    ])
    
    return list(set(recommendations))  # 去重

def visualize_sample_annotations(dataset_yaml, num_samples=5):
    """可视化样本标注质量"""
    
    with open(dataset_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    
    # 从训练集选择样本
    img_dir = dataset_path / 'train' / 'images'
    label_dir = dataset_path / 'train' / 'labels'
    
    img_files = list(img_dir.glob('*.jpg'))[:num_samples]
    
    fig, axes = plt.subplots(1, len(img_files), figsize=(4*len(img_files), 4))
    if len(img_files) == 1:
        axes = [axes]
    
    for i, img_file in enumerate(img_files):
        # 读取图像
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 读取标签
        label_file = label_dir / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x_center, y_center, width, height = map(float, parts[:5])
                        
                        # 转换为像素坐标
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # 画边界框
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img, f"C{int(class_id)}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(dataset_path / 'sample_annotations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📸 样本标注可视化已保存: {dataset_path}/sample_annotations.png")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='YOLO数据集健康检查')
    parser.add_argument('--data', type=str, required=True, help='数据集YAML文件路径')
    parser.add_argument('--visualize', action='store_true', help='生成标注可视化')
    parser.add_argument('--samples', type=int, default=5, help='可视化样本数量')
    
    args = parser.parse_args()
    
    # 执行健康检查
    results = check_dataset_health(args.data)
    
    # 可视化（可选）
    if args.visualize:
        try:
            visualize_sample_annotations(args.data, args.samples)
        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")
    
    print("\n🎉 数据集体检完成！")

if __name__ == "__main__":
    main()