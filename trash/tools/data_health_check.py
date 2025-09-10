#!/usr/bin/env python3
"""
YOLO数据集健康检查脚本
专门针对"高召回低精度"问题的数据质量诊断

用法:
    python tools/data_health_check.py --data /path/to/data.yaml
    python tools/data_health_check.py --data /home/minsea01/datasets/industrial_15_classes_ready/data.yaml

输出:
    1. 数据质量报告 (data_health_report.txt)
    2. 类别分布可视化 (class_distribution.png) 
    3. 样本可视化 (sample_visualization.png)
    4. 问题文件列表 (problem_files.txt)
"""

import os
import sys
import yaml
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from collections import Counter, defaultdict
import json

class DataHealthChecker:
    def __init__(self, data_yaml_path):
        self.data_yaml_path = Path(data_yaml_path)
        self.load_config()
        self.issues = []
        self.stats = {}
        
    def load_config(self):
        """加载数据集配置"""
        with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.dataset_path = Path(self.config['path'])
        self.num_classes = self.config['nc']
        self.class_names = self.config.get('names', [f'class_{i}' for i in range(self.num_classes)])
        
        print(f"📊 数据集路径: {self.dataset_path}")
        print(f"📊 类别数量: {self.num_classes}")
        print(f"📊 类别名称: {self.class_names}")
        
    def check_empty_labels(self, split='train'):
        """检查空标签文件"""
        print(f"\n🔍 检查 {split} 集空标签文件...")
        
        label_dir = self.dataset_path / split.replace('/images', '/labels')
        if not label_dir.exists():
            label_dir = self.dataset_path / f'{split}/labels'
            
        empty_files = []
        if label_dir.exists():
            for label_file in label_dir.glob('*.txt'):
                if label_file.stat().st_size == 0:
                    empty_files.append(str(label_file))
                    
        print(f"❌ 发现 {len(empty_files)} 个空标签文件")
        if empty_files:
            print("   示例:", empty_files[:5])
            self.issues.append(f"{split} 集有 {len(empty_files)} 个空标签文件")
            
        return empty_files
    
    def check_missing_pairs(self, split='train'):
        """检查图像和标签配对问题"""
        print(f"\n🔍 检查 {split} 集图像-标签配对...")
        
        # 获取图像和标签目录
        img_dir = self.dataset_path / self.config[split]
        label_dir = self.dataset_path / self.config[split].replace('/images', '/labels')
        
        if not label_dir.exists():
            label_dir = self.dataset_path / f'{split}/labels'
            
        # 获取文件名（不含扩展名）
        img_files = set()
        label_files = set()
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_file in img_dir.glob(ext):
                img_files.add(img_file.stem)
                
        for label_file in label_dir.glob('*.txt'):
            label_files.add(label_file.stem)
            
        # 找出不匹配的文件
        missing_labels = img_files - label_files
        missing_images = label_files - img_files
        
        print(f"📷 图像文件: {len(img_files)}")
        print(f"🏷️  标签文件: {len(label_files)}")
        print(f"❌ 缺少标签: {len(missing_labels)}")
        print(f"❌ 缺少图像: {len(missing_images)}")
        
        if missing_labels:
            print("   缺少标签的图像示例:", list(missing_labels)[:5])
            self.issues.append(f"{split} 集有 {len(missing_labels)} 个图像缺少标签")
            
        if missing_images:
            print("   缺少图像的标签示例:", list(missing_images)[:5])
            self.issues.append(f"{split} 集有 {len(missing_images)} 个标签缺少图像")
            
        return missing_labels, missing_images
    
    def analyze_class_distribution(self, split='train'):
        """分析类别分布"""
        print(f"\n🔍 分析 {split} 集类别分布...")
        
        label_dir = self.dataset_path / self.config[split].replace('/images', '/labels')
        if not label_dir.exists():
            label_dir = self.dataset_path / f'{split}/labels'
            
        class_counts = Counter()
        invalid_classes = []
        total_instances = 0
        
        for label_file in label_dir.glob('*.txt'):
            if label_file.stat().st_size == 0:
                continue
                
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            if 0 <= class_id < self.num_classes:
                                class_counts[class_id] += 1
                                total_instances += 1
                            else:
                                invalid_classes.append((str(label_file), line_num, class_id))
                        except ValueError:
                            invalid_classes.append((str(label_file), line_num, parts[0]))
                            
        print(f"📊 总实例数: {total_instances}")
        print(f"❌ 无效类别: {len(invalid_classes)}")
        
        if invalid_classes:
            print("   无效类别示例:", invalid_classes[:5])
            self.issues.append(f"{split} 集有 {len(invalid_classes)} 个无效类别")
            
        # 分析类别不均衡
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"📊 类别不均衡比例: {imbalance_ratio:.2f}")
            if imbalance_ratio > 10:
                self.issues.append(f"{split} 集类别严重不均衡 (比例: {imbalance_ratio:.2f})")
                
        self.stats[f'{split}_class_distribution'] = dict(class_counts)
        return class_counts, invalid_classes
    
    def check_annotation_quality(self, split='train', sample_count=50):
        """检查标注质量（边界框合理性）"""
        print(f"\n🔍 检查 {split} 集标注质量...")
        
        img_dir = self.dataset_path / self.config[split]
        label_dir = self.dataset_path / self.config[split].replace('/images', '/labels')
        if not label_dir.exists():
            label_dir = self.dataset_path / f'{split}/labels'
            
        problematic_annotations = []
        very_small_boxes = []
        out_of_bounds = []
        
        # 随机采样检查
        label_files = list(label_dir.glob('*.txt'))[:sample_count]
        
        for label_file in label_files:
            if label_file.stat().st_size == 0:
                continue
                
            # 找对应的图像文件
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_img = img_dir / (label_file.stem + ext)
                if potential_img.exists():
                    img_file = potential_img
                    break
                    
            if not img_file:
                continue
                
            # 读取图像尺寸
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # 检查边界
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                   0 < width <= 1 and 0 < height <= 1):
                                out_of_bounds.append((str(label_file), line_num))
                                
                            # 检查是否过小
                            pixel_width = width * w
                            pixel_height = height * h
                            if pixel_width < 8 or pixel_height < 8:
                                very_small_boxes.append((str(label_file), line_num, pixel_width, pixel_height))
                                
                        except ValueError:
                            problematic_annotations.append((str(label_file), line_num))
                            
        print(f"❌ 问题标注: {len(problematic_annotations)}")
        print(f"❌ 越界标注: {len(out_of_bounds)}")
        print(f"⚠️  极小目标: {len(very_small_boxes)}")
        
        if very_small_boxes:
            print("   极小目标示例:", very_small_boxes[:5])
            
        if len(very_small_boxes) > sample_count * 0.1:
            self.issues.append(f"{split} 集有大量极小目标 ({len(very_small_boxes)} 个)")
            
        return problematic_annotations, out_of_bounds, very_small_boxes
    
    def visualize_samples(self, split='train', num_samples=8, output_path='sample_visualization.png'):
        """可视化样本和标注"""
        print(f"\n🖼️ 生成 {split} 集可视化样本...")
        
        img_dir = self.dataset_path / self.config[split]
        label_dir = self.dataset_path / self.config[split].replace('/images', '/labels')
        if not label_dir.exists():
            label_dir = self.dataset_path / f'{split}/labels'
            
        # 随机选择有标注的图像
        valid_samples = []
        for label_file in label_dir.glob('*.txt'):
            if label_file.stat().st_size > 0:
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_file = img_dir / (label_file.stem + ext)
                    if img_file.exists():
                        valid_samples.append((img_file, label_file))
                        break
                        
        if len(valid_samples) < num_samples:
            num_samples = len(valid_samples)
            
        samples = np.random.choice(len(valid_samples), num_samples, replace=False)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        
        for i, idx in enumerate(samples):
            if i >= num_samples:
                break
                
            img_file, label_file = valid_samples[idx]
            
            # 读取图像
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            axes[i].imshow(img)
            axes[i].set_title(f"{img_file.name}")
            axes[i].axis('off')
            
            # 读取标注
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # 转换为像素坐标
                            x = (x_center - width/2) * w
                            y = (y_center - height/2) * h
                            w_box = width * w
                            h_box = height * h
                            
                            # 绘制边界框
                            rect = patches.Rectangle((x, y), w_box, h_box, 
                                                   linewidth=2, edgecolor=colors[class_id], 
                                                   facecolor='none')
                            axes[i].add_patch(rect)
                            
                            # 添加类别标签
                            axes[i].text(x, y-5, self.class_names[class_id], 
                                       color=colors[class_id], fontsize=8, 
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                        except (ValueError, IndexError):
                            continue
                            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📸 样本可视化已保存: {output_path}")
    
    def plot_class_distribution(self, output_path='class_distribution.png'):
        """绘制类别分布图"""
        train_dist = self.stats.get('train_class_distribution', {})
        val_dist = self.stats.get('val_class_distribution', {})
        
        if not train_dist:
            print("⚠️ 没有训练集类别分布数据")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 训练集分布
        classes = list(range(self.num_classes))
        train_counts = [train_dist.get(i, 0) for i in classes]
        
        ax1.bar(classes, train_counts, color='skyblue', alpha=0.8)
        ax1.set_title('训练集类别分布')
        ax1.set_xlabel('类别ID')
        ax1.set_ylabel('实例数量')
        ax1.set_xticks(classes)
        ax1.set_xticklabels([self.class_names[i] for i in classes], rotation=45, ha='right')
        
        # 在柱子上添加数值
        for i, count in enumerate(train_counts):
            if count > 0:
                ax1.text(i, count + max(train_counts) * 0.01, str(count), 
                        ha='center', va='bottom', fontsize=8)
        
        # 验证集分布（如果有）
        if val_dist:
            val_counts = [val_dist.get(i, 0) for i in classes]
            ax2.bar(classes, val_counts, color='lightcoral', alpha=0.8)
            ax2.set_title('验证集类别分布')
            ax2.set_xlabel('类别ID')
            ax2.set_ylabel('实例数量')
            ax2.set_xticks(classes)
            ax2.set_xticklabels([self.class_names[i] for i in classes], rotation=45, ha='right')
            
            for i, count in enumerate(val_counts):
                if count > 0:
                    ax2.text(i, count + max(val_counts) * 0.01, str(count), 
                            ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, '无验证集数据', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('验证集类别分布')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 类别分布图已保存: {output_path}")
    
    def generate_report(self, output_path='data_health_report.txt'):
        """生成综合报告"""
        print(f"\n📝 生成数据健康报告...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("YOLO数据集健康检查报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"数据集路径: {self.dataset_path}\n")
            f.write(f"类别数量: {self.num_classes}\n")
            f.write(f"类别名称: {', '.join(self.class_names)}\n\n")
            
            f.write("🔍 发现的问题:\n")
            f.write("-" * 40 + "\n")
            if self.issues:
                for i, issue in enumerate(self.issues, 1):
                    f.write(f"{i}. {issue}\n")
            else:
                f.write("✅ 未发现明显问题\n")
            f.write("\n")
            
            f.write("📊 统计信息:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.stats.items():
                if 'class_distribution' in key:
                    split = key.replace('_class_distribution', '')
                    total = sum(value.values())
                    f.write(f"{split} 集总实例数: {total}\n")
                    
                    # 找出最多和最少的类别
                    if value:
                        max_class = max(value, key=value.get)
                        min_class = min(value, key=value.get)
                        f.write(f"  最多类别: {self.class_names[max_class]} ({value[max_class]} 个)\n")
                        f.write(f"  最少类别: {self.class_names[min_class]} ({value[min_class]} 个)\n")
                        f.write(f"  不均衡比例: {value[max_class] / value[min_class]:.2f}\n")
            f.write("\n")
            
            f.write("💡 建议措施:\n")
            f.write("-" * 40 + "\n")
            
            # 基于发现的问题给出建议
            if any('空标签' in issue for issue in self.issues):
                f.write("• 删除或重新标注空标签文件\n")
            if any('缺少标签' in issue for issue in self.issues):
                f.write("• 为缺少标签的图像补充标注\n")
            if any('无效类别' in issue for issue in self.issues):
                f.write("• 检查并修正无效的类别ID\n")
            if any('不均衡' in issue for issue in self.issues):
                f.write("• 考虑过采样少数类或使用focal loss\n")
            if any('极小目标' in issue for issue in self.issues):
                f.write("• 提高输入分辨率 (imgsz=960 或 1280)\n")
                f.write("• 启用小目标友好的数据增强 (mosaic, copy_paste)\n")
                
            f.write("\n建议的训练参数调整:\n")
            f.write("• 提高分辨率: imgsz=960\n")
            f.write("• 增加epochs: 150-200\n") 
            f.write("• 使用focal loss: fl_gamma=1.5\n")
            f.write("• 启用多尺度训练: multi_scale=True\n")
            f.write("• 调整NMS阈值: iou=0.5-0.6\n")
            f.write("• 部署时使用更高置信度: conf=0.35-0.5\n")
            
        print(f"📋 报告已保存: {output_path}")
    
    def run_full_check(self):
        """运行完整的数据健康检查"""
        print("🚀 开始YOLO数据集健康检查...")
        print("=" * 60)
        
        # 检查各个分割
        for split in ['train', 'val']:
            if split in self.config:
                print(f"\n📂 检查 {split} 集...")
                self.check_empty_labels(split)
                self.check_missing_pairs(split)
                self.analyze_class_distribution(split)
                self.check_annotation_quality(split)
        
        # 生成可视化
        if 'train' in self.config:
            self.visualize_samples('train')
        self.plot_class_distribution()
        
        # 生成报告
        self.generate_report()
        
        print("\n✅ 数据健康检查完成！")
        print("📁 输出文件:")
        print("  - data_health_report.txt (综合报告)")
        print("  - class_distribution.png (类别分布图)")
        print("  - sample_visualization.png (样本可视化)")
        
        # 总结关键问题
        if self.issues:
            print(f"\n⚠️ 发现 {len(self.issues)} 个潜在问题，详见报告。")
            print("🎯 关键建议: 先修复标签质量问题，再调整训练参数。")
        else:
            print("\n🎉 数据质量良好！可以专注于模型调优。")

def main():
    parser = argparse.ArgumentParser(description='YOLO数据集健康检查')
    parser.add_argument('--data', required=True, help='数据集YAML配置文件路径')
    parser.add_argument('--output-dir', default='.', help='输出目录')
    
    args = parser.parse_args()
    
    # 切换到输出目录
    os.chdir(args.output_dir)
    
    # 运行检查
    checker = DataHealthChecker(args.data)
    checker.run_full_check()

if __name__ == '__main__':
    main()
