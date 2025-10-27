#!/usr/bin/env python3
"""
YOLO模型评估脚本
生成PR曲线、混淆矩阵、预测样例等诊断图表

用法:
    python tools/model_evaluation.py --model runs/train/exp/weights/best.pt --data /path/to/data.yaml
    python tools/model_evaluation.py --model runs/train/industrial_15cls_test5/weights/best.pt --data industrial_dataset/data.yaml

输出:
    1. 详细评估报告 (evaluation_report.txt)
    2. PR曲线图 (pr_curves.png)
    3. 混淆矩阵 (confusion_matrix.png)
    4. 预测样例对比 (prediction_samples.png)
    5. 置信度分布 (confidence_distribution.png)
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import json
from collections import defaultdict
import seaborn as sns
from ultralytics import YOLO
import torch

class ModelEvaluator:
    def __init__(self, model_path, data_yaml_path, conf_threshold=0.25, iou_threshold=0.6):
        self.model_path = Path(model_path)
        self.data_yaml_path = Path(data_yaml_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.load_model()
        self.load_config()
        
    def load_model(self):
        """加载YOLO模型"""
        print(f"🤖 加载模型: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print(f"✅ 模型加载成功")
        
    def load_config(self):
        """加载数据集配置"""
        with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.dataset_path = Path(self.config['path'])
        self.num_classes = self.config['nc']
        self.class_names = self.config.get('names', [f'class_{i}' for i in range(self.num_classes)])
        
        print(f"📊 数据集: {self.dataset_path}")
        print(f"📊 类别数: {self.num_classes}")
        print(f"📊 类别: {self.class_names}")
        
    def run_validation(self, split='val'):
        """运行验证并获取详细结果"""
        print(f"\n🔬 运行 {split} 集验证...")
        
        # 使用ultralytics内置验证
        results = self.model.val(
            data=str(self.data_yaml_path),
            split=split,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            plots=False,  # 我们自己生成图表
            save_json=True,
            device='0' if torch.cuda.is_available() else 'cpu'
        )
        
        self.val_results = results
        return results
    
    def analyze_predictions(self, split='val', max_samples=100):
        """分析预测结果"""
        print(f"\n🔍 分析 {split} 集预测结果...")
        
        img_dir = self.dataset_path / self.config[split]
        
        # 获取图像文件列表
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_files.extend(list(img_dir.glob(ext)))
            
        if len(img_files) > max_samples:
            img_files = img_files[:max_samples]
            
        predictions = []
        confidence_scores = []
        
        for img_file in img_files:
            # 预测
            results = self.model(str(img_file), conf=self.conf_threshold, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        confidence_scores.append(conf)
                        predictions.append({
                            'image': str(img_file),
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': int(cls),
                            'class_name': self.class_names[int(cls)]
                        })
        
        self.predictions = predictions
        self.confidence_scores = confidence_scores
        
        print(f"📊 总预测数: {len(predictions)}")
        print(f"📊 平均置信度: {np.mean(confidence_scores):.3f}")
        print(f"📊 置信度范围: {np.min(confidence_scores):.3f} - {np.max(confidence_scores):.3f}")
        
        return predictions, confidence_scores
    
    def plot_pr_curves(self, output_path='pr_curves.png'):
        """绘制PR曲线"""
        print(f"\n📈 生成PR曲线...")
        
        if not hasattr(self.val_results, 'curves'):
            print("⚠️ 无法获取PR曲线数据")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 整体PR曲线
        if hasattr(self.val_results, 'pr_curve'):
            axes[0, 0].plot(self.val_results.pr_curve[0], self.val_results.pr_curve[1])
            axes[0, 0].set_xlabel('Recall')
            axes[0, 0].set_ylabel('Precision')
            axes[0, 0].set_title('Overall PR Curve')
            axes[0, 0].grid(True, alpha=0.3)
            
        # 各类别PR曲线（如果有详细数据）
        axes[0, 1].text(0.5, 0.5, 'Per-Class PR Curves\n(需要详细验证数据)', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Per-Class PR Curves')
        
        # mAP@0.5曲线
        if hasattr(self.val_results, 'maps'):
            maps = self.val_results.maps
            axes[1, 0].bar(range(len(maps)), maps, color='skyblue')
            axes[1, 0].set_xlabel('Class')
            axes[1, 0].set_ylabel('mAP@0.5')
            axes[1, 0].set_title('mAP@0.5 per Class')
            axes[1, 0].set_xticks(range(len(self.class_names)))
            axes[1, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
            
        # 关键指标汇总
        metrics_text = f"""
关键指标:
mAP@0.5: {self.val_results.box.map50:.3f}
mAP@0.5:0.95: {self.val_results.box.map:.3f}
Precision: {self.val_results.box.mp:.3f}
Recall: {self.val_results.box.mr:.3f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center', fontfamily='monospace')
        axes[1, 1].set_title('Key Metrics Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 PR曲线已保存: {output_path}")
    
    def plot_confusion_matrix(self, output_path='confusion_matrix.png'):
        """绘制混淆矩阵"""
        print(f"\n🔄 生成混淆矩阵...")
        
        if hasattr(self.val_results, 'confusion_matrix') and self.val_results.confusion_matrix is not None:
            cm = self.val_results.confusion_matrix.matrix
            
            plt.figure(figsize=(12, 10))
            
            # 归一化混淆矩阵
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
            
            # 绘制热力图
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names + ['Background'],
                       yticklabels=self.class_names + ['Background'])
            
            plt.title('Normalized Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"🔄 混淆矩阵已保存: {output_path}")
        else:
            print("⚠️ 无法获取混淆矩阵数据")
    
    def plot_confidence_distribution(self, output_path='confidence_distribution.png'):
        """绘制置信度分布"""
        print(f"\n📊 生成置信度分布...")
        
        if not hasattr(self, 'confidence_scores'):
            print("⚠️ 请先运行analyze_predictions")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 整体置信度分布
        ax1.hist(self.confidence_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(self.confidence_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.confidence_scores):.3f}')
        ax1.axvline(self.conf_threshold, color='orange', linestyle='--', 
                   label=f'Threshold: {self.conf_threshold}')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 各类别置信度分布
        class_confidences = defaultdict(list)
        for pred in self.predictions:
            class_confidences[pred['class_name']].append(pred['confidence'])
            
        # 绘制箱线图
        if class_confidences:
            class_names = list(class_confidences.keys())
            conf_data = [class_confidences[name] for name in class_names]
            
            ax2.boxplot(conf_data, labels=class_names)
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence Distribution by Class')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 置信度分布已保存: {output_path}")
    
    def visualize_prediction_samples(self, split='val', num_samples=8, 
                                   output_path='prediction_samples.png'):
        """可视化预测样例"""
        print(f"\n🖼️ 生成预测样例可视化...")
        
        img_dir = self.dataset_path / self.config[split]
        label_dir = self.dataset_path / self.config[split].replace('/images', '/labels')
        if not label_dir.exists():
            label_dir = self.dataset_path / f'{split}/labels'
            
        # 随机选择图像
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_files.extend(list(img_dir.glob(ext)))
            
        if len(img_files) < num_samples:
            num_samples = len(img_files)
            
        selected_files = np.random.choice(img_files, num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        
        for i, img_file in enumerate(selected_files):
            if i >= num_samples:
                break
                
            # 读取图像
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            axes[i].imshow(img)
            axes[i].set_title(f"{img_file.name}")
            axes[i].axis('off')
            
            # 绘制GT标注（绿色）
            label_file = label_dir / (img_file.stem + '.txt')
            if label_file.exists() and label_file.stat().st_size > 0:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                x = (x_center - width/2) * w
                                y = (y_center - height/2) * h
                                w_box = width * w
                                h_box = height * h
                                
                                rect = patches.Rectangle((x, y), w_box, h_box, 
                                                       linewidth=2, edgecolor='green', 
                                                       facecolor='none', linestyle='-')
                                axes[i].add_patch(rect)
                                
                                axes[i].text(x, y-15, f'GT: {self.class_names[class_id]}', 
                                           color='green', fontsize=8, 
                                           bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor='white', alpha=0.7))
                            except (ValueError, IndexError):
                                continue
            
            # 绘制预测结果（红色）
            results = self.model(str(img_file), conf=self.conf_threshold, verbose=False)
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        class_id = int(cls)
                        
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                               linewidth=2, edgecolor='red', 
                                               facecolor='none', linestyle='--')
                        axes[i].add_patch(rect)
                        
                        axes[i].text(x1, y1-5, f'Pred: {self.class_names[class_id]} ({conf:.2f})', 
                                   color='red', fontsize=8, 
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor='white', alpha=0.7))
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Ground Truth'),
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Prediction')
        ]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🖼️ 预测样例已保存: {output_path}")
    
    def analyze_failure_cases(self):
        """分析失效案例"""
        print(f"\n🔍 分析失效案例...")
        
        # 简化的失效分析
        low_conf_predictions = [p for p in self.predictions if p['confidence'] < 0.5]
        high_conf_predictions = [p for p in self.predictions if p['confidence'] > 0.8]
        
        # 类别分布分析
        class_counts = defaultdict(int)
        low_conf_class_counts = defaultdict(int)
        
        for pred in self.predictions:
            class_counts[pred['class_name']] += 1
            
        for pred in low_conf_predictions:
            low_conf_class_counts[pred['class_name']] += 1
            
        failure_analysis = {
            'total_predictions': len(self.predictions),
            'low_confidence_count': len(low_conf_predictions),
            'high_confidence_count': len(high_conf_predictions),
            'low_confidence_ratio': len(low_conf_predictions) / len(self.predictions) if self.predictions else 0,
            'class_performance': {}
        }
        
        for class_name in class_counts:
            total = class_counts[class_name]
            low_conf = low_conf_class_counts.get(class_name, 0)
            failure_analysis['class_performance'][class_name] = {
                'total': total,
                'low_confidence': low_conf,
                'failure_rate': low_conf / total if total > 0 else 0
            }
        
        self.failure_analysis = failure_analysis
        return failure_analysis
    
    def generate_evaluation_report(self, output_path='evaluation_report.txt'):
        """生成评估报告"""
        print(f"\n📝 生成评估报告...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("YOLO模型评估报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"数据集: {self.dataset_path}\n")
            f.write(f"置信度阈值: {self.conf_threshold}\n")
            f.write(f"IoU阈值: {self.iou_threshold}\n\n")
            
            # 关键指标
            f.write("📊 关键指标:\n")
            f.write("-" * 40 + "\n")
            f.write(f"mAP@0.5: {self.val_results.box.map50:.3f}\n")
            f.write(f"mAP@0.5:0.95: {self.val_results.box.map:.3f}\n")
            f.write(f"Precision: {self.val_results.box.mp:.3f}\n")
            f.write(f"Recall: {self.val_results.box.mr:.3f}\n")
            f.write(f"F1-Score: {2 * self.val_results.box.mp * self.val_results.box.mr / (self.val_results.box.mp + self.val_results.box.mr):.3f}\n\n")
            
            # 预测统计
            if hasattr(self, 'predictions'):
                f.write("🔍 预测统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"总预测数: {len(self.predictions)}\n")
                f.write(f"平均置信度: {np.mean(self.confidence_scores):.3f}\n")
                f.write(f"置信度标准差: {np.std(self.confidence_scores):.3f}\n")
                f.write(f"最低置信度: {np.min(self.confidence_scores):.3f}\n")
                f.write(f"最高置信度: {np.max(self.confidence_scores):.3f}\n\n")
            
            # 失效分析
            if hasattr(self, 'failure_analysis'):
                fa = self.failure_analysis
                f.write("⚠️ 失效分析:\n")
                f.write("-" * 40 + "\n")
                f.write(f"低置信度预测比例: {fa['low_confidence_ratio']:.3f}\n")
                f.write(f"高置信度预测数: {fa['high_confidence_count']}\n")
                f.write(f"低置信度预测数: {fa['low_confidence_count']}\n\n")
                
                f.write("各类别失效率:\n")
                for class_name, perf in fa['class_performance'].items():
                    f.write(f"  {class_name}: {perf['failure_rate']:.3f} ({perf['low_confidence']}/{perf['total']})\n")
                f.write("\n")
            
            # 诊断结论
            f.write("🎯 诊断结论:\n")
            f.write("-" * 40 + "\n")
            
            # 基于指标判断问题类型
            precision = self.val_results.box.mp
            recall = self.val_results.box.mr
            
            if recall > 0.85 and precision < 0.65:
                f.write("❌ 典型的'高召回低精度'问题\n")
                f.write("可能原因:\n")
                f.write("  • 标签质量问题（漏标、错标）\n")
                f.write("  • 目标过小或密集，难以区分\n")
                f.write("  • 置信度阈值过低\n")
                f.write("  • 类别不均衡严重\n\n")
                
                f.write("建议措施:\n")
                f.write("  • 运行数据健康检查脚本验证标签质量\n")
                f.write("  • 提高输入分辨率 (imgsz=960)\n")
                f.write("  • 使用focal loss处理类别不均衡\n")
                f.write("  • 部署时提高置信度阈值至0.4-0.5\n")
                f.write("  • 考虑使用更强的数据增强\n")
                
            elif precision > 0.8 and recall < 0.6:
                f.write("❌ '高精度低召回'问题\n")
                f.write("可能原因:\n")
                f.write("  • 模型过于保守，漏检严重\n")
                f.write("  • 数据增强过强\n")
                f.write("  • 学习率过小或训练不充分\n\n")
                
                f.write("建议措施:\n")
                f.write("  • 降低置信度阈值\n")
                f.write("  • 增加训练轮数\n")
                f.write("  • 调整损失权重，增加recall权重\n")
                
            elif precision > 0.7 and recall > 0.7:
                f.write("✅ 模型性能良好\n")
                f.write("可以考虑:\n")
                f.write("  • 进一步优化超参数\n")
                f.write("  • 针对特定场景微调\n")
                f.write("  • 模型压缩和加速\n")
                
            else:
                f.write("❌ 整体性能偏低\n")
                f.write("建议措施:\n")
                f.write("  • 检查数据质量和标签一致性\n")
                f.write("  • 尝试更大的模型 (yolov8m)\n")
                f.write("  • 增加训练时间和数据量\n")
                f.write("  • 调整学习率和优化器设置\n")
            
        print(f"📋 评估报告已保存: {output_path}")
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("🚀 开始YOLO模型深度评估...")
        print("=" * 60)
        
        # 运行验证
        self.run_validation()
        
        # 分析预测
        self.analyze_predictions()
        
        # 失效分析
        self.analyze_failure_cases()
        
        # 生成图表
        self.plot_pr_curves()
        self.plot_confusion_matrix()
        self.plot_confidence_distribution()
        self.visualize_prediction_samples()
        
        # 生成报告
        self.generate_evaluation_report()
        
        print("\n✅ 模型评估完成！")
        print("📁 输出文件:")
        print("  - evaluation_report.txt (评估报告)")
        print("  - pr_curves.png (PR曲线)")
        print("  - confusion_matrix.png (混淆矩阵)")
        print("  - confidence_distribution.png (置信度分布)")
        print("  - prediction_samples.png (预测样例)")
        
        # 快速诊断
        precision = self.val_results.box.mp
        recall = self.val_results.box.mr
        
        print(f"\n🎯 快速诊断:")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        
        if recall > 0.85 and precision < 0.65:
            print("   ❌ 确认'高召回低精度'问题！")
            print("   💡 优先检查数据标签质量")
        else:
            print("   ✅ 指标相对均衡")

def main():
    parser = argparse.ArgumentParser(description='YOLO模型深度评估')
    parser.add_argument('--model', required=True, help='模型权重文件路径')
    parser.add_argument('--data', required=True, help='数据集YAML配置文件')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU阈值')
    parser.add_argument('--output-dir', default='.', help='输出目录')
    
    args = parser.parse_args()
    
    # 切换到输出目录
    os.chdir(args.output_dir)
    
    # 运行评估
    evaluator = ModelEvaluator(args.model, args.data, args.conf, args.iou)
    evaluator.run_full_evaluation()

if __name__ == '__main__':
    main()
