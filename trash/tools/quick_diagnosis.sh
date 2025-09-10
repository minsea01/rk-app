#!/bin/bash

# YOLO训练问题快速诊断脚本
# 一键运行数据体检 + 模型评估

set -e

# 默认配置
DATA_YAML="/home/minsea01/datasets/industrial_15_classes_ready/data.yaml"
MODEL_PATH=""
OUTPUT_DIR="./diagnosis_results"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -d, --data PATH     数据集YAML文件路径 (默认: $DATA_YAML)"
    echo "  -m, --model PATH    模型权重文件路径 (如果不提供，只运行数据检查)"
    echo "  -o, --output DIR    输出目录 (默认: $OUTPUT_DIR)"
    echo "  -h, --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -d /path/to/data.yaml -m runs/train/exp/weights/best.pt"
    echo "  $0 --data /path/to/data.yaml --model runs/train/exp/weights/best.pt --output results"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            DATA_YAML="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必要文件
if [[ ! -f "$DATA_YAML" ]]; then
    print_error "数据集YAML文件不存在: $DATA_YAML"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

print_header "🚀 YOLO训练问题快速诊断"

echo "📋 配置信息:"
echo "   数据集: $DATA_YAML"
echo "   模型: ${MODEL_PATH:-'未指定 (仅数据检查)'}"
echo "   输出目录: $(pwd)"
echo ""

# 1. 数据健康检查
print_header "🔍 步骤1: 数据健康检查"

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "未找到Python解释器"
    exit 1
fi

# 检查依赖
echo "检查Python依赖..."
$PYTHON_CMD -c "import yaml, matplotlib, cv2, numpy" 2>/dev/null || {
    print_warning "缺少必要的Python包，尝试安装..."
    pip install pyyaml matplotlib opencv-python numpy 2>/dev/null || {
        print_error "依赖安装失败，请手动安装: pip install pyyaml matplotlib opencv-python numpy"
        exit 1
    }
}

# 运行数据检查
echo "运行数据健康检查..."
if $PYTHON_CMD ../tools/data_health_check.py --data "$DATA_YAML" --output-dir "." > data_check.log 2>&1; then
    print_success "数据检查完成"
    echo "   📄 日志: data_check.log"
    
    # 快速查看关键问题
    if [[ -f "data_health_report.txt" ]]; then
        echo ""
        echo "🔍 发现的关键问题:"
        grep -A 10 "发现的问题:" data_health_report.txt || echo "   ✅ 未发现明显问题"
    fi
else
    print_error "数据检查失败，查看日志: data_check.log"
fi

# 2. 模型评估 (如果提供了模型路径)
if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -f "$MODEL_PATH" ]]; then
        print_warning "模型文件不存在: $MODEL_PATH，跳过模型评估"
    else
        print_header "📊 步骤2: 模型评估"
        
        # 检查YOLO依赖
        echo "检查YOLO依赖..."
        $PYTHON_CMD -c "import ultralytics" 2>/dev/null || {
            print_warning "缺少ultralytics包，尝试安装..."
            pip install ultralytics 2>/dev/null || {
                print_error "ultralytics安装失败，请手动安装: pip install ultralytics"
                exit 1
            }
        }
        
        echo "运行模型评估..."
        if $PYTHON_CMD ../tools/model_evaluation.py --model "$MODEL_PATH" --data "$DATA_YAML" --output-dir "." > model_eval.log 2>&1; then
            print_success "模型评估完成"
            echo "   📄 日志: model_eval.log"
            
            # 快速查看关键指标
            if [[ -f "evaluation_report.txt" ]]; then
                echo ""
                echo "📊 关键指标:"
                grep -A 5 "关键指标:" evaluation_report.txt || echo "   ⚠️ 无法提取关键指标"
                
                echo ""
                echo "🎯 诊断结论:"
                grep -A 10 "诊断结论:" evaluation_report.txt || echo "   ⚠️ 无法提取诊断结论"
            fi
        else
            print_error "模型评估失败，查看日志: model_eval.log"
        fi
    fi
fi

# 3. 生成改进建议
print_header "💡 步骤3: 改进建议"

cat > improvement_suggestions.md << 'EOF'
# YOLO训练改进建议

## 基于诊断结果的建议

### 🔧 立即行动项

1. **数据质量修复**
   - 查看 `data_health_report.txt` 中的具体问题
   - 删除或重新标注空标签文件
   - 补充缺失的标注
   - 修正无效类别ID

2. **模型性能优化**
   - 查看 `evaluation_report.txt` 中的详细分析
   - 根据PR曲线和混淆矩阵调整策略

### 📈 训练参数调优 (针对高召回低精度问题)

#### 优化配置模板
```yaml
# 针对工业检测优化的配置
imgsz: 960               # 提高分辨率应对小目标
epochs: 200              # 增加训练轮数
batch: auto              # 自动批次大小
patience: 80             # 增加早停耐心

# 损失函数优化
fl_gamma: 1.5            # Focal Loss应对类别不均衡
box: 7.5                 # 提高边界框损失权重
cls: 1.5                 # 提高分类损失权重

# 数据增强 (小目标友好)
mosaic: 1.0              # 启用mosaic
copy_paste: 0.2          # copy-paste增强
mixup: 0.15              # 适量mixup
multi_scale: True        # 多尺度训练

# 学习率调度
cos_lr: True             # 余弦退火
lr0: 0.005               # 较小初始学习率
lrf: 0.05                # 最终学习率比例
warmup_epochs: 5         # 预热轮数

# 缓存和性能
cache: ram               # 内存缓存
workers: 8               # 多进程加载
```

#### 训练命令示例
```bash
yolo detect train \
  data=/path/to/data.yaml \
  model=yolov8s.pt \
  imgsz=960 epochs=200 batch=auto device=0 \
  fl_gamma=1.5 box=7.5 cls=1.5 \
  mosaic=1.0 copy_paste=0.2 mixup=0.15 multi_scale=True \
  cos_lr=True lr0=0.005 lrf=0.05 warmup_epochs=5 \
  cache=ram workers=8 patience=80 \
  project=runs/train name=improved_training
```

### 🎯 部署优化

1. **置信度阈值调整**
   - 训练时使用较低阈值 (0.25)
   - 部署时提高到 0.4-0.5 减少假阳性

2. **NMS参数优化**
   - 密集场景: `iou=0.5`
   - 稀疏场景: `iou=0.6-0.7`

3. **后处理策略**
   - 考虑 per-class NMS
   - 实现置信度自适应阈值

### 📊 持续监控

1. **训练过程监控**
   - 观察loss曲线收敛情况
   - 监控验证集指标变化
   - 注意过拟合信号

2. **定期重新评估**
   - 每次数据更新后重新诊断
   - 定期验证部署效果
   - 收集难例进行针对性优化

EOF

print_success "改进建议已生成: improvement_suggestions.md"

# 4. 总结报告
print_header "📋 诊断总结"

echo "🎯 诊断完成！生成的文件:"
echo ""

if [[ -f "data_health_report.txt" ]]; then
    echo "   📊 data_health_report.txt - 数据质量报告"
fi

if [[ -f "class_distribution.png" ]]; then
    echo "   📈 class_distribution.png - 类别分布图"
fi

if [[ -f "sample_visualization.png" ]]; then
    echo "   🖼️  sample_visualization.png - 样本可视化"
fi

if [[ -f "evaluation_report.txt" ]]; then
    echo "   📊 evaluation_report.txt - 模型评估报告"
fi

if [[ -f "pr_curves.png" ]]; then
    echo "   📈 pr_curves.png - PR曲线"
fi

if [[ -f "confusion_matrix.png" ]]; then
    echo "   🔄 confusion_matrix.png - 混淆矩阵"
fi

if [[ -f "confidence_distribution.png" ]]; then
    echo "   📊 confidence_distribution.png - 置信度分布"
fi

if [[ -f "prediction_samples.png" ]]; then
    echo "   🖼️  prediction_samples.png - 预测样例"
fi

echo "   💡 improvement_suggestions.md - 改进建议"
echo ""

# 如果是高召回低精度问题，给出特别提醒
if [[ -f "evaluation_report.txt" ]] && grep -q "高召回低精度" evaluation_report.txt; then
    print_warning "检测到'高召回低精度'问题！"
    echo "🔥 立即行动:"
    echo "   1. 优先检查数据标签质量"
    echo "   2. 提高训练分辨率到960"
    echo "   3. 使用focal loss处理类别不均衡"
    echo "   4. 部署时提高置信度阈值"
fi

print_success "诊断流程全部完成！查看生成的报告文件了解详细信息。"
