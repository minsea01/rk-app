#!/bin/bash
# 快速数据集问题诊断脚本
# 专门解决"召回爆表、精度偏低"问题

set -e

DATASET_PATH="${1:-/home/minsea01/datasets/industrial_15_classes_ready}"
DATASET_YAML="${DATASET_PATH}/data.yaml"

echo "🔧 快速诊断数据集问题..."
echo "📁 数据集: $DATASET_PATH"

cd $DATASET_PATH

echo ""
echo "1️⃣ 空标签文件检查:"
for split in train val test; do
    if [ -d "$split/labels" ]; then
        empty_count=$(find $split/labels -name "*.txt" -size 0 | wc -l)
        total_count=$(find $split/labels -name "*.txt" | wc -l)
        echo "   $split: $empty_count/$total_count 空标签"
        
        if [ $empty_count -gt 0 ]; then
            echo "   ❌ 发现空标签文件，这会导致FP增加！"
            find $split/labels -name "*.txt" -size 0 | head -3
        fi
    fi
done

echo ""
echo "2️⃣ 图像-标签对应检查:"
for split in train val test; do
    if [ -d "$split/images" ] && [ -d "$split/labels" ]; then
        img_count=$(find $split/images -name "*.jpg" -o -name "*.png" | wc -l)
        label_count=$(find $split/labels -name "*.txt" | wc -l)
        echo "   $split: $img_count 图像, $label_count 标签"
        
        if [ $img_count -ne $label_count ]; then
            echo "   ❌ 图像标签数量不匹配！"
        fi
    fi
done

echo ""
echo "3️⃣ 类别分布统计:"
for split in train val test; do
    if [ -d "$split/labels" ]; then
        echo "   === $split 类别分布 ==="
        find $split/labels -name "*.txt" -exec cat {} \; | \
        awk '{if(NF>=5) print $1}' | sort -n | uniq -c | \
        awk '{printf "   类别%s: %s个\n", $2, $1}' | head -20
        
        # 检查类别不平衡
        max_count=$(find $split/labels -name "*.txt" -exec cat {} \; | \
                   awk '{if(NF>=5) print $1}' | sort -n | uniq -c | \
                   awk '{print $1}' | sort -nr | head -1)
        min_count=$(find $split/labels -name "*.txt" -exec cat {} \; | \
                   awk '{if(NF>=5) print $1}' | sort -n | uniq -c | \
                   awk '{print $1}' | sort -n | head -1)
        
        if [ ! -z "$max_count" ] && [ ! -z "$min_count" ] && [ $min_count -gt 0 ]; then
            ratio=$((max_count / min_count))
            if [ $ratio -gt 10 ]; then
                echo "   ⚠️ 类别不平衡严重: ${ratio}:1"
            fi
        fi
    fi
done

echo ""
echo "4️⃣ 边界框有效性检查:"
for split in train val test; do
    if [ -d "$split/labels" ]; then
        echo "   检查 $split 边界框..."
        
        # 检查是否有超出[0,1]范围的坐标
        invalid_bbox=$(find $split/labels -name "*.txt" -exec awk '
        {
            if(NF>=5) {
                x=$2; y=$3; w=$4; h=$5
                if(x<0 || x>1 || y<0 || y>1 || w<=0 || w>1 || h<=0 || h>1) {
                    print FILENAME": "$0
                    invalid++
                }
            }
        } 
        END {print "INVALID_COUNT:"invalid}' {} \; | grep "INVALID_COUNT" | awk -F: '{sum+=$2} END {print sum}')
        
        if [ ! -z "$invalid_bbox" ] && [ $invalid_bbox -gt 0 ]; then
            echo "   ❌ 发现 $invalid_bbox 个无效边界框"
        else
            echo "   ✅ 边界框格式正常"
        fi
    fi
done

echo ""
echo "5️⃣ 小目标统计 (面积<1%):"
for split in train val test; do
    if [ -d "$split/labels" ]; then
        small_objects=$(find $split/labels -name "*.txt" -exec awk '
        {
            if(NF>=5) {
                area = $4 * $5
                if(area < 0.01) small++
                total++
            }
        } 
        END {printf "%.1f", (small/total)*100}' {} \; | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
        
        echo "   $split: ${small_objects}% 小目标"
        
        if (( $(echo "$small_objects > 30" | bc -l) )); then
            echo "   ⚠️ 小目标过多，建议提高分辨率到960+"
        fi
    fi
done

echo ""
echo "6️⃣ 推荐修复措施:"
echo "   基于发现的问题，建议："

# 根据检查结果给出针对性建议
has_empty=$(find . -name "*.txt" -size 0 | wc -l)
if [ $has_empty -gt 0 ]; then
    echo "   🔧 删除空标签文件: find . -name '*.txt' -size 0 -delete"
fi

echo "   🔧 数据增强配置:"
echo "      mosaic=1.0 mixup=0.1 copy_paste=0.1"
echo "   🔧 损失函数优化:"
echo "      fl_gamma=1.5 (focal loss抑制易样本)"
echo "   🔧 训练分辨率:"
echo "      imgsz=960 (处理小目标)"
echo "   🔧 推理阈值:"
echo "      conf=0.4 iou=0.6 (减少FP)"

echo ""
echo "7️⃣ 一键修复命令:"
echo "   # 训练优化版本"
echo "   yolo train data=$DATASET_YAML model=yolov8s.pt \\"
echo "     imgsz=960 epochs=150 batch=auto device=0 \\"
echo "     mosaic=1.0 mixup=0.1 copy_paste=0.1 fl_gamma=1.5 \\"
echo "     cos_lr=True lr0=0.005 lrf=0.1 warmup_epochs=5 \\"
echo "     multi_scale=True cache=ram patience=80 \\"
echo "     name=precision_fix_v2"

echo ""
echo "✅ 快速诊断完成！运行完整体检:"
echo "   python tools/dataset_health_check.py --data $DATASET_YAML --visualize"