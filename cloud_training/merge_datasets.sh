#!/bin/bash
# CrowdHuman + COCO Person 数据集合并脚本
# 用于 AutoDL 云端环境
# 合并后预计: 80k+ 训练图片，可达 90%+ mAP

set -e

echo "========================================="
echo "CrowdHuman + COCO Person 数据集合并"
echo "========================================="

WORK_DIR="${WORK_DIR:-/root/autodl-tmp/pedestrian_training}"
cd "$WORK_DIR"

# 目标目录
MERGED_DIR="datasets/merged"
mkdir -p "$MERGED_DIR"/{train,val}/{images,labels}

echo ""
echo "[1/4] 检测可用数据集..."

# 检测 COCO Person
COCO_TRAIN=""
COCO_VAL=""
if [ -d "datasets/coco_person/train" ]; then
    COCO_TRAIN="datasets/coco_person/train"
    COCO_VAL="datasets/coco_person/val"
    echo "  ✅ COCO Person: $COCO_TRAIN"
elif [ -d "/root/autodl-tmp/coco/train2017" ]; then
    echo "  ⚠️  检测到完整COCO，需要先筛选person类"
    echo "     运行: python3 scripts/filter_coco_person.py"
fi

# 检测 CrowdHuman
CROWD_TRAIN=""
CROWD_VAL=""
if [ -d "datasets/crowdhuman/train" ]; then
    CROWD_TRAIN="datasets/crowdhuman/train"
    CROWD_VAL="datasets/crowdhuman/val"
    echo "  ✅ CrowdHuman: $CROWD_TRAIN"
else
    echo "  ⚠️  CrowdHuman未找到"
fi

# 检查是否有足够数据
if [ -z "$COCO_TRAIN" ] && [ -z "$CROWD_TRAIN" ]; then
    echo "❌ 未找到任何数据集"
    exit 1
fi

echo ""
echo "[2/4] 创建符号链接合并数据集..."

# 合并训练集
link_count=0

if [ -n "$COCO_TRAIN" ]; then
    echo "  链接COCO Person训练集..."
    for img in "$COCO_TRAIN/images"/*.jpg; do
        [ -f "$img" ] || continue
        base=$(basename "$img")
        ln -sf "$(realpath "$img")" "$MERGED_DIR/train/images/coco_$base" 2>/dev/null || true

        # 对应label
        label="${img%.jpg}.txt"
        label="${label/images/labels}"
        if [ -f "$label" ]; then
            ln -sf "$(realpath "$label")" "$MERGED_DIR/train/labels/coco_${base%.jpg}.txt" 2>/dev/null || true
        fi
        ((link_count++))
    done
    echo "    COCO: $link_count 张"
fi

if [ -n "$CROWD_TRAIN" ]; then
    crowd_count=0
    echo "  链接CrowdHuman训练集..."
    for img in "$CROWD_TRAIN/images"/*.jpg; do
        [ -f "$img" ] || continue
        base=$(basename "$img")
        ln -sf "$(realpath "$img")" "$MERGED_DIR/train/images/crowd_$base" 2>/dev/null || true

        label="${img%.jpg}.txt"
        label="${label/images/labels}"
        if [ -f "$label" ]; then
            ln -sf "$(realpath "$label")" "$MERGED_DIR/train/labels/crowd_${base%.jpg}.txt" 2>/dev/null || true
        fi
        ((crowd_count++))
    done
    echo "    CrowdHuman: $crowd_count 张"
    link_count=$((link_count + crowd_count))
fi

echo ""
echo "[3/4] 合并验证集..."

val_count=0

if [ -n "$COCO_VAL" ]; then
    for img in "$COCO_VAL/images"/*.jpg; do
        [ -f "$img" ] || continue
        base=$(basename "$img")
        ln -sf "$(realpath "$img")" "$MERGED_DIR/val/images/coco_$base" 2>/dev/null || true

        label="${img%.jpg}.txt"
        label="${label/images/labels}"
        if [ -f "$label" ]; then
            ln -sf "$(realpath "$label")" "$MERGED_DIR/val/labels/coco_${base%.jpg}.txt" 2>/dev/null || true
        fi
        ((val_count++))
    done
fi

if [ -n "$CROWD_VAL" ]; then
    for img in "$CROWD_VAL/images"/*.jpg; do
        [ -f "$img" ] || continue
        base=$(basename "$img")
        ln -sf "$(realpath "$img")" "$MERGED_DIR/val/images/crowd_$base" 2>/dev/null || true

        label="${img%.jpg}.txt"
        label="${label/images/labels}"
        if [ -f "$label" ]; then
            ln -sf "$(realpath "$label")" "$MERGED_DIR/val/labels/crowd_${base%.jpg}.txt" 2>/dev/null || true
        fi
        ((val_count++))
    done
fi

echo "  验证集: $val_count 张"

echo ""
echo "[4/4] 创建配置文件..."

# 创建YAML配置
cat > "$MERGED_DIR/merged.yaml" << EOF
# CrowdHuman + COCO Person 联合数据集
# 用于训练高精度行人检测模型 (目标 90%+ mAP)

path: $WORK_DIR/$MERGED_DIR
train: train/images
val: val/images

# 单类别: 行人
names:
  0: person

# 数据集统计
# - 训练集: ~80k 图片 (COCO ~64k + CrowdHuman ~15k)
# - 验证集: ~7k 图片
EOF

echo ""
echo "========================================="
echo "数据集合并完成!"
echo ""
echo "统计:"
echo "  训练集: $link_count 张"
echo "  验证集: $val_count 张"
echo ""
echo "配置文件: $MERGED_DIR/merged.yaml"
echo ""
echo "下一步: 运行训练"
echo "  ./train_90map_optimized.sh"
echo "========================================="
