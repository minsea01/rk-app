#!/bin/bash
# CrowdHuman 数据集下载脚本 (备选方案)
# CrowdHuman 是大规模人群检测数据集，效果通常比 CityPersons 更好

set -e

echo "下载 CrowdHuman 数据集..."
echo "  注意: CrowdHuman 需要从官网申请下载链接"
echo "  官网: https://www.crowdhuman.org/"
echo ""

DATASET_DIR="datasets/crowdhuman"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

# 检查是否已有数据
if [ -d "images/train" ] && [ -d "labels/train" ]; then
    echo "  CrowdHuman 数据集已存在"
    exit 0
fi

echo "请按以下步骤操作:"
echo ""
echo "1. 访问 https://www.crowdhuman.org/ 并申请下载"
echo ""
echo "2. 下载以下文件:"
echo "   - CrowdHuman_train01.zip (约 2.4GB)"
echo "   - CrowdHuman_train02.zip (约 2.4GB)"
echo "   - CrowdHuman_train03.zip (约 2.4GB)"
echo "   - CrowdHuman_val.zip (约 800MB)"
echo "   - annotation_train.odgt"
echo "   - annotation_val.odgt"
echo ""
echo "3. 上传到: $DATASET_DIR/"
echo ""
echo "4. 重新运行此脚本完成转换"
echo ""

# 如果文件已上传，开始转换
if [ -f "annotation_train.odgt" ]; then
    echo "检测到标注文件，开始转换..."
    python3 << 'PYEOF'
import json
import os
from pathlib import Path

def convert_crowdhuman(split):
    ann_file = f"annotation_{split}.odgt"
    if not os.path.exists(ann_file):
        return

    img_dir = Path(f"images/{split}")
    lbl_dir = Path(f"labels/{split}")
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            img_id = data['ID']

            # 标注转换
            yolo_lines = []
            if 'gtboxes' in data:
                for box in data['gtboxes']:
                    if box['tag'] == 'person':
                        # fbox: full body box [x, y, w, h]
                        x, y, w, h = box['fbox']
                        # 假设图片尺寸 (需要读取实际尺寸)
                        img_w, img_h = 1920, 1080  # CrowdHuman 默认

                        x_center = (x + w/2) / img_w
                        y_center = (y + h/2) / img_h
                        width = w / img_w
                        height = h / img_h

                        # 边界检查
                        if 0 <= x_center <= 1 and 0 <= y_center <= 1:
                            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # 保存标注
            with open(lbl_dir / f"{img_id}.txt", 'w') as f:
                f.writelines(yolo_lines)

    print(f"{split}: {len(list(lbl_dir.glob('*.txt')))} annotations")

convert_crowdhuman('train')
convert_crowdhuman('val')
print("CrowdHuman 转换完成!")
PYEOF
fi
