#!/bin/bash
# WiderPerson 数据集下载脚本
# WiderPerson 是专门的行人检测数据集，比 CityPersons 更容易下载

set -e

echo "下载 WiderPerson 数据集..."

DATASET_DIR="datasets/widerperson"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

# WiderPerson 数据集 (约 1.5GB)
# 官方地址需要申请，使用镜像
if [ ! -f "WiderPerson.zip" ]; then
    echo "  从镜像下载 WiderPerson..."

    # 尝试多个源
    # 源1: 学术镜像
    wget -q --show-progress -O WiderPerson.zip \
        "https://github.com/GreenTeaHua/WiderPerson/archive/refs/heads/master.zip" 2>/dev/null || \
    # 源2: HuggingFace
    wget -q --show-progress -O WiderPerson.zip \
        "https://huggingface.co/datasets/wider-person/resolve/main/WiderPerson.zip" 2>/dev/null || \
    {
        echo ""
        echo "  自动下载失败，请手动下载:"
        echo "  1. 访问 http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/"
        echo "  2. 下载 WiderPerson.zip"
        echo "  3. 上传到 $DATASET_DIR/"
        echo ""
        echo "  或者使用备用数据集 CrowdHuman:"
        echo "  ./download_crowdhuman.sh"
        exit 1
    }
fi

echo "  解压数据集..."
unzip -q WiderPerson.zip

echo "  转换为 YOLO 格式..."
cd ..
python3 << 'PYEOF'
import os
import shutil
from pathlib import Path

# WiderPerson annotations format:
# <class_label> <x1> <y1> <x2> <y2>
# class 1 = pedestrians

src_dir = Path("widerperson")
out_dir = Path("widerperson_yolo")

for split in ["train", "val"]:
    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # 读取图片列表
    list_file = src_dir / f"{split}.txt"
    if not list_file.exists():
        continue

    with open(list_file) as f:
        image_names = [l.strip() for l in f if l.strip()]

    for img_name in image_names:
        # 复制图片
        src_img = src_dir / "Images" / f"{img_name}.jpg"
        if src_img.exists():
            shutil.copy(src_img, img_out / f"{img_name}.jpg")

        # 转换标注
        src_ann = src_dir / "Annotations" / f"{img_name}.txt"
        if src_ann.exists():
            with open(src_ann) as f:
                lines = f.readlines()

            # 获取图片尺寸 (假设 1024x768，实际需要读取)
            img_w, img_h = 1024, 768

            yolo_lines = []
            for line in lines[1:]:  # 跳过第一行(数量)
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x1, y1, x2, y2 = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if cls == 1:  # pedestrian
                        # 转换为 YOLO 格式
                        x_center = (x1 + x2) / 2 / img_w
                        y_center = (y1 + y2) / 2 / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            with open(lbl_out / f"{img_name}.txt", 'w') as f:
                f.writelines(yolo_lines)

print("WiderPerson 转换完成!")
print(f"  Train: {len(list((out_dir/'images'/'train').glob('*.jpg')))} images")
print(f"  Val: {len(list((out_dir/'images'/'val').glob('*.jpg')))} images")
PYEOF

# 更新路径
mv widerperson_yolo widerperson 2>/dev/null || true

echo "WiderPerson 数据集准备完成!"
