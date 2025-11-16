#!/usr/bin/env python3
"""
从 COCO 数据集提取行人 (person) 子集
用于训练专门的行人检测模型
"""
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def prepare_person_dataset():
    """提取 COCO 中的行人数据"""

    # 路径配置
    coco_root = Path("datasets/coco")
    output_root = Path("datasets/coco_person")

    # 创建输出目录
    (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 处理训练集和验证集
    for split in ["train", "val"]:
        print(f"\n处理 {split} 集...")

        # 加载 COCO 标注
        ann_file = coco_root / "annotations" / f"instances_{split}2017.json"
        with open(ann_file) as f:
            coco = json.load(f)

        # 找出所有包含 person 的图像
        person_id = 1  # person 类别 ID
        person_anns = [a for a in coco["annotations"] if a["category_id"] == person_id]
        person_image_ids = set(a["image_id"] for a in person_anns)

        print(f"找到 {len(person_image_ids)} 张包含行人的图像")

        # 创建图像ID到文件名的映射
        id_to_image = {img["id"]: img for img in coco["images"]}

        # 处理每张图像
        copied = 0
        for img_id in tqdm(person_image_ids, desc=f"处理 {split}"):
            img_info = id_to_image[img_id]
            img_filename = img_info["file_name"]

            # 复制图像
            src_img = coco_root / f"{split}2017" / img_filename
            dst_img = output_root / "images" / split / img_filename

            if src_img.exists():
                shutil.copy(src_img, dst_img)

                # 生成 YOLO 格式标注
                img_width = img_info["width"]
                img_height = img_info["height"]

                # 获取该图像的所有 person 标注
                img_anns = [a for a in person_anns if a["image_id"] == img_id]

                # 写入 YOLO 格式标签文件
                label_file = output_root / "labels" / split / img_filename.replace(".jpg", ".txt")
                with open(label_file, "w") as f:
                    for ann in img_anns:
                        # COCO bbox: [x, y, width, height]
                        x, y, w, h = ann["bbox"]

                        # 转换为 YOLO 格式: [class, x_center, y_center, width, height] (归一化)
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        w_norm = w / img_width
                        h_norm = h / img_height

                        # class_id = 0 (只有一个类别: person)
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                copied += 1

        print(f"✅ {split} 集: 复制了 {copied} 张图像和标注")

    # 创建 data.yaml 配置文件
    data_yaml = output_root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"""# COCO Person Dataset
path: {output_root.absolute()}
train: images/train
val: images/val

# Classes
nc: 1
names: ['person']
""")

    print(f"\n✅ 数据集准备完成!")
    print(f"   路径: {output_root.absolute()}")
    print(f"   配置: {data_yaml}")

    # 统计信息
    train_images = len(list((output_root / "images" / "train").glob("*.jpg")))
    val_images = len(list((output_root / "images" / "val").glob("*.jpg")))
    train_labels = len(list((output_root / "labels" / "train").glob("*.txt")))
    val_labels = len(list((output_root / "labels" / "val").glob("*.txt")))

    print(f"\n数据集统计:")
    print(f"  训练集: {train_images} 张图像, {train_labels} 个标注文件")
    print(f"  验证集: {val_images} 张图像, {val_labels} 个标注文件")

if __name__ == "__main__":
    prepare_person_dataset()
