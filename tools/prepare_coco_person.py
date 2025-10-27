#!/usr/bin/env python3
"""
COCO Person数据集准备工具
功能：
1. 下载COCO 2017数据集（可选）
2. 筛选包含person类别的图像
3. 转换为YOLO格式（只保留person类）
4. 生成训练/验证集划分
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


def filter_person_annotations(coco_json_path, output_dir, split='train'):
    """筛选person类别并转换为YOLO格式"""

    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # COCO person类别ID为1
    PERSON_CATEGORY_ID = 1

    # 创建输出目录
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images' / split
    labels_dir = output_dir / 'labels' / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 建立图像ID到文件名的映射
    image_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}

    # 收集每张图像的person标注
    image_annotations = defaultdict(list)
    person_count = 0

    for ann in coco['annotations']:
        if ann['category_id'] == PERSON_CATEGORY_ID:
            image_annotations[ann['image_id']].append(ann)
            person_count += 1

    print(f"Found {len(image_annotations)} images with person annotations")
    print(f"Total {person_count} person instances")

    # 转换为YOLO格式
    converted_count = 0
    image_list = []

    for image_id, annotations in image_annotations.items():
        if image_id not in image_id_to_file:
            continue

        filename = image_id_to_file[image_id]
        width, height = image_id_to_size[image_id]

        # 生成YOLO标注文件
        label_filename = Path(filename).stem + '.txt'
        label_path = labels_dir / label_filename

        yolo_annotations = []
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]

            # 转换为YOLO格式 [class_id, x_center, y_center, w, h] (归一化)
            x_center = (bbox[0] + bbox[2] / 2) / width
            y_center = (bbox[1] + bbox[3] / 2) / height
            w = bbox[2] / width
            h = bbox[3] / height

            # class_id=0 (person是唯一类别)
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # 写入标注文件
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

        # 记录图像路径（相对路径）
        image_list.append(filename)
        converted_count += 1

    print(f"Converted {converted_count} images to YOLO format")

    # 保存图像列表
    image_list_file = output_dir / f'{split}_images.txt'
    with open(image_list_file, 'w') as f:
        f.write('\n'.join(image_list))

    return converted_count


def create_data_yaml(output_dir, train_count, val_count):
    """生成data.yaml配置文件"""

    output_dir = Path(output_dir)

    data_yaml = f"""# COCO Person数据集配置
path: {output_dir.absolute()}

# 训练/验证集路径（相对于path）
train: images/train
val: images/val

# 类别数量
nc: 1

# 类别名称
names:
  0: person

# 数据集统计
# 训练集: {train_count} 张图像
# 验证集: {val_count} 张图像
"""

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)

    print(f"\nDataset configuration saved to: {yaml_path}")
    print(f"Train: {train_count} images")
    print(f"Val: {val_count} images")


def copy_images_from_list(image_list_file, src_dir, dst_dir):
    """根据列表复制图像文件"""

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with open(image_list_file, 'r') as f:
        image_files = [line.strip() for line in f if line.strip()]

    print(f"\nCopying {len(image_files)} images...")
    copied = 0

    for img_file in image_files:
        src = src_dir / img_file
        dst = dst_dir / img_file

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
            if copied % 1000 == 0:
                print(f"  Copied {copied}/{len(image_files)} images...")
        else:
            print(f"Warning: {src} not found")

    print(f"Copied {copied} images to {dst_dir}")
    return copied


def main():
    parser = argparse.ArgumentParser(description='Prepare COCO Person dataset for YOLO training')
    parser.add_argument('--coco-dir', type=str, required=True,
                        help='Path to COCO dataset directory')
    parser.add_argument('--output-dir', type=str, default='datasets/coco_person',
                        help='Output directory for processed dataset')
    parser.add_argument('--copy-images', action='store_true',
                        help='Copy image files to output directory (otherwise only labels)')

    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)

    # 检查COCO数据集目录
    train_json = coco_dir / 'annotations' / 'instances_train2017.json'
    val_json = coco_dir / 'annotations' / 'instances_val2017.json'

    if not train_json.exists():
        print(f"Error: {train_json} not found")
        print("\nPlease download COCO 2017 dataset:")
        print("  wget http://images.cocodataset.org/zips/train2017.zip")
        print("  wget http://images.cocodataset.org/zips/val2017.zip")
        print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        return

    print("=" * 60)
    print("COCO Person Dataset Preparation")
    print("=" * 60)

    # 处理训练集
    print("\n[1/4] Processing training set...")
    train_count = filter_person_annotations(train_json, output_dir, 'train')

    # 处理验证集
    print("\n[2/4] Processing validation set...")
    val_count = filter_person_annotations(val_json, output_dir, 'val')

    # 生成data.yaml
    print("\n[3/4] Creating data.yaml...")
    create_data_yaml(output_dir, train_count, val_count)

    # 可选：复制图像文件
    if args.copy_images:
        print("\n[4/4] Copying images...")
        train_img_dir = coco_dir / 'train2017'
        val_img_dir = coco_dir / 'val2017'

        copy_images_from_list(
            output_dir / 'train_images.txt',
            train_img_dir,
            output_dir / 'images' / 'train'
        )

        copy_images_from_list(
            output_dir / 'val_images.txt',
            val_img_dir,
            output_dir / 'images' / 'val'
        )
    else:
        print("\n[4/4] Skipping image copy (use --copy-images to copy)")
        print("NOTE: Make sure to update image paths in data.yaml or create symlinks")

    print("\n" + "=" * 60)
    print("✓ Dataset preparation completed!")
    print("=" * 60)
    print(f"\nDataset location: {output_dir.absolute()}")
    print(f"Config file: {output_dir / 'data.yaml'}")
    print("\nNext steps:")
    print("  1. Review data.yaml configuration")
    print("  2. Start training with: python tools/train_yolov8.py --data datasets/coco_person/data.yaml")


if __name__ == '__main__':
    main()
