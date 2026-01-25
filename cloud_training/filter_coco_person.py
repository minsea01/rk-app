#!/usr/bin/env python3
"""
从完整COCO数据集中筛选Person类图片
用于行人检测训练

使用方法:
    python3 filter_coco_person.py --coco-root /path/to/coco --output datasets/coco_person
"""

import os
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def filter_coco_person(coco_root: str, output_dir: str, copy_mode: bool = False):
    """
    筛选COCO数据集中包含person类的图片

    Args:
        coco_root: COCO数据集根目录
        output_dir: 输出目录
        copy_mode: True=复制文件, False=创建符号链接(节省空间)
    """
    coco_root = Path(coco_root)
    output_dir = Path(output_dir)

    # COCO person类ID = 0 (在YOLO标签中)
    PERSON_CLASS = 0

    for split in ['train2017', 'val2017']:
        images_dir = coco_root / 'images' / split
        labels_dir = coco_root / 'labels' / split

        if not images_dir.exists():
            print(f"跳过 {split}: 图片目录不存在")
            continue

        if not labels_dir.exists():
            print(f"跳过 {split}: 标签目录不存在")
            continue

        # 输出目录
        out_split = 'train' if 'train' in split else 'val'
        out_images = output_dir / out_split / 'images'
        out_labels = output_dir / out_split / 'labels'
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        print(f"\n处理 {split}...")

        # 遍历标签文件
        label_files = list(labels_dir.glob('*.txt'))
        total = len(label_files)
        person_count = 0
        skipped = 0

        for i, label_path in enumerate(label_files):
            if (i + 1) % 10000 == 0:
                print(f"  进度: {i+1}/{total}")

            # 读取标签，检查是否包含person类
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                # 筛选person类标注
                person_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id == PERSON_CLASS:
                            # 保持class_id为0
                            person_lines.append(line)

                if not person_lines:
                    skipped += 1
                    continue

                # 找到对应图片
                img_name = label_path.stem + '.jpg'
                img_path = images_dir / img_name

                if not img_path.exists():
                    skipped += 1
                    continue

                # 链接/复制文件
                out_img = out_images / img_name
                out_lbl = out_labels / (label_path.stem + '.txt')

                if copy_mode:
                    shutil.copy2(img_path, out_img)
                else:
                    if not out_img.exists():
                        out_img.symlink_to(img_path.resolve())

                # 写入筛选后的标签
                with open(out_lbl, 'w') as f:
                    f.writelines(person_lines)

                person_count += 1

            except Exception as e:
                skipped += 1
                continue

        print(f"  {split}: {person_count} 张包含行人 (跳过 {skipped})")

    # 创建YAML配置
    yaml_content = f"""# COCO Person 子集
# 从完整COCO中筛选的行人检测数据集

path: {output_dir.resolve()}
train: train/images
val: val/images

names:
  0: person
"""

    yaml_path = output_dir / 'coco_person.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✅ 完成! 配置文件: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='筛选COCO Person类数据')
    parser.add_argument('--coco-root', type=str, default='/root/autodl-tmp/coco',
                        help='COCO数据集根目录')
    parser.add_argument('--output', type=str, default='datasets/coco_person',
                        help='输出目录')
    parser.add_argument('--copy', action='store_true',
                        help='复制文件而非创建符号链接')

    args = parser.parse_args()

    print("========================================")
    print("COCO Person 数据筛选")
    print("========================================")
    print(f"输入: {args.coco_root}")
    print(f"输出: {args.output}")
    print(f"模式: {'复制' if args.copy else '符号链接'}")

    filter_coco_person(args.coco_root, args.output, args.copy)


if __name__ == '__main__':
    main()
