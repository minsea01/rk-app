#!/usr/bin/env python3
"""Pick 300 person images from COCO val2017 for RKNN calibration"""
import json
import random
import shutil
import os
from pathlib import Path

# Load annotations
ann_file = Path('annotations/instances_val2017.json')
print(f"Loading {ann_file}...")
with open(ann_file) as f:
    ann = json.load(f)

# Person class ID is 1 in COCO
person_id = 1
has_person = {a["image_id"] for a in ann["annotations"] if a["category_id"] == person_id}
images = [i for i in ann["images"] if i["id"] in has_person]

print(f"Found {len(images)} images with person annotations")

# Pick 300 random images
random.seed(42)
num_calib = min(300, len(images))
pick = random.sample(images, num_calib)

# Copy to calib_images directory
calib_dir = Path('calib_images')
calib_dir.mkdir(exist_ok=True)

print(f"Copying {num_calib} images to {calib_dir}...")
for im in pick:
    src = Path('val2017') / im["file_name"]
    dst = calib_dir / im["file_name"]
    if src.exists():
        shutil.copy(src, dst)
    else:
        print(f"Warning: {src} not found")

print(f"Done! {num_calib} images ready for RKNN calibration")
