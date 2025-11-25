#!/usr/bin/env python3
"""
Quantization Dataset Preparation Script
Creates representative dataset for INT8 quantization from industrial images
"""

import os
import random
import argparse
from pathlib import Path
import cv2

def collect_images(data_dir, output_file, num_samples=300, min_size=200):
    """Collect representative images for quantization"""
    
    print("🔍 Collecting images for quantization dataset...")
    print(f"   - Source directory: {data_dir}")
    print(f"   - Target samples: {num_samples}")
    print(f"   - Min image size: {min_size}px")
    print(f"   - Output file: {output_file}")
    
    # Supported image extensions
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all images
    all_images = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTS:
                img_path = os.path.join(root, file)
                all_images.append(img_path)
    
    if not all_images:
        print(f"❌ No images found in {data_dir}")
        return False
    
    print(f"   ✓ Found {len(all_images)} total images")
    
    # Filter images by size and quality
    valid_images = []
    skipped_count = 0
    for img_path in all_images:
        try:
            # Quick size check using cv2
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                if min(h, w) >= min_size:
                    valid_images.append(img_path)
        except (cv2.error, IOError, OSError) as e:
            # Skip corrupted or unreadable images
            skipped_count += 1
            continue
    
    if skipped_count > 0:
        print(f"   ⚠️ Skipped {skipped_count} corrupted/unreadable images")
    
    print(f"   ✓ {len(valid_images)} images meet size requirements")
    
    if len(valid_images) < num_samples:
        print(f"⚠️  Only {len(valid_images)} valid images available (requested {num_samples})")
        num_samples = len(valid_images)
    
    # Random sampling with stratification by directory
    # This ensures diversity across different conditions/scenes
    dir_groups = {}
    for img_path in valid_images:
        dir_name = os.path.dirname(img_path)
        if dir_name not in dir_groups:
            dir_groups[dir_name] = []
        dir_groups[dir_name].append(img_path)
    
    selected_images = []
    samples_per_dir = max(1, num_samples // len(dir_groups))
    
    for dir_name, images in dir_groups.items():
        # Sample from each directory
        dir_samples = min(samples_per_dir, len(images))
        selected = random.sample(images, dir_samples)
        selected_images.extend(selected)
        print(f"   ✓ Selected {dir_samples} images from {os.path.basename(dir_name)}")
    
    # Fill remaining slots randomly if needed
    if len(selected_images) < num_samples:
        remaining = num_samples - len(selected_images)
        available = [img for img in valid_images if img not in selected_images]
        if available:
            additional = random.sample(available, min(remaining, len(available)))
            selected_images.extend(additional)
    
    # Shuffle final selection
    random.shuffle(selected_images)
    selected_images = selected_images[:num_samples]
    
    # Write dataset file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for img_path in selected_images:
            f.write(f"{os.path.abspath(img_path)}\n")
    
    print(f"✅ Quantization dataset created: {output_file}")
    print(f"   - Total samples: {len(selected_images)}")
    print(f"   - Diversity: {len(set(os.path.dirname(p) for p in selected_images))} directories")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare quantization dataset from images")
    parser.add_argument("data_dir", help="Directory containing training/validation images")
    parser.add_argument("-o", "--output", default="config/quant_dataset.txt",
                       help="Output dataset file path")
    parser.add_argument("-n", "--num-samples", type=int, default=300,
                       help="Number of images to select")
    parser.add_argument("--min-size", type=int, default=200,
                       help="Minimum image dimension (px)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("📊 Quantization Dataset Preparation")
    print("=" * 40)
    
    # Validate input directory
    if not os.path.isdir(args.data_dir):
        print(f"❌ Directory not found: {args.data_dir}")
        return 1
    
    # Collect images
    success = collect_images(
        args.data_dir, 
        args.output, 
        args.num_samples, 
        args.min_size
    )
    
    if success:
        print("\n🎯 Ready for RKNN conversion!")
        print(f"   Use: python tools/export_rknn.py <model.onnx> -d {args.output}")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())