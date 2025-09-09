#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import sys


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_images(root: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return files


def from_yaml(yaml_path: Path, include_val: bool = True):
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(
            f"PyYAML not installed. Please `pip install pyyaml` in your env. Error: {e}"
        )
    y = yaml.safe_load(yaml_path.read_text())
    base = Path(y.get('path') or yaml_path.parent)
    parts = []
    tr = y.get('train')
    if tr is not None:
        parts.append(base / tr)
    if include_val:
        va = y.get('val')
        if va is not None:
            parts.append(base / va)
    images = []
    for p in parts:
        p = Path(p)
        if p.exists():
            images.extend(find_images(p))
    return images


def save_list(paths, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / 'calib.txt'
    with open(list_path, 'w') as f:
        for p in paths:
            f.write(str(Path(p).resolve()) + '\n')
    return list_path


def main():
    ap = argparse.ArgumentParser(description='Auto-build calibration image list from dataset YAML or folder')
    ap.add_argument('--src', type=Path, required=True, help='dataset YAML or image folder')
    ap.add_argument('--out', type=Path, default=Path('datasets/calib'), help='output folder to place calib.txt')
    ap.add_argument('--num', type=int, default=300, help='number of images to sample')
    ap.add_argument('--include-val', action='store_true', default=True, help='include val set (for YAML source)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    src = args.src
    if not src.exists():
        raise SystemExit(f'Source not found: {src}')

    if src.suffix.lower() in ('.yaml', '.yml'):
        candidates = from_yaml(src, include_val=args.include_val)
    else:
        candidates = find_images(src)

    if not candidates:
        raise SystemExit('No images found from source. Check --src path or YAML content.')

    random.shuffle(candidates)
    picked = candidates[: args.num]
    list_path = save_list(picked, args.out)
    print(f'Generated {len(picked)} entries at {list_path}')
    print('Use it with:  python tools/convert_onnx_to_rknn.py --onnx artifacts/models/best.onnx --out artifacts/models/yolov8s_int8.rknn --calib', list_path)


if __name__ == '__main__':
    main()

