#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_numeric_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def is_numeric_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def resolve_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        # broken symlink case
        try:
            return p.resolve(strict=False).stat().st_size
        except Exception:
            return -1
    except Exception:
        return -1


def collect_images(root: Path) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                p = Path(dirpath) / fn
                stem = p.stem
                # If multiple extensions per stem, keep the first seen
                if stem not in images:
                    images[stem] = p
    return images


def collect_labels(root: Path) -> Dict[str, Path]:
    labels: Dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                p = Path(dirpath) / fn
                labels[p.stem] = p
    return labels


class LabelStats:
    def __init__(self) -> None:
        self.empty_files: List[Path] = []
        self.labels_without_image: List[Path] = []
        self.images_without_label: List[Path] = []
        self.bad_format: List[Tuple[Path, int, str]] = []  # (file, line_no, line)
        self.bad_bbox: List[Tuple[Path, int, str]] = []    # (file, line_no, line)
        self.class_hist: Dict[int, int] = {}
        self.valid_image_list: List[Path] = []
        self.valid_label_files: List[Path] = []


def validate_label_file(label_path: Path, nclasses: Optional[int]) -> Tuple[bool, Dict[int, int], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Returns: (is_valid, local_hist, bad_format_lines, bad_bbox_lines)
    A file is considered valid if:
      - file is non-empty
      - every non-empty line has 5 numeric fields
      - bbox fields are in [0,1] with w,h > 0 and <=1
      - if nclasses provided: class id in [0, nclasses-1]
    """
    bad_fmt: List[Tuple[int, str]] = []
    bad_box: List[Tuple[int, str]] = []
    local_hist: Dict[int, int] = {}

    size = resolve_size(label_path)
    if size <= 0:
        return False, local_hist, bad_fmt, bad_box

    try:
        with label_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception:
        # unreadable -> treat as invalid
        return False, local_hist, bad_fmt, bad_box

    if not lines:
        return False, local_hist, bad_fmt, bad_box

    ok = True
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            # blank lines are ignored, but file must have at least one valid line overall
            continue
        parts = s.split()
        if len(parts) != 5:
            ok = False
            bad_fmt.append((i, line))
            continue
        c_str, x_str, y_str, w_str, h_str = parts
        if not (is_numeric_int(c_str) and is_numeric_float(x_str) and is_numeric_float(y_str) and is_numeric_float(w_str) and is_numeric_float(h_str)):
            ok = False
            bad_fmt.append((i, line))
            continue
        c = int(c_str)
        x = float(x_str)
        y = float(y_str)
        w = float(w_str)
        h = float(h_str)
        # Class range check (if known)
        if nclasses is not None and (c < 0 or c >= nclasses):
            ok = False
            bad_fmt.append((i, line))
            continue
        # BBox checks: coordinates [0,1], w/h > 0 and <= 1
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            ok = False
            bad_box.append((i, line))
            continue
        local_hist[c] = local_hist.get(c, 0) + 1

    # File valid only if at least one valid line exists and no invalids
    has_any = sum(local_hist.values()) > 0
    return ok and has_any, local_hist, bad_fmt, bad_box


def process_split(root: Path, split: str, nclasses: Optional[int]) -> LabelStats:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    stats = LabelStats()

    img_map = collect_images(images_dir) if images_dir.exists() else {}
    lbl_map = collect_labels(labels_dir) if labels_dir.exists() else {}

    img_stems = set(img_map.keys())
    lbl_stems = set(lbl_map.keys())

    # Labels without corresponding image
    for stem in sorted(lbl_stems - img_stems):
        stats.labels_without_image.append(lbl_map[stem])

    # Images without label file
    for stem in sorted(img_stems - lbl_stems):
        stats.images_without_label.append(img_map[stem])

    # Validate label files that have images
    for stem in sorted(img_stems & lbl_stems):
        lp = lbl_map[stem]
        size = resolve_size(lp)
        if size <= 0:
            stats.empty_files.append(lp)
            continue
        ok, local_hist, bad_fmt, bad_box = validate_label_file(lp, nclasses)
        if not ok:
            for i, line in bad_fmt:
                stats.bad_format.append((lp, i, line))
            for i, line in bad_box:
                stats.bad_bbox.append((lp, i, line))
            continue
        # valid
        stats.valid_label_files.append(lp)
        for c, n in local_hist.items():
            stats.class_hist[c] = stats.class_hist.get(c, 0) + n
        stats.valid_image_list.append(img_map[stem])

    return stats


def write_list_file(out_path: Path, items: List[Path]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in items:
            f.write(str(p) + "\n")


def make_names(n: int) -> List[str]:
    return [f"class{i}" for i in range(n)]


def write_data_yaml(out_path: Path, train_list: Path, val_list: Path, names: Optional[List[str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"train: {train_list}\n")
        f.write(f"val:   {val_list}\n")
        if names is not None:
            f.write("names:\n")
            for i, n in enumerate(names):
                f.write(f"  {i}: {n}\n")


def plural(n: int, s: str) -> str:
    return f"{n} {s}" if n == 1 else f"{n} {s}s"


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan a YOLO dataset, generate clean train/val lists and a minimal data YAML.")
    ap.add_argument("root", type=Path, help="Dataset root containing images/{train,val} and labels/{train,val}")
    ap.add_argument("--splits", nargs="*", default=["train", "val"], help="Splits to process (default: train val)")
    ap.add_argument("--nclasses", type=int, default=None, help="Number of classes (optional). If omitted, class-id range will not be enforced.")
    ap.add_argument("--names", type=str, default=None, help="Comma-separated class names (overrides --nclasses)")
    ap.add_argument("--out-data-yaml", type=Path, default=None, help="Output data yaml path (default: <root>/data_lists.yaml)")
    ap.add_argument("--prefix", type=str, default="", help="Prefix for list filenames (default: none)")
    ap.add_argument("--allow-missing-splits", action="store_true", default=False,
                    help="Allow missing train/val splits. If set, writes empty list files for missing splits and proceeds; otherwise exits with error.")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2

    names: Optional[List[str]] = None
    if args.names is not None:
        names = [s.strip() for s in args.names.split(",") if s.strip()]
        nclasses = len(names)
    else:
        nclasses = args.nclasses
        if nclasses is not None:
            names = make_names(nclasses)

    total_summary = {}
    list_paths: Dict[str, Path] = {}

    for split in args.splits:
        stats = process_split(root, split, nclasses)

        # Write split list (only valid images)
        out_list = root / f"{args.prefix}{split}.txt"
        write_list_file(out_list, stats.valid_image_list)
        list_paths[split] = out_list

        # Summaries
        total_imgs = len(collect_images(root / "images" / split)) if (root / "images" / split).exists() else 0
        print(f"=== {split} summary ===")
        print(f"images total:        {total_imgs}")
        print(f"labels valid imgs:   {len(stats.valid_image_list)}")
        print(f"labels empty files:  {len(stats.empty_files)}")
        print(f"images w/o label:    {len(stats.images_without_label)}")
        print(f"labels w/o image:    {len(stats.labels_without_image)}")
        print(f"bad format lines:    {len(stats.bad_format)}")
        print(f"bad bbox lines:      {len(stats.bad_bbox)}")

        # Small previews of issues
        if stats.images_without_label:
            print("-- images without label (first 10):")
            for p in stats.images_without_label[:10]:
                print(f"   {p}")
        if stats.labels_without_image:
            print("-- labels without image (first 10):")
            for p in stats.labels_without_image[:10]:
                print(f"   {p}")
        if stats.bad_format:
            print("-- bad format (first 10):")
            for p, ln, line in stats.bad_format[:10]:
                print(f"   {p}:{ln}: {line}")
        if stats.bad_bbox:
            print("-- bad bbox (first 10):")
            for p, ln, line in stats.bad_bbox[:10]:
                print(f"   {p}:{ln}: {line}")

        # Class histogram
        if stats.class_hist:
            print("-- class histogram:")
            for c in sorted(stats.class_hist.keys()):
                print(f"   {c:3d}: {stats.class_hist[c]}")
        else:
            print("-- class histogram: (none)")

        total_summary[split] = {
            "valid_imgs": len(stats.valid_image_list),
            "total_imgs": total_imgs,
            "bad_format": len(stats.bad_format),
            "bad_bbox": len(stats.bad_bbox),
        }

    # Write data yaml
    data_yaml = args.out_data_yaml or (root / "data_lists.yaml")
    train_list = list_paths.get("train")
    val_list = list_paths.get("val")

    missing = []
    if train_list is None:
        missing.append("train")
    if val_list is None:
        missing.append("val")

    if missing and not args.allow_missing_splits:
        print(f"ERROR: missing split(s): {', '.join(missing)}. Refusing to write an invalid data YAML.", file=sys.stderr)
        print("Hint: provide both images/ and labels/ for the missing split(s), or rerun with --allow-missing-splits to generate empty lists.", file=sys.stderr)
        return 3

    if train_list is None:
        # write an empty placeholder list under the dataset root
        train_list = root / f"{args.prefix}train_empty.txt"
        write_list_file(train_list, [])
        print(f"WARN: train split missing; wrote empty list: {train_list}")
    if val_list is None:
        val_list = root / f"{args.prefix}val_empty.txt"
        write_list_file(val_list, [])
        print(f"WARN: val split missing; wrote empty list: {val_list}")

    write_data_yaml(data_yaml, train_list, val_list, names)
    print(f"\nWrote lists and data yaml:")
    for k, p in list_paths.items():
        print(f" - {k}.txt: {p} ({sum(1 for _ in open(p, 'r', encoding='utf-8'))} images)")
    print(f" - data:    {data_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
