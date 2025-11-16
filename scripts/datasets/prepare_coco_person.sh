#!/usr/bin/env bash
set -euo pipefail

# Download COCO val2017 person subset for mAP evaluation

echo "=========================================="
echo "COCO Person Validation Set Preparation"
echo "=========================================="
echo ""

cd "$(dirname "$0")/../.."
COCO_DIR="datasets/coco"

# Check if annotations exist
if [ ! -f "$COCO_DIR/annotations/instances_val2017.json" ]; then
    echo "Downloading COCO val2017 annotations..."
    cd "$COCO_DIR"
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    cd -
fi

# Check if val2017 images exist
if [ ! -d "$COCO_DIR/val2017" ]; then
    echo "Downloading COCO val2017 images (1GB, may take 10-20 minutes)..."
    cd "$COCO_DIR"
    wget -c http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
    rm val2017.zip
    cd -
fi

echo ""
echo "Creating person-only subset..."

python3 << 'PYEOF'
import json
from pathlib import Path

coco_dir = Path("datasets/coco")
ann_file = coco_dir / "annotations/instances_val2017.json"

with open(ann_file) as f:
    coco = json.load(f)

# Person category_id = 1 in COCO
person_id = 1

# Filter annotations
person_anns = [a for a in coco["annotations"] if a["category_id"] == person_id]
person_img_ids = {a["image_id"] for a in person_anns}
person_images = [i for i in coco["images"] if i["id"] in person_img_ids]

print(f"Total val2017 images: {len(coco['images'])}")
print(f"Images with person: {len(person_images)}")
print(f"Person annotations: {len(person_anns)}")

# Create person-only annotation file
person_coco = {
    "images": person_images,
    "annotations": person_anns,
    "categories": [c for c in coco["categories"] if c["id"] == person_id]
}

out_file = coco_dir / "annotations/person_val2017.json"
with open(out_file, 'w') as f:
    json.dump(person_coco, f)

print(f"\nSaved to: {out_file}")
print(f"Ready for mAP evaluation!")
PYEOF

echo ""
echo "=========================================="
echo "✅ COCO person validation set ready"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  python scripts/evaluation/pedestrian_map_evaluator.py \\"
echo "    --model artifacts/models/best.rknn \\"
echo "    --annotations datasets/coco/annotations/person_val2017.json \\"
echo "    --images datasets/coco/val2017"
