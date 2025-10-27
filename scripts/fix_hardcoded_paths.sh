#!/usr/bin/env bash
# Auto-fix hardcoded paths in rk-app project
# Usage: ./scripts/fix_hardcoded_paths.sh

set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "=== Fixing Hardcoded Paths in rk-app ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# 1. Fix calibration file paths
echo "[1/3] Regenerating calibration file with correct paths..."
CALIB_DIR="$PROJECT_ROOT/datasets/coco/calib_images"
if [ -d "$CALIB_DIR" ]; then
    cd "$CALIB_DIR"
    find "$CALIB_DIR" -maxdepth 1 -name "*.jpg" | sort > calib.txt
    NUM_IMAGES=$(wc -l < calib.txt)
    echo "  ✓ Generated calib.txt with $NUM_IMAGES images"
    echo "  Sample paths:"
    head -3 calib.txt | sed 's/^/    /'
else
    echo "  ⚠ Calibration directory not found: $CALIB_DIR"
fi

cd "$PROJECT_ROOT"

# 2. Fix YAML config files
echo ""
echo "[2/3] Fixing paths in YAML config files..."

CONFIG_FILES=$(find config -name "*.yaml" 2>/dev/null || true)
if [ -n "$CONFIG_FILES" ]; then
    for file in $CONFIG_FILES; do
        if grep -q "/home/minsea" "$file" 2>/dev/null; then
            echo "  Updating: $file"
            # Create backup
            cp "$file" "$file.bak"
            # Replace hardcoded paths with relative paths
            sed -i 's|/home/minsea/rk-app/||g' "$file"
            sed -i 's|/home/minsea/datasets/|datasets/|g' "$file"
            sed -i 's|/home/minsea/models/|artifacts/models/|g' "$file"
        fi
    done
    echo "  ✓ Fixed YAML config files"
else
    echo "  ⚠ No YAML config files found"
fi

# 3. Fix other data files
echo ""
echo "[3/3] Checking other data files..."

# Fix industrial dataset config if exists
if [ -f "industrial_dataset/data.yaml" ]; then
    if grep -q "/home/minsea" "industrial_dataset/data.yaml" 2>/dev/null; then
        echo "  Updating: industrial_dataset/data.yaml"
        cp "industrial_dataset/data.yaml" "industrial_dataset/data.yaml.bak"
        sed -i "s|/home/minsea/rk-app|$PROJECT_ROOT|g" "industrial_dataset/data.yaml"
    fi
fi

echo ""
echo "=== Fix Complete ==="
echo ""
echo "Summary:"
echo "  - Calibration file: datasets/coco/calib_images/calib.txt"
echo "  - Config files: config/*.yaml"
echo "  - Backups saved with .bak extension"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Test scripts: python3 scripts/run_rknn_sim.py --help"
echo "  3. Commit if everything looks good"
