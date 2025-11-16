#!/bin/bash
# CityPersons Dataset Downloader
# Dataset: 5,000+ images, 35,000+ person annotations
# Paper: https://arxiv.org/abs/1702.05693

set -e

DATASET_DIR="datasets/citypersons"
DOWNLOAD_DIR="$DATASET_DIR/raw"

echo "============================================="
echo "CityPersons Dataset Download"
echo "============================================="

# Create directories
mkdir -p "$DOWNLOAD_DIR"
cd "$DATASET_DIR"

echo ""
echo "Note: CityPersons requires CityScapes dataset as base"
echo "You need to:"
echo "1. Register at https://www.cityscapes-dataset.com/"
echo "2. Download leftImg8bit_trainvaltest.zip (11GB)"
echo "3. Download gtBboxCityPersons.zip (annotations)"
echo ""
echo "Direct download links (after registration):"
echo "  Images: https://www.cityscapes-dataset.com/file-handling/?packageID=3"
echo "  Annotations: https://github.com/cvgroup-njust/CityPersons"
echo ""

# Check if user has already downloaded files
if [ -f "raw/leftImg8bit_trainvaltest.zip" ] && [ -f "raw/gtBboxCityPersons.zip" ]; then
    echo "✓ Files already downloaded"
else
    echo "⚠️  Please download the following files manually and place in $DOWNLOAD_DIR:"
    echo "   1. leftImg8bit_trainvaltest.zip (11GB)"
    echo "   2. gtBboxCityPersons.zip (annotations)"
    echo ""
    echo "Waiting for files..."

    while true; do
        if [ -f "raw/leftImg8bit_trainvaltest.zip" ] && [ -f "raw/gtBboxCityPersons.zip" ]; then
            break
        fi
        sleep 5
    done
fi

echo ""
echo "Extracting images..."
if [ ! -d "raw/leftImg8bit" ]; then
    unzip -q raw/leftImg8bit_trainvaltest.zip -d raw/
    echo "✓ Images extracted"
else
    echo "✓ Images already extracted"
fi

echo ""
echo "Extracting annotations..."
if [ ! -d "raw/annotations" ]; then
    unzip -q raw/gtBboxCityPersons.zip -d raw/
    echo "✓ Annotations extracted"
else
    echo "✓ Annotations already extracted"
fi

echo ""
echo "============================================="
echo "Dataset Structure:"
echo "============================================="
tree -L 2 "$DATASET_DIR" || find "$DATASET_DIR" -maxdepth 2 -type d

echo ""
echo "✓ CityPersons download complete!"
echo ""
echo "Next steps:"
echo "  1. Convert annotations: bash scripts/datasets/prepare_citypersons.sh"
echo "  2. Verify dataset: python scripts/datasets/verify_citypersons.py"
echo "  3. Train model: bash scripts/train/train_citypersons.sh"
