#!/bin/bash
# Quick Start Script for Docker GPU Training
# Usage: bash docker_quick_start.sh

set -e

echo "=================================================="
echo "   RK3588 Pedestrian Detection - GPU Training"
echo "=================================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi
echo "✅ Docker found: $(docker --version)"

# Check NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker runtime not working."
    echo "   Please install nvidia-docker2 and restart Docker daemon."
    exit 1
fi
echo "✅ NVIDIA Docker runtime working"

# Check GPU
echo ""
echo "=== GPU Information ==="
docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "=== Next Steps ==="
echo "1. Build training image:"
echo "   docker-compose -f docker-compose.train.yml build train-gpu"
echo ""
echo "2. Start training container:"
echo "   docker-compose -f docker-compose.train.yml run --rm train-gpu bash"
echo ""
echo "3. Inside container, prepare dataset:"
echo "   bash scripts/datasets/download_citypersons.sh"
echo "   python scripts/datasets/prepare_citypersons.py"
echo ""
echo "4. Start training:"
echo "   bash scripts/train/train_citypersons.sh"
echo ""
echo "See docs/DOCKER_TRAINING_GUIDE.md for detailed instructions."
