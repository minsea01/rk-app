#!/usr/bin/env bash
set -euo pipefail

# 打包WSL项目文件用于传输到RK3588板子
# 生成最小化部署包，仅包含板上运行必需的文件

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_FILE="$ROOT_DIR/rk-app-board-deploy.tar.gz"
TEMP_DIR="/tmp/rk-app-pack-$$"

echo "=========================================="
echo "Packing RK3588 Board Deployment Package"
echo "=========================================="
echo ""

# 创建临时目录
mkdir -p "$TEMP_DIR/rk-app"
cd "$ROOT_DIR"

echo "[1/8] Copying Python application code..."
cp -r apps "$TEMP_DIR/rk-app/"
find "$TEMP_DIR/rk-app/apps" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR/rk-app/apps" -name "*.pyc" -delete 2>/dev/null || true

echo "[2/8] Copying essential scripts..."
mkdir -p "$TEMP_DIR/rk-app/scripts"
cp -r scripts/deploy "$TEMP_DIR/rk-app/scripts/"
cp -r scripts/profiling "$TEMP_DIR/rk-app/scripts/" 2>/dev/null || true

# 创建profiling目录（如果不存在）
mkdir -p "$TEMP_DIR/rk-app/scripts/profiling"

echo "[3/8] Copying configuration files..."
cp -r config "$TEMP_DIR/rk-app/"

echo "[4/8] Copying RKNN models..."
mkdir -p "$TEMP_DIR/rk-app/artifacts/models"
cp artifacts/models/*.rknn "$TEMP_DIR/rk-app/artifacts/models/" 2>/dev/null || {
    echo "⚠️  Warning: No .rknn models found. Run conversion pipeline first:"
    echo "   python3 tools/convert_onnx_to_rknn.py ..."
}

echo "[5/8] Copying class names..."
cp config/industrial_classes.txt "$TEMP_DIR/rk-app/config/" 2>/dev/null || {
    echo "⚠️  Warning: industrial_classes.txt not found"
}

echo "[6/8] Copying test assets..."
mkdir -p "$TEMP_DIR/rk-app/assets"
cp assets/*.jpg "$TEMP_DIR/rk-app/assets/" 2>/dev/null || true
cp assets/*.png "$TEMP_DIR/rk-app/assets/" 2>/dev/null || true

echo "[7/8] Creating board-specific files..."
# 创建README for 板子
cat > "$TEMP_DIR/rk-app/README_BOARD.md" << 'EOF'
# RK3588 Board Deployment Package

## Quick Start

### 1. Install Dependencies
```bash
bash scripts/deploy/install_dependencies.sh
```

### 2. Health Check
```bash
bash scripts/deploy/board_health_check.sh
```

### 3. Run Inference
```bash
export PYTHONPATH=$PWD
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg \
  --save result.jpg \
  --imgsz 416 \
  --conf 0.5
```

### 4. Performance Test
```bash
python3 scripts/profiling/board_benchmark.py \
  --model artifacts/models/yolo11n_416.rknn \
  --iterations 100
```

### 5. Configure Dual NIC (if needed)
```bash
sudo bash scripts/deploy/configure_dual_nic.sh
```

## File Structure
- `apps/` - Python inference application
- `scripts/deploy/` - Deployment and configuration scripts
- `scripts/profiling/` - Performance benchmarking
- `config/` - Configuration files
- `artifacts/models/` - RKNN models
- `assets/` - Test images

## Troubleshooting
See `docs/deployment/BOARD_DEPLOYMENT_QUICKSTART.md` for detailed guide.
EOF

# 创建requirements.txt for 板子（精简版）
cat > "$TEMP_DIR/rk-app/requirements-board.txt" << 'EOF'
# Minimal requirements for RK3588 board
numpy>=1.20.0,<2.0
opencv-python-headless==4.9.0.80
pillow==11.0.0
pyyaml>=6.0
# rknn-toolkit-lite2>=1.6.0  # Install via wheel from Rockchip GitHub
EOF

echo "[8/8] Creating tarball..."
cd "$TEMP_DIR"
tar czf "$OUTPUT_FILE" rk-app/

# 清理
rm -rf "$TEMP_DIR"

# 统计
SIZE=$(du -h "$OUTPUT_FILE" | awk '{print $1}')
FILE_COUNT=$(tar tzf "$OUTPUT_FILE" | wc -l)

echo ""
echo "=========================================="
echo "✅ Package created successfully!"
echo "=========================================="
echo ""
echo "Output: $OUTPUT_FILE"
echo "Size: $SIZE"
echo "Files: $FILE_COUNT"
echo ""
echo "Next steps:"
echo "  1. Transfer to board:"
echo "     scp $OUTPUT_FILE radxa@<board_ip>:/home/radxa/"
echo ""
echo "  2. On board, extract:"
echo "     tar xzf rk-app-board-deploy.tar.gz"
echo "     cd rk-app"
echo ""
echo "  3. Install dependencies:"
echo "     bash scripts/deploy/install_dependencies.sh"
echo ""
echo "  4. Run inference:"
echo "     bash scripts/deploy/rk3588_run.sh --model artifacts/models/yolo11n_416.rknn"
echo ""
