#!/usr/bin/env bash
set -euo pipefail

# Simplified ARM64 Docker test (避免QEMU崩溃)
# 只测试Python依赖，不涉及C++编译

echo "=========================================="
echo "Simplified ARM64 Docker Test"
echo "=========================================="
echo ""

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装"
    exit 1
fi

echo "✅ Docker版本: $(docker --version)"

# 检查buildx
if ! docker buildx version &> /dev/null; then
    echo "❌ Docker buildx未安装"
    exit 1
fi

echo "✅ Buildx版本: $(docker buildx version)"

# 配置多架构支持
echo ""
echo "配置多架构支持..."
docker run --privileged --rm tonistiigi/binfmt --install all 2>&1 | head -5 || true
docker buildx create --name multiarch --driver docker-container --use 2>/dev/null || docker buildx use multiarch || true
docker buildx inspect --bootstrap | head -10

# 验证支持的平台
echo ""
echo "支持的平台:"
docker buildx inspect | grep "Platforms:" || echo "  无法获取平台列表"

# 构建简化的ARM64镜像
echo ""
echo "=========================================="
echo "构建简化的ARM64测试镜像"
echo "使用 Dockerfile.test（无C++编译，避免QEMU崩溃）"
echo "=========================================="
echo ""

cd "$(dirname "$0")/.."

# 使用简化的Dockerfile
docker buildx build \
  --platform linux/arm64 \
  --file Dockerfile.test \
  --load \
  -t rk-app:arm64-test \
  .

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Docker构建失败"
    echo ""
    echo "可能原因:"
    echo "  1. rknn-toolkit-lite2在PyPI上不存在（这是预期的）"
    echo "  2. 网络问题（镜像源连接失败）"
    echo "  3. QEMU问题（某些包无法在模拟环境中安装）"
    echo ""
    echo "建议:"
    echo "  - 检查网络连接"
    echo "  - 等待清华/阿里云镜像恢复"
    echo "  - 或者跳过Docker测试，直接用tar.gz部署包"
    exit 1
fi

echo ""
echo "✅ ARM64镜像构建成功"
echo ""

# 测试1: 基础Python版本
echo "=========================================="
echo "测试1: Python环境"
echo "=========================================="

docker run --platform linux/arm64 --rm rk-app:arm64-test \
  python3 -c "import sys; print(f'Python {sys.version}')"

echo "✅ 测试1通过"
echo ""

# 测试2: NumPy导入
echo "=========================================="
echo "测试2: NumPy"
echo "=========================================="

docker run --platform linux/arm64 --rm rk-app:arm64-test \
  python3 -c "import numpy as np; print(f'NumPy {np.__version__}')"

echo "✅ 测试2通过"
echo ""

# 测试3: OpenCV导入
echo "=========================================="
echo "测试3: OpenCV"
echo "=========================================="

docker run --platform linux/arm64 --rm rk-app:arm64-test \
  python3 -c "import cv2; print(f'OpenCV {cv2.__version__}')"

echo "✅ 测试3通过"
echo ""

# 测试4: RKNNLite导入（可能失败）
echo "=========================================="
echo "测试4: RKNNLite（可能失败，需手动下载wheel）"
echo "=========================================="

if docker run --platform linux/arm64 --rm rk-app:arm64-test \
  python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')" 2>&1; then
    echo "✅ 测试4通过 - RKNNLite可从PyPI安装！"
else
    echo "⚠️  测试4失败 - RKNNLite无法从PyPI安装（预期内）"
    echo ""
    echo "解决方案:"
    echo "  板子上手动安装: pip3 install <wheel文件>"
    echo "  Wheel下载地址: https://github.com/rockchip-linux/rknn-toolkit2/releases"
fi

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo ""
echo "结论:"
echo "  ✅ ARM64环境可以运行"
echo "  ✅ Python基础依赖可以安装"
echo "  ⚠️  RKNNLite可能需要手动安装（板子上用wheel）"
echo ""
echo "下一步:"
echo "  使用 rk-deploy-complete.tar.gz 部署到实际板子"
echo "  板上运行 bash scripts/deploy/install_dependencies.sh"
echo ""
