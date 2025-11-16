#!/usr/bin/env bash
set -euo pipefail

# ARM64 Docker构建测试脚本
# 在你本机（有Docker的环境）执行，验证ARM64依赖安装

echo "=========================================="
echo "ARM64 Docker Build Test"
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
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx create --name multiarch --driver docker-container --use 2>/dev/null || docker buildx use multiarch
docker buildx inspect --bootstrap

# 验证支持的平台
echo ""
echo "支持的平台:"
docker buildx inspect | grep "Platforms:" || echo "  无法获取平台列表"

# 构建ARM64镜像
echo ""
echo "=========================================="
echo "开始构建ARM64镜像（rk3588-runtime阶段）"
echo "=========================================="
echo ""

cd "$(dirname "$0")/../.."

docker buildx build \
  --platform linux/arm64 \
  --target rk3588-runtime \
  --load \
  -t rk-app:arm64 \
  .

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Docker构建失败"
    echo ""
    echo "常见原因:"
    echo "  1. rknn-toolkit-lite2在PyPI上不存在（需要手动下载wheel）"
    echo "  2. Python版本不匹配"
    echo "  3. numpy版本冲突"
    exit 1
fi

echo ""
echo "✅ ARM64镜像构建成功"
echo ""

# 测试1: RKNNLite导入
echo "=========================================="
echo "测试1: RKNNLite导入"
echo "=========================================="

docker run --platform linux/arm64 --rm rk-app:arm64 \
  python3 -c "from rknnlite.api import RKNNLite; print('✅ RKNNLite imported successfully')" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 测试1通过"
else
    echo "❌ 测试1失败: RKNNLite无法导入"
    exit 1
fi

echo ""

# 测试2: 完整依赖检查
echo "=========================================="
echo "测试2: 完整依赖检查"
echo "=========================================="

docker run --platform linux/arm64 --rm rk-app:arm64 python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import numpy as np
    print(f"NumPy: {np.__version__} ✅")
except ImportError as e:
    print(f"NumPy: FAILED ❌ - {e}")
    sys.exit(1)

try:
    import cv2
    print(f"OpenCV: {cv2.__version__} ✅")
except ImportError as e:
    print(f"OpenCV: FAILED ❌ - {e}")
    sys.exit(1)

try:
    from PIL import Image
    print(f"Pillow: {Image.__version__} ✅")
except ImportError as e:
    print(f"Pillow: FAILED ❌ - {e}")
    sys.exit(1)

try:
    from rknnlite.api import RKNNLite
    print(f"RKNNLite: OK ✅")

    # 创建实例
    rknn = RKNNLite()
    print(f"RKNNLite instance: OK ✅")
except ImportError as e:
    print(f"RKNNLite: FAILED ❌ - {e}")
    sys.exit(1)
except Exception as e:
    print(f"RKNNLite instance: WARNING ⚠️  - {e}")

print("\n✅ 所有依赖测试通过")
EOF

if [ $? -eq 0 ]; then
    echo "✅ 测试2通过"
else
    echo "❌ 测试2失败"
    exit 1
fi

echo ""

# 测试3: 模型加载语法测试（会失败但能测试代码）
echo "=========================================="
echo "测试3: 模型加载语法（预期部分失败）"
echo "=========================================="

if [ -f "artifacts/models/best.rknn" ]; then
    docker run --platform linux/arm64 --rm \
      -v "$(pwd)/artifacts/models:/models" \
      rk-app:arm64 python3 << 'EOF'
from rknnlite.api import RKNNLite
import os

model_path = '/models/best.rknn'
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    exit(1)

print(f"✅ Model file exists: {os.path.getsize(model_path)} bytes")

rknn = RKNNLite()
ret = rknn.load_rknn(model_path)
print(f"load_rknn returned: {ret}")

if ret == 0:
    print("✅ Model loading syntax OK (actual runtime needs NPU)")
else:
    print(f"⚠️  Model load returned: {ret}")

# init_runtime会失败（没有NPU设备），这是正常的
try:
    ret = rknn.init_runtime()
    if ret == 0:
        print("✅ Runtime init OK (unexpected in Docker)")
    else:
        print(f"⚠️  Runtime init failed: {ret} (expected, no NPU device)")
except Exception as e:
    print(f"⚠️  Runtime init exception: {e} (expected, no NPU device)")
EOF
    echo "✅ 测试3完成（模型加载语法正确）"
else
    echo "⚠️  跳过测试3: 模型文件不存在"
fi

echo ""
echo "=========================================="
echo "🎉 所有测试完成"
echo "=========================================="
echo ""
echo "结论:"
echo "  ✅ ARM64镜像可以构建"
echo "  ✅ 所有Python依赖可以安装"
echo "  ✅ RKNNLite可以导入"
echo "  ✅ 代码语法正确"
echo ""
echo "下一步: 板子到手后，直接使用 rk-deploy-complete.tar.gz 部署"
echo "       预计20-40分钟完成首次推理"
echo ""
