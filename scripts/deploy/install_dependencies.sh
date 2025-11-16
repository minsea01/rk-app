#!/usr/bin/env bash
set -euo pipefail

# RK3588板上依赖安装脚本
# 处理常见的pip安装问题（国内镜像、ARM64兼容性等）

echo "=========================================="
echo "Installing RK3588 Runtime Dependencies"
echo "=========================================="

# 检测架构
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "❌ 错误: 当前架构是 $ARCH，不是 aarch64"
    echo "   这个脚本只能在RK3588板上运行"
    exit 1
fi

echo "✅ 架构检测: $ARCH"

# Python版本检查
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✅ Python版本: $PYTHON_VERSION"

# 配置pip镜像（加速下载）
echo ""
echo "配置pip镜像源（清华镜像）..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

echo "✅ pip镜像配置完成"

# 更新pip
echo ""
echo "更新pip..."
python3 -m pip install --upgrade pip

# 安装基础依赖
echo ""
echo "安装基础依赖（numpy, opencv, pillow）..."
pip3 install numpy==1.24.3 opencv-python-headless==4.9.0.80 pillow==11.0.0 pyyaml

# 检查numpy版本（必须<2.0，RKNN兼容性）
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
echo "  NumPy版本: $NUMPY_VERSION"

# 安装rknn-toolkit-lite2
echo ""
echo "安装rknn-toolkit-lite2..."

# 方法1: 尝试从PyPI安装（可能不存在）
if pip3 install rknn-toolkit-lite2 2>/dev/null; then
    echo "✅ 从PyPI安装成功"
else
    echo "⚠️  PyPI安装失败，尝试从Rockchip官方下载..."

    # 方法2: 从GitHub下载预编译wheel
    RKNN_VERSION="1.6.0"
    PYTHON_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    WHEEL_NAME="rknn_toolkit_lite2-${RKNN_VERSION}-${PYTHON_VER}-${PYTHON_VER}-linux_aarch64.whl"

    echo "  检测Python版本: $PYTHON_VER"
    echo "  需要的wheel文件: $WHEEL_NAME"
    echo ""
    echo "请手动下载并安装:"
    echo "  1. 访问: https://github.com/rockchip-linux/rknn-toolkit2/releases"
    echo "  2. 下载: $WHEEL_NAME"
    echo "  3. 安装: pip3 install $WHEEL_NAME"
    echo ""
    echo "或者使用以下命令尝试自动下载（需要网络）："
    echo "  wget https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v${RKNN_VERSION}/${WHEEL_NAME}"
    echo "  pip3 install ${WHEEL_NAME}"

    exit 1
fi

# 验证安装
echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="

python3 << 'PYEOF'
import sys
print(f"Python: {sys.version}")

import numpy as np
print(f"NumPy: {np.__version__}")

import cv2
print(f"OpenCV: {cv2.__version__}")

from PIL import Image
print(f"Pillow: {Image.__version__}")

import yaml
print(f"PyYAML: OK")

try:
    from rknnlite.api import RKNNLite
    print(f"RKNNLite: OK")

    # 尝试初始化
    rknn = RKNNLite()
    ret = rknn.init_runtime()
    if ret == 0:
        print(f"NPU初始化: ✅ SUCCESS")
    else:
        print(f"NPU初始化: ⚠️  FAILED (ret={ret})")
        print("  可能原因: NPU驱动未加载，运行 'sudo modprobe rknpu'")
    rknn.release()
except Exception as e:
    print(f"RKNNLite: ❌ FAILED - {e}")
PYEOF

echo ""
echo "=========================================="
echo "✅ 依赖安装完成"
echo "=========================================="
