#!/usr/bin/env bash
set -euo pipefail

# RK3588板上依赖安装脚本
# 处理常见的pip安装问题（国内镜像、ARM64兼容性等）

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
REQ_FILE="$ROOT/requirements_board.txt"

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

echo ""
if [[ -f "$REQ_FILE" ]]; then
    echo "安装 board Python 依赖: $REQ_FILE"
    mapfile -t COMMON_REQUIREMENTS < <(grep -vi '^rknn-toolkit-lite2' "$REQ_FILE" | sed '/^#/d;/^$/d')
    if ((${#COMMON_REQUIREMENTS[@]})); then
        python3 -m pip install "${COMMON_REQUIREMENTS[@]}"
    fi
else
    echo "requirements_board.txt 不存在，使用内置依赖列表"
    python3 -m pip install numpy==1.24.3 opencv-python-headless==4.9.0.80 pillow==11.3.0 'PyYAML>=6.0,<7.0'
fi

if python3 -c "from rknnlite.api import RKNNLite" >/dev/null 2>&1; then
    echo "✅ rknn-toolkit-lite2 已安装"
else
    echo ""
    echo "安装 rknn-toolkit-lite2..."
    if [[ -f "$REQ_FILE" ]]; then
        RKNN_REQUIREMENT="$(grep -i '^rknn-toolkit-lite2' "$REQ_FILE" | head -n 1 || true)"
    else
        RKNN_REQUIREMENT="rknn-toolkit-lite2>=2.3.2"
    fi
    RKNN_REQUIREMENT="${RKNN_REQUIREMENT:-rknn-toolkit-lite2>=2.3.2}"

    if python3 -m pip install "$RKNN_REQUIREMENT"; then
        echo "✅ rknn-toolkit-lite2 安装成功"
    else
        echo "⚠️  自动安装 rknn-toolkit-lite2 失败"
        echo "   请确认镜像源可访问，或手动安装与板端驱动匹配的 wheel。"
        exit 1
    fi
fi

NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
echo "  NumPy版本: $NUMPY_VERSION"

# 验证安装
echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="

python3 << 'PYEOF'
import sys
from pathlib import Path

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

    model_path = next(iter(sorted(Path.cwd().glob("artifacts/models/*.rknn"))), None)
    if model_path is None:
        print("NPU初始化: SKIPPED (no local .rknn model found)")
    else:
        rknn = RKNNLite()
        try:
            ret = rknn.load_rknn(str(model_path))
            if ret != 0:
                print(f"模型加载: ⚠️  FAILED (ret={ret})")
            else:
                ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
                if ret == 0:
                    print(f"NPU初始化: ✅ SUCCESS ({model_path.name})")
                else:
                    print(f"NPU初始化: ⚠️  FAILED (ret={ret})")
        finally:
            rknn.release()
except Exception as e:
    print(f"RKNNLite: ❌ FAILED - {e}")
PYEOF

echo ""
echo "=========================================="
echo "✅ 依赖安装完成"
echo "=========================================="
