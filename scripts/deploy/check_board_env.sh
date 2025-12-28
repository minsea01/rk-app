#!/bin/bash
# =============================================================================
# RK3588 上板前验证脚本
# 在板子上运行此脚本，获取所有版本信息
# =============================================================================

echo "=== RK3588 环境检测 ==="
echo ""

# 1. 系统信息
echo "[1] 系统信息"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "Arch: $(uname -m)"
echo ""

# 2. NPU 驱动版本
echo "[2] NPU 驱动"
if [ -f /sys/kernel/debug/rknpu/version ]; then
    echo "RKNPU Driver: $(cat /sys/kernel/debug/rknpu/version)"
elif [ -f /sys/class/devfreq/fdab0000.npu/cur_freq ]; then
    echo "NPU Device: Found"
else
    echo "NPU Device: NOT FOUND - 请检查驱动"
fi

# NPU 设备权限
if [ -c /dev/rknpu0 ]; then
    echo "/dev/rknpu0: $(ls -la /dev/rknpu0)"
else
    echo "/dev/rknpu0: NOT FOUND"
fi
echo ""

# 3. Python 版本
echo "[3] Python 环境"
echo "Python: $(python3 --version 2>&1)"
echo "Pip: $(pip3 --version 2>&1 | head -1)"
echo ""

# 4. RKNN 相关库版本
echo "[4] RKNN 库版本"
if python3 -c "from rknnlite.api import RKNNLite; print('rknnlite:', RKNNLite().sdk_version)" 2>/dev/null; then
    :
else
    echo "rknn-toolkit-lite2: NOT INSTALLED"
fi

# librknpu.so 版本
if [ -f /usr/lib/aarch64-linux-gnu/librknpu.so ]; then
    echo "librknpu.so: Found at /usr/lib/aarch64-linux-gnu/"
elif [ -f /opt/rknpu2/lib/librknpu.so ]; then
    echo "librknpu.so: Found at /opt/rknpu2/lib/"
else
    echo "librknpu.so: NOT FOUND"
fi
echo ""

# 5. OpenCV
echo "[5] OpenCV"
python3 -c "import cv2; print('OpenCV:', cv2.__version__)" 2>/dev/null || echo "OpenCV: NOT INSTALLED"
echo ""

# 6. 输出推荐版本
echo "=== 版本建议 ==="
echo "请根据以上信息，在 PC 上使用对应版本的 rknn-toolkit2 转换模型"
echo ""
echo "RKNN SDK 版本对应表:"
echo "  - 驱动 0.9.x -> rknn-toolkit2 1.5.x"
echo "  - 驱动 1.0.x -> rknn-toolkit2 1.6.x"  
echo "  - 驱动 2.0.x -> rknn-toolkit2 2.0.x"
echo ""
echo "将此脚本输出发给开发者，确保模型转换版本正确！"
