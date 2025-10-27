# Ubuntu 22.04 兼容性指南

本文档说明如何在 Ubuntu 22.04 系统上构建和运行 rk-app 项目。

## 系统要求

- **操作系统**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Python**: 3.10 (系统默认版本)
- **CMake**: 3.22+ (系统自带)
- **GCC**: 11.x (系统自带，支持 C++17)

## 方法 1: 使用 Docker（推荐）

### 构建 Docker 镜像

```bash
cd /home/minsea/rk-app
docker build -f docker/ubuntu22-rknn-build.Dockerfile -t rk-app:ubuntu22 .
```

### 运行容器

```bash
# 运行交互式容器
docker run -it --rm \
  -v $(pwd):/work \
  rk-app:ubuntu22 bash

# 在容器内构建项目
cd /work
cmake --preset x86-debug
cmake --build --preset x86-debug
```

### ONNX 到 RKNN 转换

```bash
# 在容器内运行转换脚本
docker run -it --rm \
  -v $(pwd):/work \
  rk-app:ubuntu22 \
  python3 /work/test_rknn_convert.py /work/yolov8n.onnx /work/yolov8n.rknn
```

## 方法 2: 原生安装

### 1. 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 python3-dev python3-pip python3-venv \
  build-essential cmake ninja-build \
  libopencv-dev libyaml-cpp-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libgl1-mesa-dev pkg-config git
```

### 2. 创建 Python 虚拟环境

```bash
cd /home/minsea/rk-app
python3 -m venv venv-ubuntu22
source venv-ubuntu22/bin/activate
```

### 3. 安装 Python 依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装 RKNN 工具链（版本参考 configs/toolchain_versions.toml）
pip install rknn-toolkit2==1.7.5

# 安装其他依赖
pip install numpy==1.26.4 \
            onnx==1.16.1 \
            onnxsim \
            onnxruntime \
            opencv-python-headless \
            ultralytics==8.2.79

# 可选：PyTorch（用于模型训练/转换）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. 构建 C++ 项目

```bash
# x86 调试版本
cmake --preset x86-debug
cmake --build --preset x86-debug

# ARM64 交叉编译（需要交叉编译工具链）
cmake --preset arm64
cmake --build --preset arm64
```

### 5. 运行测试

```bash
# 运行单元测试
ctest --preset x86-debug

# 运行检测程序
./build/x86-debug/detect_cli --cfg config/detection/detect.yaml
```

## 关键差异说明

### Ubuntu 22.04 vs Ubuntu 24.04

| 组件 | Ubuntu 22.04 | Ubuntu 24.04 | 兼容性 |
|------|--------------|--------------|--------|
| Python | 3.10 | 3.12 | ✅ RKNN-Toolkit2 1.7.5 支持两者 |
| GCC | 11.x | 13.x | ✅ 都支持 C++17 |
| CMake | 3.22 | 3.28 | ✅ 都满足 >=3.16 要求 |
| OpenCV | 4.5.4 | 4.6+ | ✅ API 兼容 |
| ONNX | 1.16.1 | 1.19+ | ⚠️ 需要固定版本 1.16.1 |

### Ubuntu 22.04 vs Ubuntu 20.04

| 组件 | Ubuntu 20.04 | Ubuntu 22.04 | 优势 |
|------|--------------|--------------|------|
| Python | 3.8 | 3.10 | 22.04 更新，性能更好 |
| GCC | 9.x | 11.x | 22.04 支持更多 C++20 特性 |
| CMake | 3.16 | 3.22 | 22.04 版本更新 |
| 系统支持 | 到 2025 年 4 月 | 到 2027 年 4 月 | 22.04 支持更久 |

## 常见问题

### Q1: ONNX 版本冲突

**问题**: `rknn-toolkit2` 需要 `onnx>=1.16.1`，但 1.19+ 不兼容

**解决**: 固定安装 `onnx==1.16.1`

```bash
pip install onnx==1.16.1
```

### Q2: NumPy 版本过新

**问题**: NumPy 2.x 可能与某些库不兼容

**解决**: 使用 NumPy 1.x

```bash
pip install "numpy<2.0"
```

### Q3: GStreamer 开发库缺失

**问题**: 编译 GigE 支持时找不到 GStreamer

**解决**: 安装完整的开发包

```bash
sudo apt-get install -y \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-good1.0-dev
```

### Q4: yaml-cpp 找不到

**问题**: CMake 提示 `yaml-cpp not found`

**解决**: 安装开发库

```bash
sudo apt-get install -y libyaml-cpp-dev
```

## 验证安装

运行以下命令验证环境配置正确：

```bash
# 检查 Python 版本
python3 --version  # 应显示 Python 3.10.x

# 检查 CMake 版本
cmake --version    # 应显示 >= 3.22

# 检查 GCC 版本
gcc --version      # 应显示 11.x

# 检查 RKNN Toolkit
python3 -c "from rknn.api import RKNN; print('RKNN Toolkit OK')"

# 检查 ONNX
python3 -c "import onnx; print(f'ONNX {onnx.__version__}')"
```

## 推荐配置

对于 Ubuntu 22.04 系统，推荐使用以下配置：

```bash
# Python 依赖版本（已验证兼容，如需更新请同步 configs/toolchain_versions.toml）
rknn-toolkit2==1.7.5
numpy==1.26.4
onnx==1.16.1
onnxruntime==1.18.1
opencv-python-headless==4.9.0.80
ultralytics==8.2.79
```

## 总结

Ubuntu 22.04 完全兼容 rk-app 项目，提供了以下优势：

✅ 系统库版本适中，稳定性好
✅ Python 3.10 支持广泛
✅ 长期支持到 2027 年
✅ 包管理器软件源丰富

建议生产环境使用 Ubuntu 22.04 LTS。
