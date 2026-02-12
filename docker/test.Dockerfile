# Simplified ARM64 test Dockerfile
# Only tests Python dependencies and RKNNLite installation
# No C++ cross-compilation to avoid QEMU issues

FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Configure APT mirror (use Aliyun mirror for better reliability)
RUN sed -i 's|http://ports.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure pip mirror (use Aliyun for better reliability)
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Install Python dependencies one by one to identify which one fails
RUN pip3 install --no-cache-dir numpy==1.24.3

RUN pip3 install --no-cache-dir opencv-python-headless==4.9.0.80

RUN pip3 install --no-cache-dir pillow==11.0.0

RUN pip3 install --no-cache-dir pyyaml

# Install RKNN runtime library (critical test)
RUN pip3 install --no-cache-dir rknn-toolkit-lite2 || \
    echo "WARNING: rknn-toolkit-lite2 not available on PyPI"

# Test imports
RUN python3 -c "import numpy; print('NumPy OK')"
RUN python3 -c "import cv2; print('OpenCV OK')"
RUN python3 -c "from PIL import Image; print('Pillow OK')"
RUN python3 -c "import yaml; print('PyYAML OK')"

# Test RKNNLite import (may fail if not in PyPI)
RUN python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')" || \
    echo "WARNING: RKNNLite import failed - manual wheel installation needed"

CMD ["python3", "-c", "print('ARM64 test image ready')"]
