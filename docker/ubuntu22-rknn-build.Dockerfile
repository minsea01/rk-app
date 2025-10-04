FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ubuntu 22.04 uses Python 3.10 by default
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    python3-opencv python3-numpy \
    git curl ca-certificates pkg-config build-essential \
    cmake ninja-build \
    libopencv-dev libyaml-cpp-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment for Python dependencies
RUN python3 -m venv /opt/rknn-env
ENV PATH="/opt/rknn-env/bin:$PATH"

# Install RKNN-Toolkit2 (compatible with Python 3.10)
ARG RKNN_TOOLKIT2_VER=2.3.2
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.26.4 Pillow onnx==1.16.1 onnxsim onnxruntime \
      opencv-python-headless \
      ultralytics==8.3.204 \
      rknn-toolkit2==${RKNN_TOOLKIT2_VER}

# Install PyTorch for model conversion (CPU version for lighter build)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /work
CMD ["bash"]
