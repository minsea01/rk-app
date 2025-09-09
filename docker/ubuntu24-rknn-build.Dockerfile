FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ubuntu 24.04 uses Python 3.12 by default
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    python3-opencv python3-numpy \
    git curl ca-certificates pkg-config build-essential \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolation
RUN python3 -m venv /opt/rknn-env
ENV PATH="/opt/rknn-env/bin:$PATH"

# Install RKNN-Toolkit2 (compatible with Python 3.12)
ARG RKNN_TOOLKIT2_VER=1.7.5
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      numpy Pillow onnx onnxsim onnxruntime \
      opencv-python-headless \
      ultralytics==8.2.79 \
      rknn-toolkit2==${RKNN_TOOLKIT2_VER}

# Install additional tools for Ubuntu 24.04 compatibility
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /work
CMD ["bash"]