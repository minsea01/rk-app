FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3.10-dev \
    python3-pip git curl ca-certificates \
    libgl1 libglib2.0-0 build-essential pkg-config \
    python3-opencv && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install PyTorch (CUDA 12.1) and Ultralytics
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio && \
    python -m pip install --no-cache-dir ultralytics==8.2.79 pycocotools opencv-python-headless tqdm matplotlib

WORKDIR /work
CMD ["bash"]

