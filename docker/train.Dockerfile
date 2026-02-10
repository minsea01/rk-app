ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG PYTHON_PACKAGE=python3
ARG PIP_PACKAGE=python3-pip
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG ULTRALYTICS_VER=8.2.79

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${PYTHON_PACKAGE} ${PIP_PACKAGE} python3-dev \
    git curl ca-certificates \
    libgl1 libglib2.0-0 build-essential pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      --index-url ${TORCH_INDEX_URL} \
      torch torchvision torchaudio && \
    python3 -m pip install --no-cache-dir \
      ultralytics==${ULTRALYTICS_VER} \
      opencv-python-headless \
      pycocotools \
      onnx onnxsim \
      tensorboard wandb clearml

WORKDIR /work
CMD ["bash"]

