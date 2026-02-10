ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG RKNN_TOOLKIT2_VER=1.7.5
ARG ULTRALYTICS_VER=8.2.79

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    python3-opencv python3-numpy \
    git curl ca-certificates pkg-config build-essential \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/rknn-env
ENV PATH="/opt/rknn-env/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.26.4 Pillow \
      onnx==1.16.1 onnxsim onnxruntime \
      opencv-python-headless \
      ultralytics==${ULTRALYTICS_VER} \
      rknn-toolkit2==${RKNN_TOOLKIT2_VER} && \
    pip install --no-cache-dir \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /work
CMD ["bash"]

