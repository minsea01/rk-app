FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-distutils python3.8-dev \
    python3-pip python3-opencv \
    git curl ca-certificates pkg-config \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Default Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Version pins are coordinated in configs/toolchain_versions.toml
ARG RKNN_TOOLKIT2_VER=1.7.5
ARG ULTRALYTICS_VER=8.2.79

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      numpy Pillow onnx onnxsim onnxruntime \
      opencv-python-headless \
      ultralytics==${ULTRALYTICS_VER} \
      rknn-toolkit2==${RKNN_TOOLKIT2_VER}

WORKDIR /work
CMD ["bash"]
