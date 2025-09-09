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
    libgl1 libglib2.0-0 libdrm2 v4l-utils ethtool iperf3 \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# rknn-toolkit-lite2 is recommended for device-side Python runtime
ARG RKNN_LITE2_VER=1.7.5
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      numpy Pillow opencv-python-headless \
      rknn-toolkit-lite2==${RKNN_LITE2_VER}

WORKDIR /app
CMD ["bash"]

