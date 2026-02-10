ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG RKNN_LITE2_VER=1.7.5

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-opencv python3-numpy \
    libdrm2 v4l-utils ethtool iperf3 \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-libav libgstreamer1.0-0 \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
      numpy Pillow opencv-python-headless \
      rknn-toolkit-lite2==${RKNN_LITE2_VER}

WORKDIR /app
EXPOSE 9000 9001
CMD ["bash"]

