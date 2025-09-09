FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install runtime dependencies for RK3588
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-opencv python3-numpy \
    libdrm2 v4l-utils ethtool iperf3 \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-libav libgstreamer1.0-0 \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Install RKNN-Toolkit-Lite2 for device runtime
ARG RKNN_LITE2_VER=1.7.5
RUN pip3 install --no-cache-dir --break-system-packages \
      numpy Pillow opencv-python-headless \
      rknn-toolkit-lite2==${RKNN_LITE2_VER}

# Create app directory structure
RUN mkdir -p /app/{bin,lib,config,models,logs}

# Setup library path for RKNN runtime
ENV LD_LIBRARY_PATH=/app/lib:/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /app
EXPOSE 9000 9001

# Health check for RK3588 deployment
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep rk_app || exit 1

CMD ["bash"]