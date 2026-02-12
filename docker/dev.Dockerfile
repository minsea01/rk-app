# Multi-stage Dockerfile for RK3588 Pedestrian Detection System
# Supports both x86 development and ARM64 production deployment

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} as base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Configure APT mirror for base stage (using HTTP to avoid VPN SSL issues)
RUN sed -i 's|http://ports.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libopencv-dev \
    libyaml-cpp-dev \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies in virtual environment
RUN python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"

# ======================
# Development Stage
# ======================
FROM base as development

# Install development tools
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements-dev.txt

# Install additional dev tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    strace \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /app

# Expose ports for development
EXPOSE 8888 5201 8000

CMD ["/bin/bash"]

# ======================
# Build Stage (C++)
# ======================
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    libopencv-dev \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /app

# Build C++ components
RUN cmake --preset x86-release && \
    cmake --build build/x86 --parallel $(nproc) && \
    cmake --install build/x86

# ======================
# Production Stage (Python)
# ======================
FROM base as production-python

# Copy only necessary files
COPY apps/ /app/apps/
COPY tools/ /app/tools/
COPY config/ /app/config/
COPY artifacts/models/ /app/artifacts/models/
COPY scripts/deploy/ /app/scripts/deploy/

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import apps.config; print('OK')" || exit 1

# Default command
CMD ["python3", "-m", "apps.yolov8_rknn_infer", "--help"]

# ======================
# Production Stage (C++)
# ======================
FROM base as production-cpp

# Copy built binaries from builder
COPY --from=builder /app/out/x86/ /app/out/x86/

# Copy config files
COPY config/ /app/config/

# Set library path
ENV LD_LIBRARY_PATH=/app/out/x86/lib:$LD_LIBRARY_PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/out/x86/bin/detect_cli --help > /dev/null || exit 1

# Default command
CMD ["/app/out/x86/bin/detect_cli", "--help"]

# ======================
# ARM64 Cross-compile Stage
# ======================
# Use build platform (x86) to cross-compile, avoiding QEMU issues
FROM --platform=$BUILDPLATFORM ubuntu:22.04 as arm64-builder

ENV DEBIAN_FRONTEND=noninteractive

# Configure APT mirror for builder stage (using HTTP to avoid VPN SSL issues)
RUN sed -i 's|http://ports.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# Install cross-compilation toolchain
RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Cross-compile for ARM64
RUN cmake --preset arm64-release && \
    cmake --build build/arm64 --parallel $(nproc)

# ======================
# RK3588 Runtime (ARM64)
# ======================
FROM arm64v8/ubuntu:22.04 as rk3588-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Configure APT mirror (use Tsinghua mirror for better connectivity in China)
# Using HTTP to avoid VPN SSL certificate issues
RUN sed -i 's|http://ports.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libopencv-core4.5d \
    libyaml-cpp0.7 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ARM64 binaries
COPY --from=arm64-builder /app/out/arm64/ /app/out/arm64/

# Copy Python code
COPY apps/ /app/apps/
COPY config/ /app/config/
COPY artifacts/models/ /app/artifacts/models/
COPY scripts/deploy/rk3588_run.sh /app/scripts/deploy/

# Configure pip mirror (use Tsinghua mirror for faster download)
# Using HTTP to avoid VPN SSL certificate issues
RUN pip3 config set global.index-url http://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# Install RKNN runtime library (rknn-toolkit-lite2)
RUN pip3 install --no-cache-dir rknn-toolkit-lite2

# Set environment
ENV LD_LIBRARY_PATH=/app/out/arm64/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/app
ENV PATH="/app/out/arm64/bin:$PATH"

# Default entry point
CMD ["/app/scripts/deploy/rk3588_run.sh"]
