FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgssapi-krb5-2 \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      ultralytics==8.3.193 \
      opencv-python-headless \
      albumentations \
      wandb \
      clearml \
      tensorboard \
      onnx \
      onnxsim

WORKDIR /work
CMD ["bash"]