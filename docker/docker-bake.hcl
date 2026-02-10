group "default" {
  targets = ["train-cu121", "rknn-build-ubuntu22", "rk3588-runtime-ubuntu22"]
}

target "train-base" {
  dockerfile = "docker/train.Dockerfile"
  context = "."
}

target "train-cu121" {
  inherits = ["train-base"]
  tags = ["rk-app/train:cu121"]
  args = {
    BASE_IMAGE      = "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"
    TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
  }
}

target "train-nvcr" {
  inherits = ["train-base"]
  tags = ["rk-app/train:nvcr"]
  args = {
    BASE_IMAGE      = "nvcr.io/nvidia/pytorch:24.03-py3"
    TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
  }
}

target "rknn-build-base" {
  dockerfile = "docker/rknn-build.Dockerfile"
  context = "."
}

target "rknn-build-ubuntu22" {
  inherits = ["rknn-build-base"]
  tags = ["rk-app/rknn-build:ubuntu22"]
  args = {
    BASE_IMAGE = "ubuntu:22.04"
  }
}

target "rknn-build-ubuntu24" {
  inherits = ["rknn-build-base"]
  tags = ["rk-app/rknn-build:ubuntu24"]
  args = {
    BASE_IMAGE = "ubuntu:24.04"
  }
}

target "rk3588-runtime-base" {
  dockerfile = "docker/rk3588-runtime.Dockerfile"
  context = "."
}

target "rk3588-runtime-ubuntu22" {
  inherits = ["rk3588-runtime-base"]
  tags = ["rk-app/rk3588-runtime:ubuntu22"]
  args = {
    BASE_IMAGE = "ubuntu:22.04"
  }
}

target "rk3588-runtime-ubuntu24" {
  inherits = ["rk3588-runtime-base"]
  tags = ["rk-app/rk3588-runtime:ubuntu24"]
  args = {
    BASE_IMAGE = "ubuntu:24.04"
  }
}

