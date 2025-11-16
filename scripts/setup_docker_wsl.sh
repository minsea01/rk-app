#!/usr/bin/env bash
set -euo pipefail

# WSL上安装Docker并配置buildx（用于ARM64交叉构建）

echo "=========================================="
echo "Installing Docker on WSL"
echo "=========================================="

# 检查是否在WSL中
if ! grep -qi microsoft /proc/version; then
    echo "⚠️  警告: 似乎不在WSL环境中"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 卸载旧版本（如果有）
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# 更新apt
sudo apt-get update

# 安装依赖
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 添加Docker GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 添加Docker仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 启动Docker服务
sudo service docker start

# 添加当前用户到docker组（避免每次sudo）
sudo usermod -aG docker $USER

# 验证安装
echo ""
echo "=========================================="
echo "验证Docker安装..."
echo "=========================================="
sudo docker --version
sudo docker buildx version

# 配置buildx for ARM64
echo ""
echo "=========================================="
echo "配置buildx多架构支持..."
echo "=========================================="
sudo docker run --privileged --rm tonistiigi/binfmt --install all
sudo docker buildx create --name multiarch --driver docker-container --use
sudo docker buildx inspect --bootstrap

echo ""
echo "=========================================="
echo "✅ Docker安装完成"
echo "=========================================="
echo ""
echo "注意: 需要重新登录shell才能让docker组生效"
echo "或者运行: newgrp docker"
echo ""
echo "测试命令:"
echo "  docker run hello-world"
echo "  docker buildx build --platform linux/arm64 -t test ."
