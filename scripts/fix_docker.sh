#!/bin/bash

# Docker构建修复脚本
# 解决WSL2环境下Docker buildx插件缺失的问题

set -e

echo "🔧 修复Docker构建环境..."

# 方案1: 使用Legacy Builder (推荐)
echo "📝 设置环境变量使用Legacy Builder"
echo 'export DOCKER_BUILDKIT=0' >> ~/.bashrc
export DOCKER_BUILDKIT=0

# 方案2: 安装独立的buildx (备用)
install_buildx() {
    echo "📥 下载并安装Docker Buildx..."
    
    # 获取最新版本
    BUILDX_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | cut -d'"' -f4 | tr -d 'v')
    
    if [ -z "$BUILDX_VERSION" ]; then
        BUILDX_VERSION="0.18.0"  # 备用版本
    fi
    
    echo "下载版本: $BUILDX_VERSION"
    
    # 下载buildx
    curl -Lo docker-buildx "https://github.com/docker/buildx/releases/download/v${BUILDX_VERSION}/buildx-v${BUILDX_VERSION}.linux-amd64"
    
    # 安装
    chmod +x docker-buildx
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo mv docker-buildx /usr/local/lib/docker/cli-plugins/docker-buildx
    
    echo "✅ Buildx 安装完成"
}

# 检查当前状态
echo "🔍 检查Docker环境..."
docker --version

if docker buildx version &>/dev/null; then
    echo "✅ Docker Buildx 可用"
else
    echo "⚠️ Docker Buildx 不可用，使用Legacy Builder"
fi

# 测试构建
echo "🧪 测试Docker构建功能..."

# 创建简单测试Dockerfile
cat > /tmp/test.Dockerfile << 'EOF'
FROM ubuntu:20.04
RUN echo "Docker build test successful"
EOF

if docker build -f /tmp/test.Dockerfile -t docker-test . &>/dev/null; then
    echo "✅ Docker构建功能正常"
    docker rmi docker-test &>/dev/null || true
    rm /tmp/test.Dockerfile
else
    echo "❌ Docker构建失败"
    rm /tmp/test.Dockerfile
    exit 1
fi

echo "🎉 Docker环境修复完成!"
echo ""
echo "📋 使用方法:"
echo "1. 直接构建: docker build -f Dockerfile -t image:tag ."
echo "2. 如需buildx: 运行 install_buildx 函数"
echo "3. 环境变量已设置: DOCKER_BUILDKIT=0"
echo ""
echo "🚀 现在可以正常使用Docker构建了!"