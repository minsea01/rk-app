#!/bin/bash
set -euo pipefail

# RK3588 Docker 部署脚本

echo "=== RK3588 Docker 部署 ==="

# 步骤 1: 在 PC 上构建 ARM64 镜像
build_image() {
    echo "步骤 1: 构建 ARM64 镜像..."

    # 使用 buildx 支持跨平台构建
    docker buildx create --use --name arm64-builder || true

    docker buildx build \
        --platform linux/arm64 \
        -t rk3588-person-detector:latest \
        -f Dockerfile.rk3588 \
        --load \
        .

    echo "✅ 镜像构建完成"
}

# 步骤 2: 保存镜像到 tar
save_image() {
    echo "步骤 2: 保存镜像..."
    docker save rk3588-person-detector:latest -o rk3588-detector.tar
    echo "✅ 镜像已保存到: rk3588-detector.tar"
}

# 步骤 3: 传输到 RK3588
transfer_to_board() {
    local BOARD_IP=${1:-192.168.1.100}

    echo "步骤 3: 传输到 RK3588 ($BOARD_IP)..."

    # 传输镜像
    scp rk3588-detector.tar root@$BOARD_IP:/tmp/

    # 传输 docker-compose 文件
    scp docker-compose.rk3588.yml root@$BOARD_IP:/root/

    echo "✅ 文件已传输"
}

# 步骤 4: 在板子上加载镜像
load_on_board() {
    local BOARD_IP=${1:-192.168.1.100}

    echo "步骤 4: 在板子上加载镜像..."

    ssh root@$BOARD_IP << 'EOF'
        # 加载镜像
        docker load -i /tmp/rk3588-detector.tar

        # 验证
        docker images | grep rk3588-person-detector

        echo "✅ 镜像已加载"
EOF
}

# 步骤 5: 启动容器
start_container() {
    local BOARD_IP=${1:-192.168.1.100}

    echo "步骤 5: 启动容器..."

    ssh root@$BOARD_IP << 'EOF'
        cd /root

        # 启动容器
        docker-compose -f docker-compose.rk3588.yml up -d

        # 查看日志
        docker-compose -f docker-compose.rk3588.yml logs

        echo "✅ 容器已启动"
EOF
}

# 主流程
main() {
    local BOARD_IP=${1:-}

    if [ -z "$BOARD_IP" ]; then
        echo "用法: $0 <RK3588_IP地址>"
        echo "例如: $0 192.168.1.100"
        exit 1
    fi

    build_image
    save_image
    transfer_to_board "$BOARD_IP"
    load_on_board "$BOARD_IP"
    start_container "$BOARD_IP"

    echo ""
    echo "=== 部署完成! ==="
    echo "在板子上运行以下命令查看状态:"
    echo "  docker ps"
    echo "  docker logs rk3588-detector"
}

main "$@"
