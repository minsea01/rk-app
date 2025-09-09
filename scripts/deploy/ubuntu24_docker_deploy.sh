#!/bin/bash
# Ubuntu 24.04 + RK3588 Docker部署脚本

set -e

RK3588_IP=${1:-"192.168.10.100"}
DEPLOY_USER=${2:-"root"}

echo "🐳 Ubuntu 24.04 + RK3588 Docker部署方案"
echo "目标设备: $DEPLOY_USER@$RK3588_IP"
echo "============================================"

# 第一步：构建Ubuntu 24.04兼容镜像
echo "[步骤1] 构建Ubuntu 24.04兼容镜像..."
docker build -f docker/ubuntu24-rknn-build.Dockerfile -t ubuntu24-rknn-build .
docker build -f docker/ubuntu24-rk3588-runtime.Dockerfile -t ubuntu24-rk3588-runtime .

# 第二步：在容器中交叉编译
echo "[步骤2] 容器化交叉编译..."
docker run --rm -v "$PWD":/work -w /work ubuntu24-rknn-build bash -c "
    cmake --preset arm64-release
    cmake --build --preset arm64-release
    cmake --install build/arm64
"

# 第三步：打包部署镜像
echo "[步骤3] 创建RK3588部署包..."
docker run --rm -v "$PWD":/work -w /work ubuntu24-rk3588-runtime bash -c "
    cp -r /work/out/arm64/* /app/
    cp -r /work/artifacts/models /app/
    cp -r /work/config /app/
    tar czf /work/rk3588-deploy.tar.gz -C /app .
"

# 第四步：传输并部署到RK3588
echo "[步骤4] 部署到RK3588设备..."
if ping -c 1 $RK3588_IP >/dev/null 2>&1; then
    echo "✅ 设备连通性检查通过"
    
    # 传输部署包
    scp rk3588-deploy.tar.gz $DEPLOY_USER@$RK3588_IP:/tmp/
    
    # 远程部署
    ssh $DEPLOY_USER@$RK3588_IP "
        # 创建应用目录
        mkdir -p /opt/rk-app
        cd /opt/rk-app
        
        # 解压部署包
        tar xzf /tmp/rk3588-deploy.tar.gz
        
        # 安装Docker（如果未安装）
        if ! command -v docker &> /dev/null; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            sh get-docker.sh
            systemctl enable docker
            systemctl start docker
        fi
        
        # 导入运行时镜像
        docker load < /tmp/ubuntu24-rk3588-runtime.tar || echo '镜像已存在'
        
        # 启动应用容器
        docker run -d --name rk-app-runtime \\
            --privileged \\
            --network host \\
            -v /opt/rk-app:/app \\
            -v /dev:/dev \\
            ubuntu24-rk3588-runtime \\
            /app/bin/rk_app --config /app/config/app.yaml
            
        echo '✅ RK3588应用容器启动完成'
    "
    
    echo "[完成] 部署成功！"
    echo "监控命令: ssh $DEPLOY_USER@$RK3588_IP 'docker logs -f rk-app-runtime'"
    
else
    echo "❌ 无法连接到RK3588设备: $RK3588_IP"
    echo "请检查网络连接和IP地址"
    exit 1
fi

# 第五步：验证部署
echo "[步骤5] 验证部署..."
ssh $DEPLOY_USER@$RK3588_IP "
    docker ps | grep rk-app-runtime
    docker exec rk-app-runtime /app/bin/rk_app --version
"

echo "🎉 Ubuntu 24.04 + RK3588 Docker部署完成！"