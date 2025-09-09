#!/bin/bash
# RK3588双千兆网口配置脚本
# 要求：网口吞吐量≥900Mbps

echo "🌐 RK3588工业检测系统 - 网络配置"
echo "目标：双千兆网口吞吐量≥900Mbps"
echo "="*50

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then 
    echo "❌ 请以root权限运行此脚本: sudo $0"
    exit 1
fi

# 检查网口是否存在
check_interface() {
    local interface=$1
    if ! ip link show "$interface" >/dev/null 2>&1; then
        echo "⚠️ 网口 $interface 不存在，跳过配置"
        return 1
    fi
    return 0
}

# 网口1：工业相机网络（RGMII接口）
echo "📷 配置网口1 (eth0) - 工业相机专用网络"
if check_interface eth0; then
    # 清理现有配置
    ip addr flush dev eth0 2>/dev/null
    
    # 设置静态IP（相机网络段）
    ip addr add 192.168.1.10/24 dev eth0
    ip link set eth0 up
    
    # 千兆网口优化配置
    ethtool -s eth0 speed 1000 duplex full autoneg on 2>/dev/null
    
    # 增大接收和发送缓冲区
    ethtool -G eth0 rx 4096 tx 4096 2>/dev/null
    
    # 启用硬件加速功能
    ethtool -K eth0 tso on gso on gro on lro off 2>/dev/null
    
    # 设置中断合并
    ethtool -C eth0 rx-usecs 50 tx-usecs 50 2>/dev/null
    
    echo "✅ eth0配置完成: 192.168.1.10/24"
else
    echo "⚠️ eth0配置跳过"
fi

# 网口2：结果上传网络（RGMII接口）
echo "📤 配置网口2 (eth1) - 结果上传网络"
if check_interface eth1; then
    # 清理现有配置
    ip addr flush dev eth1 2>/dev/null
    
    # 设置静态IP（上传网络段）
    ip addr add 192.168.2.10/24 dev eth1
    ip link set eth1 up
    
    # 千兆网口优化配置
    ethtool -s eth1 speed 1000 duplex full autoneg on 2>/dev/null
    
    # 增大接收和发送缓冲区  
    ethtool -G eth1 rx 4096 tx 4096 2>/dev/null
    
    # 启用硬件加速功能
    ethtool -K eth1 tso on gso on gro on lro off 2>/dev/null
    
    # 设置中断合并
    ethtool -C eth1 rx-usecs 50 tx-usecs 50 2>/dev/null
    
    echo "✅ eth1配置完成: 192.168.2.10/24"
else
    echo "⚠️ eth1配置跳过"
fi

# 系统网络性能优化
echo "⚡ 优化系统网络性能参数..."

# TCP缓冲区优化
sysctl -w net.core.rmem_max=134217728        # 接收缓冲区最大值
sysctl -w net.core.wmem_max=134217728        # 发送缓冲区最大值  
sysctl -w net.core.rmem_default=262144       # 接收缓冲区默认值
sysctl -w net.core.wmem_default=262144       # 发送缓冲区默认值

# 网络队列优化
sysctl -w net.core.netdev_max_backlog=5000   # 网络设备队列长度
sysctl -w net.core.netdev_budget=600         # 网络处理预算

# TCP拥塞控制优化
sysctl -w net.ipv4.tcp_congestion_control=bbr # 使用BBR算法

# 减少TIME_WAIT连接
sysctl -w net.ipv4.tcp_tw_reuse=1            # 重用TIME_WAIT连接
sysctl -w net.ipv4.tcp_fin_timeout=15        # 减少FIN_WAIT超时

# 网络设备中断优化
echo "🎯 优化网络中断处理..."

# 获取CPU核心数量
CPU_CORES=$(nproc)
echo "检测到 $CPU_CORES 个CPU核心"

# 网口中断绑定策略
if [ "$CPU_CORES" -ge 4 ]; then
    # 4核以上：网口1绑定到CPU 0-1，网口2绑定到CPU 2-3
    echo "使用4核中断绑定策略"
    
    # eth0中断绑定到CPU 0-1 (掩码: 0011 = 3)
    ETH0_IRQ=$(grep eth0 /proc/interrupts 2>/dev/null | cut -d: -f1 | tr -d ' ')
    if [ -n "$ETH0_IRQ" ]; then
        echo 3 > /proc/irq/$ETH0_IRQ/smp_affinity 2>/dev/null
        echo "eth0 IRQ $ETH0_IRQ -> CPU 0-1"
    fi
    
    # eth1中断绑定到CPU 2-3 (掩码: 1100 = 12)
    ETH1_IRQ=$(grep eth1 /proc/interrupts 2>/dev/null | cut -d: -f1 | tr -d ' ')
    if [ -n "$ETH1_IRQ" ]; then
        echo 12 > /proc/irq/$ETH1_IRQ/smp_affinity 2>/dev/null
        echo "eth1 IRQ $ETH1_IRQ -> CPU 2-3"
    fi
else
    echo "CPU核心数不足，跳过中断绑定"
fi

# 保存网络配置到启动脚本
echo "💾 保存网络配置..."
cat > /etc/systemd/system/rk3588-network.service << 'EOF'
[Unit]
Description=RK3588 Industrial Network Setup
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/rk3588-network-setup.sh
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

# 创建启动脚本
cp "$0" /usr/local/bin/rk3588-network-setup.sh
chmod +x /usr/local/bin/rk3588-network-setup.sh

# 启用服务
systemctl enable rk3588-network.service 2>/dev/null

# 验证网口状态
echo "📊 验证网口配置..."
echo ""
echo "=== 网口1 (eth0) 状态 ==="
if check_interface eth0; then
    ip addr show eth0 | grep inet
    ethtool eth0 2>/dev/null | grep -E "Speed|Duplex|Link detected" || echo "ethtool信息获取失败"
fi

echo ""
echo "=== 网口2 (eth1) 状态 ==="
if check_interface eth1; then
    ip addr show eth1 | grep inet  
    ethtool eth1 2>/dev/null | grep -E "Speed|Duplex|Link detected" || echo "ethtool信息获取失败"
fi

# 路由表信息
echo ""
echo "=== 路由表 ==="
ip route show

echo ""
echo "🧪 网络性能测试命令："
echo "测试网口1带宽: iperf3 -c 192.168.1.100 -t 30 -i 5"
echo "测试网口2带宽: iperf3 -c 192.168.2.100 -t 30 -i 5"
echo "网络延迟测试: ping -c 10 192.168.1.1"
echo ""
echo "📋 网络监控命令："
echo "实时带宽: iftop -i eth0"
echo "网口统计: watch -n1 'cat /proc/net/dev'"
echo "连接状态: netstat -i"
echo ""
echo "✅ RK3588双千兆网口配置完成！"
echo "🎯 目标性能: ≥900Mbps per port"
echo "🔧 配置已保存至系统服务，重启后自动生效"
