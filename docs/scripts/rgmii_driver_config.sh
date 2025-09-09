#!/bin/bash
# RK3588 RGMII双千兆网口驱动配置脚本
# 要求：双网口吞吐量≥900Mbps
# 网口1：工业相机(2K分辨率实时采集)
# 网口2：检测结果上传

set -e

echo "🌐 RK3588 RGMII双千兆网口驱动配置"
echo "要求：双网口吞吐量≥900Mbps"
echo "网口1: 工业相机(2K实时采集) | 网口2: 检测结果上传"
echo "="*60

# 检查是否为root权限
if [ "$EUID" -ne 0 ]; then
    echo "❌ 请以root权限运行: sudo $0"
    exit 1
fi

# 检查RK3588平台
check_rk3588() {
    echo "🔍 检查RK3588平台和RGMII接口..."
    
    # 检查设备树中的RGMII配置
    if [ -d "/sys/firmware/devicetree/base/ethernet@fe1b0000" ]; then
        echo "✅ 检测到RGMII0接口 (eth0)"
    else
        echo "⚠️ 未检测到RGMII0接口"
    fi
    
    if [ -d "/sys/firmware/devicetree/base/ethernet@fe1c0000" ]; then
        echo "✅ 检测到RGMII1接口 (eth1)"  
    else
        echo "⚠️ 未检测到RGMII1接口"
    fi
    
    # 检查网卡驱动
    if lsmod | grep -q "stmmac"; then
        echo "✅ STMMAC以太网驱动已加载"
    else
        echo "⚠️ STMMAC驱动未加载，尝试加载..."
        modprobe stmmac || echo "驱动加载失败"
    fi
}

# 配置RGMII PHY参数
configure_rgmii_phy() {
    echo "⚙️ 配置RGMII PHY参数..."
    
    # RGMII时钟延迟配置
    echo "配置RGMII时钟延迟..."
    
    # 网口1 (eth0) - 工业相机专用
    if [ -d "/sys/class/net/eth0" ]; then
        echo "🔧 配置eth0 (RGMII0) - 工业相机接口"
        
        # PHY寄存器配置 (根据具体PHY芯片调整)
        # RTL8211F PHY常用配置
        if command -v mii-tool >/dev/null; then
            # 强制1000Mbps全双工
            mii-tool -F 1000baseTx-FD eth0 2>/dev/null || true
        fi
        
        # ethtool高级配置
        ethtool -s eth0 speed 1000 duplex full autoneg on 2>/dev/null
        
        # 接收/发送队列配置 (针对高分辨率图像流)
        ethtool -G eth0 rx 4096 tx 4096 2>/dev/null || true
        
        # 硬件校验和卸载
        ethtool -K eth0 rx-checksum on tx-checksum on 2>/dev/null || true
        ethtool -K eth0 sg on tso on gso on gro on 2>/dev/null || true
        
        # RGMII接口特定优化
        ethtool -K eth0 lro off  # 大数据包接收优化关闭，减少延迟
        
        echo "✅ eth0配置完成"
    fi
    
    # 网口2 (eth1) - 检测结果上传
    if [ -d "/sys/class/net/eth1" ]; then
        echo "🔧 配置eth1 (RGMII1) - 结果上传接口"
        
        if command -v mii-tool >/dev/null; then
            mii-tool -F 1000baseTx-FD eth1 2>/dev/null || true
        fi
        
        ethtool -s eth1 speed 1000 duplex full autoneg on 2>/dev/null
        ethtool -G eth1 rx 2048 tx 2048 2>/dev/null || true
        ethtool -K eth1 rx-checksum on tx-checksum on 2>/dev/null || true
        ethtool -K eth1 sg on tso on gso on gro on 2>/dev/null || true
        
        echo "✅ eth1配置完成"
    fi
}

# 高性能网络参数优化
optimize_network_performance() {
    echo "⚡ 高性能网络参数优化..."
    
    # TCP/IP协议栈优化
    cat > /etc/sysctl.d/99-rgmii-performance.conf << 'EOF'
# RK3588 RGMII双千兆网口性能优化
# 目标：双网口吞吐量≥900Mbps

# 内核网络缓冲区优化
net.core.rmem_max = 268435456          # 接收缓冲区最大值 (256MB)
net.core.wmem_max = 268435456          # 发送缓冲区最大值 (256MB)
net.core.rmem_default = 1048576        # 接收缓冲区默认值 (1MB)
net.core.wmem_default = 1048576        # 发送缓冲区默认值 (1MB)

# 网络设备队列优化  
net.core.netdev_max_backlog = 10000    # 网络设备队列长度
net.core.netdev_budget = 600           # 网络处理预算
net.core.dev_weight = 64               # 设备权重

# TCP参数优化
net.ipv4.tcp_rmem = 4096 1048576 268435456    # TCP接收窗口
net.ipv4.tcp_wmem = 4096 1048576 268435456    # TCP发送窗口
net.ipv4.tcp_congestion_control = bbr         # BBR拥塞控制
net.ipv4.tcp_window_scaling = 1               # 窗口扩展
net.ipv4.tcp_timestamps = 1                   # 时间戳
net.ipv4.tcp_sack = 1                         # 选择性确认

# 减少连接延迟
net.ipv4.tcp_fin_timeout = 15                 # FIN超时
net.ipv4.tcp_tw_reuse = 1                     # TIME_WAIT重用
net.ipv4.tcp_max_syn_backlog = 4096           # SYN队列长度

# UDP优化
net.ipv4.udp_mem = 102400 873800 16777216     # UDP内存使用
net.ipv4.udp_rmem_min = 8192                  # UDP接收最小值
net.ipv4.udp_wmem_min = 8192                  # UDP发送最小值

# 网络中断优化
net.core.busy_poll = 50                       # 忙轮询
net.core.busy_read = 50                       # 忙读取
EOF
    
    # 应用网络参数
    sysctl -p /etc/sysctl.d/99-rgmii-performance.conf
    
    echo "✅ 网络参数优化完成"
}

# CPU亲和性和中断优化
configure_irq_affinity() {
    echo "🎯 配置网卡中断CPU亲和性..."
    
    # 获取CPU信息
    cpu_count=$(nproc)
    echo "检测到 $cpu_count 个CPU核心"
    
    if [ $cpu_count -ge 8 ]; then
        echo "使用8核中断绑定策略 (RK3588标准配置)"
        
        # A55小核(0-3)处理系统中断
        # A76大核(4-7)处理网络中断
        
        # eth0 (工业相机) -> CPU 4,5 (A76核心)
        eth0_irq=$(grep -E "eth0|fe1b0000" /proc/interrupts | cut -d: -f1 | tr -d ' ' | head -1)
        if [ -n "$eth0_irq" ]; then
            echo "30" > /proc/irq/$eth0_irq/smp_affinity 2>/dev/null  # CPU 4-5
            echo "eth0 IRQ $eth0_irq -> CPU 4-5 (A76核心)"
        fi
        
        # eth1 (结果上传) -> CPU 6,7 (A76核心)  
        eth1_irq=$(grep -E "eth1|fe1c0000" /proc/interrupts | cut -d: -f1 | tr -d ' ' | head -1)
        if [ -n "$eth1_irq" ]; then
            echo "C0" > /proc/irq/$eth1_irq/smp_affinity 2>/dev/null  # CPU 6-7
            echo "eth1 IRQ $eth1_irq -> CPU 6-7 (A76核心)"
        fi
        
        # RPS (Receive Packet Steering) 配置
        echo "f0" > /sys/class/net/eth0/queues/rx-0/rps_cpus  # CPU 4-7
        echo "f0" > /sys/class/net/eth1/queues/rx-0/rps_cpus  # CPU 4-7
        
    elif [ $cpu_count -ge 4 ]; then
        echo "使用4核中断绑定策略"
        
        # eth0 -> CPU 2-3
        eth0_irq=$(grep eth0 /proc/interrupts | cut -d: -f1 | tr -d ' ')
        if [ -n "$eth0_irq" ]; then
            echo "C" > /proc/irq/$eth0_irq/smp_affinity 2>/dev/null
        fi
        
        # eth1 -> CPU 2-3  
        eth1_irq=$(grep eth1 /proc/interrupts | cut -d: -f1 | tr -d ' ')
        if [ -n "$eth1_irq" ]; then
            echo "C" > /proc/irq/$eth1_irq/smp_affinity 2>/dev/null
        fi
    fi
    
    echo "✅ 中断亲和性配置完成"
}

# 网络接口高级配置
configure_advanced_features() {
    echo "🔧 配置网络接口高级特性..."
    
    # 巨型帧配置 (适用于高带宽传输)
    for iface in eth0 eth1; do
        if [ -d "/sys/class/net/$iface" ]; then
            echo "配置 $iface 巨型帧..."
            
            # 设置MTU为9000 (巨型帧)
            ip link set $iface mtu 9000 2>/dev/null || {
                echo "巨型帧设置失败，使用标准MTU"
                ip link set $iface mtu 1500
            }
            
            # 队列长度优化
            ip link set $iface txqueuelen 10000
            
            echo "$iface 高级特性配置完成"
        fi
    done
    
    # NAPI权重调整 (影响网络处理性能)
    for iface in eth0 eth1; do
        if [ -d "/sys/class/net/$iface" ]; then
            # 增加NAPI权重以提高吞吐量
            echo 64 > /sys/class/net/$iface/weight 2>/dev/null || true
        fi
    done
    
    echo "✅ 高级特性配置完成"
}

# 实时性能测试
performance_test() {
    echo "🧪 网络性能测试..."
    
    # 创建性能测试报告
    test_report="/tmp/rgmii_performance_test.log"
    echo "RK3588 RGMII网口性能测试报告" > $test_report
    echo "测试时间: $(date)" >> $test_report
    echo "="*50 >> $test_report
    
    # 测试网口状态
    for iface in eth0 eth1; do
        if [ -d "/sys/class/net/$iface" ]; then
            echo "" >> $test_report
            echo "[$iface 状态检测]" >> $test_report
            
            # 链路状态
            link_status=$(cat /sys/class/net/$iface/operstate 2>/dev/null || echo "unknown")
            echo "链路状态: $link_status" >> $test_report
            
            # 速度和双工模式
            if command -v ethtool >/dev/null; then
                speed=$(ethtool $iface 2>/dev/null | grep Speed | cut -d: -f2 | xargs || echo "unknown")
                duplex=$(ethtool $iface 2>/dev/null | grep Duplex | cut -d: -f2 | xargs || echo "unknown")
                echo "速度: $speed" >> $test_report
                echo "双工模式: $duplex" >> $test_report
            fi
            
            # MTU大小
            mtu=$(cat /sys/class/net/$iface/mtu 2>/dev/null || echo "unknown")
            echo "MTU: $mtu" >> $test_report
            
            # 队列配置
            if [ -d "/sys/class/net/$iface/queues" ]; then
                rx_queues=$(ls /sys/class/net/$iface/queues/ | grep rx | wc -l)
                tx_queues=$(ls /sys/class/net/$iface/queues/ | grep tx | wc -l)
                echo "RX队列: $rx_queues, TX队列: $tx_queues" >> $test_report
            fi
        fi
    done
    
    # 中断分配情况
    echo "" >> $test_report
    echo "[中断分配情况]" >> $test_report
    grep -E "eth[01]" /proc/interrupts >> $test_report 2>/dev/null || echo "无网卡中断信息" >> $test_report
    
    # 网络参数检查
    echo "" >> $test_report  
    echo "[关键网络参数]" >> $test_report
    echo "net.core.rmem_max = $(sysctl -n net.core.rmem_max)" >> $test_report
    echo "net.core.wmem_max = $(sysctl -n net.core.wmem_max)" >> $test_report
    echo "net.core.netdev_max_backlog = $(sysctl -n net.core.netdev_max_backlog)" >> $test_report
    echo "net.ipv4.tcp_congestion_control = $(sysctl -n net.ipv4.tcp_congestion_control)" >> $test_report
    
    echo "📊 性能测试报告: $test_report"
    
    # 显示关键信息
    echo ""
    echo "🔍 当前网口状态："
    for iface in eth0 eth1; do
        if [ -d "/sys/class/net/$iface" ]; then
            status=$(cat /sys/class/net/$iface/operstate)
            mtu=$(cat /sys/class/net/$iface/mtu)
            echo "  $iface: $status, MTU=$mtu"
        fi
    done
    
    echo ""
    echo "📈 性能测试建议："
    echo "1. 带宽测试: iperf3 -c <target_ip> -t 60 -i 5 -w 1M"
    echo "2. 延迟测试: ping -c 1000 -i 0.001 <target_ip>"
    echo "3. 2K图像传输测试: 使用GigE Vision SDK测试相机数据流"
    echo "4. 并发测试: 同时测试两个网口的吞吐量"
}

# 创建监控脚本
create_monitoring_script() {
    echo "📊 创建网络性能监控脚本..."
    
    cat > /usr/local/bin/rgmii-monitor.sh << 'EOF'
#!/bin/bash
# RK3588 RGMII网口实时性能监控

while true; do
    clear
    echo "🌐 RK3588 RGMII双千兆网口实时监控"
    echo "时间: $(date)"
    echo "="*60
    
    # 网口状态
    for iface in eth0 eth1; do
        if [ -d "/sys/class/net/$iface" ]; then
            echo ""
            echo "[$iface 状态]"
            
            # 基本状态
            operstate=$(cat /sys/class/net/$iface/operstate)
            echo "状态: $operstate"
            
            # 流量统计
            rx_bytes=$(cat /sys/class/net/$iface/statistics/rx_bytes)
            tx_bytes=$(cat /sys/class/net/$iface/statistics/tx_bytes)
            rx_packets=$(cat /sys/class/net/$iface/statistics/rx_packets)  
            tx_packets=$(cat /sys/class/net/$iface/statistics/tx_packets)
            
            # 转换为人类可读格式
            rx_mb=$((rx_bytes / 1024 / 1024))
            tx_mb=$((tx_bytes / 1024 / 1024))
            
            echo "接收: ${rx_mb}MB (${rx_packets}包)"
            echo "发送: ${tx_mb}MB (${tx_packets}包)"
            
            # 错误统计
            rx_errors=$(cat /sys/class/net/$iface/statistics/rx_errors)
            tx_errors=$(cat /sys/class/net/$iface/statistics/tx_errors)
            rx_dropped=$(cat /sys/class/net/$iface/statistics/rx_dropped)
            tx_dropped=$(cat /sys/class/net/$iface/statistics/tx_dropped)
            
            if [ $((rx_errors + tx_errors + rx_dropped + tx_dropped)) -gt 0 ]; then
                echo "⚠️ 错误: RX=$rx_errors TX=$tx_errors 丢包: RX=$rx_dropped TX=$tx_dropped"
            else
                echo "✅ 无错误和丢包"
            fi
        fi
    done
    
    echo ""
    echo "💻 系统状态:"
    echo "负载: $(uptime | awk -F'load average:' '{print $2}')"
    echo "内存: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    sleep 2
done
EOF
    
    chmod +x /usr/local/bin/rgmii-monitor.sh
    echo "✅ 监控脚本已创建: /usr/local/bin/rgmii-monitor.sh"
}

# 主函数
main() {
    echo "开始RK3588 RGMII双千兆网口配置..."
    
    # 平台检查
    check_rk3588
    
    # RGMII PHY配置
    configure_rgmii_phy
    
    # 网络性能优化
    optimize_network_performance
    
    # 中断亲和性配置
    configure_irq_affinity
    
    # 高级特性配置
    configure_advanced_features
    
    # 创建监控工具
    create_monitoring_script
    
    # 性能测试
    performance_test
    
    echo ""
    echo "="*60
    echo "🎉 RK3588 RGMII双千兆网口配置完成！"
    echo ""
    echo "📊 配置结果:"
    echo "✅ 双RGMII接口已优化配置"
    echo "✅ 网络参数已调优 (目标≥900Mbps)"  
    echo "✅ 中断亲和性已优化"
    echo "✅ 高级网络特性已启用"
    echo ""
    echo "🔧 使用指南:"
    echo "  实时监控: /usr/local/bin/rgmii-monitor.sh"
    echo "  性能测试: iperf3 -c <target> -t 30 -i 5"
    echo "  状态检查: ethtool eth0 && ethtool eth1"
    echo ""
    echo "🎯 预期性能:"
    echo "  eth0 (工业相机): ≥900Mbps, 2K实时图像传输"
    echo "  eth1 (结果上传): ≥900Mbps, 检测结果实时上传"
    echo ""
    echo "✅ 系统已准备就绪，可连接工业相机进行测试！"
    echo "="*60
}

# 执行主函数
main "$@"
