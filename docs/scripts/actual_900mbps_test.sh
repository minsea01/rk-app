#!/bin/bash
# RK3588双千兆网口实际900Mbps吞吐量验证脚本
# 验证目标：每个网口实际能跑到≥900Mbps

echo "🚀 RK3588双千兆网口 ≥900Mbps 实际吞吐量验证"
echo "要求：eth0和eth1各自都能达到≥900Mbps"
echo "="*60

# 检查环境
check_environment() {
    echo "🔍 检查测试环境..."
    
    # 检查iperf3
    if ! command -v iperf3 >/dev/null 2>&1; then
        echo "❌ iperf3未安装，请安装: sudo apt install iperf3"
        exit 1
    fi
    
    # 检查网口
    for iface in eth0 eth1; do
        if [ ! -d "/sys/class/net/$iface" ]; then
            echo "⚠️ 网口 $iface 不存在"
        else
            state=$(cat /sys/class/net/$iface/operstate 2>/dev/null || echo "unknown")
            echo "📍 $iface 状态: $state"
        fi
    done
}

# 实际900Mbps测试方案
test_900mbps_capability() {
    echo ""
    echo "🎯 900Mbps吞吐量实际测试方案"
    echo "注意：需要在RK3588实际硬件环境中执行"
    echo ""
    
    echo "📋 测试准备步骤："
    echo ""
    echo "1. 准备测试设备："
    echo "   - RK3588开发板 (运行本脚本)"
    echo "   - 2台PC或服务器 (作为iperf3服务端)"
    echo "   - 千兆交换机"
    echo "   - Cat6网线"
    echo ""
    
    echo "2. 网络拓扑连接："
    echo "   PC1(192.168.1.100) ←→ 交换机 ←→ RK3588-eth0(192.168.1.10)"
    echo "   PC2(192.168.2.100) ←→ 交换机 ←→ RK3588-eth1(192.168.2.10)"
    echo ""
    
    echo "3. 在PC1上启动iperf3服务器："
    echo "   iperf3 -s -B 192.168.1.100"
    echo ""
    echo "4. 在PC2上启动iperf3服务器："
    echo "   iperf3 -s -B 192.168.2.100" 
    echo ""
    
    echo "5. RK3588测试命令："
    echo ""
    
    # 生成实际测试命令
    cat << 'TESTCMD'
# === RK3588上执行的实际测试命令 ===

# 1. 配置网口 (如果未配置)
sudo ip addr add 192.168.1.10/24 dev eth0
sudo ip addr add 192.168.2.10/24 dev eth1
sudo ip link set eth0 up
sudo ip link set eth1 up

# 2. 优化网口性能
sudo ethtool -G eth0 rx 4096 tx 4096
sudo ethtool -G eth1 rx 4096 tx 4096  
sudo ethtool -K eth0 tso on gso on
sudo ethtool -K eth1 tso on gso on

# 3. 测试eth0吞吐量 → PC1
echo "🧪 测试eth0吞吐量..."
iperf3 -c 192.168.1.100 -t 30 -i 5 -w 4M -P 4

# 预期结果示例:
# [SUM]   0.00-30.00  sec  3.31 GBytes   946 Mbits/sec  ✅
# 目标: ≥900 Mbits/sec

# 4. 测试eth1吞吐量 → PC2  
echo "🧪 测试eth1吞吐量..."
iperf3 -c 192.168.2.100 -t 30 -i 5 -w 4M -P 4

# 预期结果示例:
# [SUM]   0.00-30.00  sec  3.28 GBytes   934 Mbits/sec  ✅
# 目标: ≥900 Mbits/sec

# 5. 并发测试 (同时测试两个网口)
echo "🔥 双网口并发测试..."
iperf3 -c 192.168.1.100 -t 30 -w 4M -P 2 &
iperf3 -c 192.168.2.100 -t 30 -w 4M -P 2 &
wait

# 6. 验证结果
echo "📊 测试结果验证："
echo "如果看到："
echo "  eth0: ≥900 Mbits/sec ✅"  
echo "  eth1: ≥900 Mbits/sec ✅"
echo "则证明：双千兆网口实际吞吐量达标！"

TESTCMD
}

# 现在能做的验证 (非RK3588环境)
test_current_environment() {
    echo ""
    echo "💻 当前PC环境预验证："
    echo ""
    
    # 检查网卡能力
    for iface in $(ls /sys/class/net/ | grep -E "eth|ens|eno"); do
        echo "📍 检查网口: $iface"
        
        # 检查速度
        if command -v ethtool >/dev/null; then
            speed=$(ethtool $iface 2>/dev/null | grep "Speed:" | awk '{print $2}' || echo "unknown")
            duplex=$(ethtool $iface 2>/dev/null | grep "Duplex:" | awk '{print $2}' || echo "unknown")
            
            echo "  当前速度: $speed"
            echo "  双工模式: $duplex"
            
            # 检查支持的速度
            supported=$(ethtool $iface 2>/dev/null | grep -A20 "Supported link modes:")
            if echo "$supported" | grep -q "1000baseT/Full"; then
                echo "  ✅ 支持千兆全双工"
            else
                echo "  ⚠️ 千兆支持未确认"
            fi
        fi
        echo ""
    done
    
    # 本地回环测试最大吞吐量
    echo "🧪 本地回环最大吞吐量测试..."
    iperf3 -s -D  # 后台启动服务器
    sleep 2
    
    # 测试本地最大性能
    iperf3 -c localhost -t 10 -P 8 | grep -E "sender|receiver"
    
    # 停止后台服务器
    pkill iperf3
    
    echo ""
    echo "📊 预验证结论："
    echo "如果本地回环能跑到5-10Gbps，说明:"
    echo "✅ 系统网络栈性能充足"
    echo "✅ RK3588上运行900Mbps完全可行"
    echo "⚠️ 最终验证需要在RK3588实际硬件环境"
}

# 主函数
main() {
    check_environment
    test_900mbps_capability  
    test_current_environment
    
    echo ""
    echo "="*60
    echo "🎯 实际900Mbps验证结论："
    echo ""
    echo "❌ 当前测试条件不足："
    echo "  - 不是RK3588硬件环境"
    echo "  - 没有双网口物理连接"
    echo "  - 没有外部iperf3服务器"
    echo ""
    echo "✅ 技术方案完整："
    echo "  - RGMII驱动配置脚本已提供"
    echo "  - 网络优化参数已配置"
    echo "  - 实际测试命令已准备"
    echo ""
    echo "🚀 RK3588实际验证："
    echo "  1. 部署到RK3588: scp -r RK3588_Deploy/ user@rk3588:"
    echo "  2. 运行配置: sudo ./deploy.sh"
    echo "  3. 执行测试: sudo ./scripts/actual_900mbps_test.sh"
    echo "  4. 验证结果: eth0≥900Mbps, eth1≥900Mbps"
    echo ""
    echo "💡 预期结果: 基于千兆以太网标准，RK3588应该能达到950+Mbps"
    echo "="*60
}

# 执行主函数
main "$@"
