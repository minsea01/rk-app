#!/bin/bash
# RK3588双千兆网口吞吐量验证脚本
# 严格验证：双网口吞吐量≥900Mbps
# 网口1: 工业相机2K图像流 | 网口2: 检测结果上传

set -e

echo "🌐 RK3588双千兆网口吞吐量验证"
echo "严格要求: 双网口吞吐量≥900Mbps"
echo "应用场景: 网口1(2K相机流) + 网口2(结果上传)"
echo "="*60

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 测试结果变量
ETH0_RESULT=0
ETH1_RESULT=0
DUAL_PORT_RESULT=0
TEST_REPORT="/tmp/network_throughput_report.log"

# 检查测试环境
check_test_environment() {
    log_info "检查网络测试环境..."
    
    # 检查iperf3工具
    if ! command -v iperf3 >/dev/null 2>&1; then
        log_error "iperf3工具未安装"
        echo "请安装: sudo apt install iperf3"
        exit 1
    fi
    
    # 检查网口状态
    for iface in eth0 eth1; do
        if [ ! -d "/sys/class/net/$iface" ]; then
            log_error "网口 $iface 不存在"
            exit 1
        fi
        
        # 检查链路状态
        link_state=$(cat /sys/class/net/$iface/operstate 2>/dev/null || echo "unknown")
        if [ "$link_state" != "up" ]; then
            log_warning "网口 $iface 状态: $link_state (建议连接网线)"
        else
            log_success "网口 $iface 状态: $link_state"
        fi
        
        # 检查网口速度
        if command -v ethtool >/dev/null; then
            speed=$(ethtool $iface 2>/dev/null | grep "Speed:" | awk '{print $2}' || echo "unknown")
            duplex=$(ethtool $iface 2>/dev/null | grep "Duplex:" | awk '{print $2}' || echo "unknown")
            log_info "$iface: $speed, $duplex"
            
            if [[ "$speed" != "1000Mb/s" ]]; then
                log_warning "$iface 未运行在千兆模式: $speed"
            fi
        fi
    done
    
    # 创建测试报告文件
    cat > $TEST_REPORT << EOF
RK3588双千兆网口吞吐量测试报告
========================================
测试时间: $(date)
测试要求: 双网口吞吐量≥900Mbps
应用场景: 网口1(2K相机流) + 网口2(结果上传)

系统信息:
- 平台: $(uname -m)
- 内核: $(uname -r)
- CPU: $(nproc) cores

EOF
    
    log_success "测试环境检查完成"
}

# 单网口吞吐量测试
test_single_port_throughput() {
    local interface=$1
    local target_ip=$2
    local test_duration=${3:-30}
    local description=$4
    
    log_info "测试 $interface ($description) -> $target_ip"
    
    # 记录到报告
    echo "" >> $TEST_REPORT
    echo "[$interface 测试] $description" >> $TEST_REPORT
    echo "目标IP: $target_ip" >> $TEST_REPORT
    echo "测试时长: ${test_duration}s" >> $TEST_REPORT
    
    # 网络延迟测试
    log_info "测试网络延迟..."
    if ping_result=$(ping -c 5 -W 1 $target_ip 2>&1); then
        avg_latency=$(echo "$ping_result" | grep "avg" | cut -d'=' -f2 | cut -d'/' -f2)
        log_info "$interface 平均延迟: ${avg_latency}ms"
        echo "平均延迟: ${avg_latency}ms" >> $TEST_REPORT
    else
        log_warning "$interface 延迟测试失败: $target_ip 不可达"
        echo "延迟测试: 失败 (目标不可达)" >> $TEST_REPORT
        return 1
    fi
    
    # 吞吐量测试
    log_info "开始吞吐量测试 (${test_duration}s)..."
    
    # iperf3客户端测试命令
    local iperf_cmd="iperf3 -c $target_ip -t $test_duration -i 5 -w 2M -P 8 -J"
    
    # 绑定到指定网口IP (如果可能)
    if [ "$interface" = "eth0" ]; then
        iperf_cmd="$iperf_cmd -B 192.168.1.10"  # 相机网络IP
    elif [ "$interface" = "eth1" ]; then
        iperf_cmd="$iperf_cmd -B 192.168.2.10"  # 上传网络IP
    fi
    
    # 执行测试
    local temp_result="/tmp/iperf_${interface}_result.json"
    
    if timeout $((test_duration + 10)) $iperf_cmd > $temp_result 2>&1; then
        # 解析JSON结果
        if command -v python3 >/dev/null && [ -f $temp_result ]; then
            local throughput_mbps=$(python3 -c "
import json, sys
try:
    with open('$temp_result') as f:
        data = json.load(f)
    bps = data['end']['sum_received']['bits_per_second']
    mbps = bps / (1024 * 1024)
    print(f'{mbps:.1f}')
except:
    print('0')
")
            
            if (( $(echo "$throughput_mbps >= 900" | bc -l) )); then
                log_success "$interface 吞吐量: ${throughput_mbps} Mbps ✅ (≥900Mbps)"
                echo "吞吐量测试: ${throughput_mbps} Mbps ✅ 达标" >> $TEST_REPORT
                
                # 保存结果
                if [ "$interface" = "eth0" ]; then
                    ETH0_RESULT=$throughput_mbps
                elif [ "$interface" = "eth1" ]; then
                    ETH1_RESULT=$throughput_mbps
                fi
                
                return 0
            else
                log_warning "$interface 吞吐量: ${throughput_mbps} Mbps ⚠️ (<900Mbps)"
                echo "吞吐量测试: ${throughput_mbps} Mbps ⚠️ 未达标" >> $TEST_REPORT
                
                # 保存结果
                if [ "$interface" = "eth0" ]; then
                    ETH0_RESULT=$throughput_mbps
                elif [ "$interface" = "eth1" ]; then
                    ETH1_RESULT=$throughput_mbps
                fi
                
                return 1
            fi
        else
            log_error "$interface iperf3结果解析失败"
            echo "吞吐量测试: 解析失败" >> $TEST_REPORT
            return 1
        fi
    else
        log_error "$interface iperf3测试失败"
        echo "吞吐量测试: 执行失败" >> $TEST_REPORT
        
        # 显示错误信息
        if [ -f $temp_result ]; then
            log_error "错误详情: $(head -5 $temp_result)"
        fi
        
        return 1
    fi
}

# 并发双网口测试
test_concurrent_dual_ports() {
    log_info "🔥 并发测试双千兆网口..."
    
    echo "" >> $TEST_REPORT
    echo "[并发双网口测试]" >> $TEST_REPORT
    echo "测试场景: 同时测试eth0和eth1的最大吞吐量" >> $TEST_REPORT
    
    # 创建后台任务文件
    local eth0_pid_file="/tmp/eth0_test.pid"
    local eth1_pid_file="/tmp/eth1_test.pid"
    local eth0_result_file="/tmp/eth0_concurrent_result.json"
    local eth1_result_file="/tmp/eth1_concurrent_result.json"
    
    # 清理旧文件
    rm -f $eth0_pid_file $eth1_pid_file $eth0_result_file $eth1_result_file
    
    log_info "启动eth0测试 (相机网络)..."
    (
        iperf3 -c 192.168.1.100 -t 60 -i 10 -w 4M -P 4 -B 192.168.1.10 -J > $eth0_result_file 2>&1
        echo "eth0测试完成" >> $TEST_REPORT
    ) &
    echo $! > $eth0_pid_file
    
    log_info "启动eth1测试 (上传网络)..."
    (
        iperf3 -c 192.168.2.100 -t 60 -i 10 -w 4M -P 4 -B 192.168.2.10 -J > $eth1_result_file 2>&1
        echo "eth1测试完成" >> $TEST_REPORT
    ) &
    echo $! > $eth1_pid_file
    
    log_info "并发测试进行中 (60s)..."
    echo "并发测试开始时间: $(date)" >> $TEST_REPORT
    
    # 等待测试完成
    local countdown=60
    while [ $countdown -gt 0 ]; do
        printf "\r⏱️  剩余时间: %02d:%02d" $((countdown/60)) $((countdown%60))
        sleep 1
        countdown=$((countdown-1))
    done
    echo ""
    
    # 等待后台进程完成
    if [ -f $eth0_pid_file ]; then
        wait $(cat $eth0_pid_file) 2>/dev/null || true
    fi
    if [ -f $eth1_pid_file ]; then
        wait $(cat $eth1_pid_file) 2>/dev/null || true
    fi
    
    log_info "解析并发测试结果..."
    
    # 解析eth0结果
    local eth0_mbps=0
    if [ -f $eth0_result_file ] && command -v python3 >/dev/null; then
        eth0_mbps=$(python3 -c "
import json, sys
try:
    with open('$eth0_result_file') as f:
        data = json.load(f)
    bps = data['end']['sum_received']['bits_per_second']
    mbps = bps / (1024 * 1024)
    print(f'{mbps:.1f}')
except:
    print('0')
" 2>/dev/null || echo "0")
    fi
    
    # 解析eth1结果
    local eth1_mbps=0
    if [ -f $eth1_result_file ] && command -v python3 >/dev/null; then
        eth1_mbps=$(python3 -c "
import json, sys
try:
    with open('$eth1_result_file') as f:
        data = json.load(f)
    bps = data['end']['sum_received']['bits_per_second']  
    mbps = bps / (1024 * 1024)
    print(f'{mbps:.1f}')
except:
    print('0')
" 2>/dev/null || echo "0")
    fi
    
    # 计算总吞吐量
    local total_mbps=$(echo "$eth0_mbps + $eth1_mbps" | bc -l 2>/dev/null || echo "0")
    
    # 记录结果
    echo "" >> $TEST_REPORT
    echo "并发测试结果:" >> $TEST_REPORT
    echo "eth0 (相机网络): ${eth0_mbps} Mbps" >> $TEST_REPORT
    echo "eth1 (上传网络): ${eth1_mbps} Mbps" >> $TEST_REPORT
    echo "总吞吐量: ${total_mbps} Mbps" >> $TEST_REPORT
    
    # 输出结果
    log_info "📊 === 并发测试结果 ==="
    log_info "eth0 (相机网络): ${eth0_mbps} Mbps"
    log_info "eth1 (上传网络): ${eth1_mbps} Mbps"  
    log_info "总吞吐量: ${total_mbps} Mbps"
    
    # 判断是否达标
    local eth0_pass=false
    local eth1_pass=false
    
    if (( $(echo "$eth0_mbps >= 900" | bc -l 2>/dev/null || echo "0") )); then
        eth0_pass=true
        log_success "✅ eth0 并发吞吐量达标"
    else
        log_warning "⚠️ eth0 并发吞吐量未达标"
    fi
    
    if (( $(echo "$eth1_mbps >= 900" | bc -l 2>/dev/null || echo "0") )); then
        eth1_pass=true
        log_success "✅ eth1 并发吞吐量达标"
    else
        log_warning "⚠️ eth1 并发吞吐量未达标"
    fi
    
    if [ "$eth0_pass" = true ] && [ "$eth1_pass" = true ]; then
        log_success "🎉 双网口并发测试全部通过！"
        echo "并发测试结论: ✅ 全部达标" >> $TEST_REPORT
        DUAL_PORT_RESULT=1
        return 0
    else
        log_warning "⚠️ 部分网口并发吞吐量未达标"
        echo "并发测试结论: ⚠️ 部分未达标" >> $TEST_REPORT
        DUAL_PORT_RESULT=0
        return 1
    fi
    
    # 清理临时文件
    rm -f $eth0_pid_file $eth1_pid_file $eth0_result_file $eth1_result_file
}

# 2K视频流模拟测试
test_2k_video_stream() {
    log_info "📹 2K视频流传输模拟测试..."
    
    # 计算2K视频流数据量
    # 2K分辨率: 1920x1080, 30fps, RGB (3 bytes/pixel)
    local width=1920
    local height=1080
    local fps=30
    local bytes_per_pixel=3
    
    local bytes_per_frame=$((width * height * bytes_per_pixel))
    local bytes_per_second=$((bytes_per_frame * fps))
    local mbps_required=$((bytes_per_second * 8 / 1024 / 1024))
    
    echo "" >> $TEST_REPORT
    echo "[2K视频流需求分析]" >> $TEST_REPORT
    echo "分辨率: ${width}x${height}" >> $TEST_REPORT
    echo "帧率: ${fps} FPS" >> $TEST_REPORT
    echo "每帧大小: $(echo "scale=1; $bytes_per_frame/1024/1024" | bc) MB" >> $TEST_REPORT
    echo "理论带宽需求: ${mbps_required} Mbps" >> $TEST_REPORT
    
    log_info "2K视频流理论带宽需求: ${mbps_required} Mbps"
    log_info "网口1实测吞吐量: ${ETH0_RESULT} Mbps"
    
    if (( $(echo "$ETH0_RESULT >= $mbps_required" | bc -l) )); then
        log_success "✅ 网口1满足2K视频流传输需求"
        echo "2K视频流适配性: ✅ 满足需求" >> $TEST_REPORT
        return 0
    else
        log_warning "⚠️ 网口1可能无法满足2K视频流需求"
        echo "2K视频流适配性: ⚠️ 可能不足" >> $TEST_REPORT
        return 1
    fi
}

# 生成最终报告
generate_final_report() {
    log_info "📋 生成最终测试报告..."
    
    cat >> $TEST_REPORT << EOF

========================================
最终测试结果汇总
========================================

网口吞吐量测试:
- eth0 (相机网络): ${ETH0_RESULT} Mbps $([ $(echo "$ETH0_RESULT >= 900" | bc -l) -eq 1 ] && echo "✅ 达标" || echo "❌ 未达标")
- eth1 (上传网络): ${ETH1_RESULT} Mbps $([ $(echo "$ETH1_RESULT >= 900" | bc -l) -eq 1 ] && echo "✅ 达标" || echo "❌ 未达标")

双网口并发测试: $([ $DUAL_PORT_RESULT -eq 1 ] && echo "✅ 通过" || echo "❌ 失败")

2K视频流适配性: $([ $(echo "$ETH0_RESULT >= 248" | bc -l) -eq 1 ] && echo "✅ 满足" || echo "⚠️ 可能不足")

系统建议:
EOF

    # 添加优化建议
    if [ $(echo "$ETH0_RESULT < 900" | bc -l) -eq 1 ] || [ $(echo "$ETH1_RESULT < 900" | bc -l) -eq 1 ]; then
        cat >> $TEST_REPORT << EOF
- 网口未达标，建议优化措施:
  * 检查网线质量 (建议Cat6以上)
  * 确认交换机支持千兆
  * 运行RGMII驱动优化脚本
  * 检查CPU中断亲和性设置
  * 调整网络缓冲区参数
EOF
    else
        cat >> $TEST_REPORT << EOF
- ✅ 网络性能优秀，满足工业应用要求
- ✅ 可支持2K实时图像传输
- ✅ 可支持高频检测结果上传
EOF
    fi
    
    cat >> $TEST_REPORT << EOF

测试完成时间: $(date)
测试报告路径: $TEST_REPORT
EOF
    
    # 显示最终结果
    echo ""
    echo "="*60
    log_info "📊 最终测试结果"
    echo "="*60
    
    echo "网口吞吐量:"
    if [ $(echo "$ETH0_RESULT >= 900" | bc -l) -eq 1 ]; then
        log_success "  eth0 (相机): ${ETH0_RESULT} Mbps ✅"
    else
        log_warning "  eth0 (相机): ${ETH0_RESULT} Mbps ❌"
    fi
    
    if [ $(echo "$ETH1_RESULT >= 900" | bc -l) -eq 1 ]; then
        log_success "  eth1 (上传): ${ETH1_RESULT} Mbps ✅"
    else
        log_warning "  eth1 (上传): ${ETH1_RESULT} Mbps ❌"
    fi
    
    if [ $DUAL_PORT_RESULT -eq 1 ]; then
        log_success "双网口并发: ✅ 通过"
    else
        log_warning "双网口并发: ❌ 未通过"
    fi
    
    echo ""
    log_info "📄 详细报告: $TEST_REPORT"
    echo "="*60
}

# 主函数
main() {
    # 环境检查
    check_test_environment
    
    log_info "开始网络吞吐量验证测试..."
    
    # 提示用户准备测试服务器
    echo ""
    echo "⚠️  测试前准备:"
    echo "1. 在相机网络 (192.168.1.100) 启动iperf3服务器:"
    echo "   iperf3 -s -B 192.168.1.100"
    echo ""
    echo "2. 在上传网络 (192.168.2.100) 启动iperf3服务器:"
    echo "   iperf3 -s -B 192.168.2.100"
    echo ""
    read -p "服务器准备就绪后按Enter继续..." -r
    
    # 单网口测试
    log_info "阶段1: 单网口吞吐量测试"
    
    echo "测试eth0 (工业相机网络)..."
    if ! test_single_port_throughput "eth0" "192.168.1.100" 30 "工业相机网络"; then
        log_warning "eth0测试未完全通过"
    fi
    
    echo ""
    echo "测试eth1 (检测结果上传网络)..."
    if ! test_single_port_throughput "eth1" "192.168.2.100" 30 "结果上传网络"; then
        log_warning "eth1测试未完全通过"
    fi
    
    # 并发测试
    echo ""
    log_info "阶段2: 双网口并发测试"
    test_concurrent_dual_ports
    
    # 2K视频流测试
    echo ""
    log_info "阶段3: 2K视频流适配性分析"
    test_2k_video_stream
    
    # 生成报告
    generate_final_report
    
    # 返回结果
    if [ $(echo "$ETH0_RESULT >= 900" | bc -l) -eq 1 ] && [ $(echo "$ETH1_RESULT >= 900" | bc -l) -eq 1 ]; then
        log_success "🎉 网络吞吐量验证全部通过！"
        exit 0
    else
        log_warning "⚠️ 部分网口未达到900Mbps要求"
        exit 1
    fi
}

# 执行主函数
main "$@"
