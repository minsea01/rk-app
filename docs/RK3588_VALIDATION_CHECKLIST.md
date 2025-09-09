# RK3588实机验证执行单
**目标**: 补齐吞吐量与FPS实证数据，完成项目闭环验证

## 📋 验证清单

### **第一步：RK3588系统环境准备**

#### **1.1 系统安装验证**
```bash
# 期望输出示例
$ lsb_release -a
Description: Ubuntu 20.04.6 LTS
Release: 20.04

$ uname -r
5.10.110-rockchip-rk3588  # 或类似内核版本

$ ls /sys/class/devfreq/ | grep npu
fdab0000.npu  # 确认NPU设备存在
```

#### **1.2 RKNN环境配置**
```bash
# 安装RKNN运行时
sudo apt install python3-rknnlite2

# 验证RKNN环境
python3 -c "
from rknnlite.api import RKNNLite
print('✅ RKNN环境正常')
"

# 期望输出: ✅ RKNN环境正常
```

#### **1.3 交叉编译或本地构建**
```bash
cd /path/to/rk-app
cmake -S . -B build -DENABLE_GIGE=ON -DENABLE_RKNN=ON -DRKNN_HOME=/opt/rknpu2 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 期望输出: [100%] Built target detect_cli
```

### **第二步：双千兆网口验证**

#### **2.1 网络配置部署**
```bash
sudo ./RK3588_Deploy/scripts/setup_network.sh

# 期望输出示例:
# ✅ eth0配置完成: 192.168.1.10/24
# ✅ eth1配置完成: 192.168.2.10/24
```

#### **2.2 网口状态验证**
```bash
# 检查网口速度
ethtool eth0 | grep -E "Speed|Duplex|Link detected"
ethtool eth1 | grep -E "Speed|Duplex|Link detected"

# 期望输出:
# Speed: 1000Mb/s
# Duplex: Full  
# Link detected: yes
```

#### **2.3 吞吐量实测**
```bash
# 在连接的设备上启动iperf3服务器
# 设备1: iperf3 -s -B 192.168.1.100
# 设备2: iperf3 -s -B 192.168.2.100

# RK3588上测试网口1
iperf3 -c 192.168.1.100 -t 60 -i 5 -w 4M -P 4 -B 192.168.1.10

# 期望输出示例:
# [SUM]   0.00-60.00  sec  6.60 GBytes   944 Mbits/sec  # ≥900Mbps ✅

# RK3588上测试网口2  
iperf3 -c 192.168.2.100 -t 60 -i 5 -w 4M -P 4 -B 192.168.2.10

# 期望输出示例:
# [SUM]   0.00-60.00  sec  6.72 GBytes   960 Mbits/sec  # ≥900Mbps ✅

# 并发测试（关键验证）
iperf3 -c 192.168.1.100 -t 60 -w 4M -P 2 -B 192.168.1.10 &
iperf3 -c 192.168.2.100 -t 60 -w 4M -P 2 -B 192.168.2.10 &
wait

# 期望: 两个网口同时≥900Mbps
```

### **第三步：NPU推理性能验证**

#### **3.1 RKNN模型验证**
```bash
# 复制模型到RK3588
cp artifacts/models/industrial_15cls_rk3588_w8a8.rknn /tmp/

# 验证模型加载
python3 -c "
from rknnlite.api import RKNNLite
rknn = RKNNLite()
ret = rknn.load_rknn('/tmp/industrial_15cls_rk3588_w8a8.rknn')
print(f'模型加载: {\"成功\" if ret == 0 else \"失败\"}')
ret = rknn.init_runtime()
print(f'运行时初始化: {\"成功\" if ret == 0 else \"失败\"}')
"

# 期望输出:
# 模型加载: 成功
# 运行时初始化: 成功
```

#### **3.2 NPU推理性能测试**
```bash
# 使用RKNN配置运行检测
timeout 120s ./build/detect_cli --cfg config/deploy/rk3588_industrial_final.yaml 2>&1 | tee rknn_performance.log

# 性能数据提取
echo "=== NPU性能统计 ==="
echo "检测帧数: $(grep 'Frame.*detections' rknn_performance.log | wc -l)"
echo "平均推理时间: $(grep -o '([0-9]*ms)' rknn_performance.log | sed 's/[()]//g' | sed 's/ms//' | awk '{sum+=$1; count++} END {print sum/count \"ms\"}')"
echo "平均FPS: $(grep -o '([0-9]*ms)' rknn_performance.log | sed 's/[()]//g' | sed 's/ms//' | awk '{sum+=$1; count++} END {print 1000/(sum/count) \"fps\"}')"

# 期望输出示例:
# 检测帧数: 200
# 平均推理时间: 25ms     # <40ms目标
# 平均FPS: 40fps        # ≥24fps ✅
```

#### **3.3 NPU多核使用率验证**
```bash
# 监控NPU使用率
while true; do
    echo "$(date): NPU频率 $(cat /sys/class/devfreq/fdab0000.npu/cur_freq)"
    sleep 1
done &

# 运行推理并观察NPU工作状态
./build/detect_cli --cfg config/rk3588_industrial_final.yaml &
PID=$!

# 10秒后停止
sleep 10 && kill $PID

# 期望: 看到NPU频率变化，说明NPU被正确使用
```

### **第四步：2K工业相机验证**

#### **4.1 真实相机连接测试**
```bash
# 假设连接真实2K工业相机到eth0网段
arv-tool-0.10 list

# 期望输出示例:
# IndustrialCamera-2K (192.168.1.100)

# 测试2K分辨率采集
gst-launch-1.0 -v aravissrc camera-name="IndustrialCamera-2K" ! video/x-raw,width=2048,height=1536,framerate=24/1 ! videoconvert ! video/x-raw,format=BGR ! fakesink sync=false

# 期望: 管道正常运行，无错误
```

#### **4.2 2K@24fps完整系统测试**
```bash
# 修改配置为2K分辨率
sed -i 's/width=512,height=512,framerate=25\/1/width=2048,height=1536,framerate=24\/1/' config/deploy/rk3588_industrial_final.yaml

# 运行完整2K检测测试
timeout 120s ./build/detect_cli --cfg config/deploy/rk3588_industrial_final.yaml 2>&1 | tee 2k_system_test.log

# 验证数据量
echo "2K图像处理验证:"
echo "图像尺寸: 2048x1536"
echo "处理帧数: $(grep 'Frame.*detections' 2k_system_test.log | wc -l)"
echo "系统稳定性: $(grep 'Frame.*detections' 2k_system_test.log | tail -1)"

# 期望: 稳定处理2K图像，帧率满足要求
```

### **第五步：结果上传验证**

#### **5.1 网络上传测试**
```bash
# 在eth1网段启动简单接收服务器
python3 -c "
import socket, threading, time
def server():
    s = socket.socket()
    s.bind(('192.168.2.10', 8080))
    s.listen(5)
    print('服务器监听 192.168.2.10:8080')
    while True:
        conn, addr = s.accept()
        data = conn.recv(4096)
        print(f'收到数据: {len(data)}字节 from {addr}')
        conn.close()
threading.Thread(target=server, daemon=True).start()
time.sleep(3600)  # 保持1小时
" &

# 修改配置启用网络上传
sed -i 's/ip: \"127.0.0.1\"/ip: \"192.168.2.10\"/' config/deploy/rk3588_industrial_final.yaml
sed -i 's/port: 9000/port: 8080/' config/deploy/rk3588_industrial_final.yaml

# 运行检测验证上传
timeout 60s ./build/detect_cli --cfg config/deploy/rk3588_industrial_final.yaml

# 期望: 看到"收到数据"的服务器日志输出
```

## 📊 **验证成功标准**

### **✅ 系统移植+网络性能**
- [ ] eth0和eth1都达到1000Mb/s链路速度
- [ ] iperf3测试双网口各≥900Mbps
- [ ] 双网口并发测试各≥900Mbps

### **✅ NPU推理性能**  
- [ ] RKNN模型成功加载到NPU
- [ ] 推理FPS ≥24fps
- [ ] 单帧推理延迟 <40ms
- [ ] NPU设备正常工作

### **✅ 完整系统集成**
- [ ] 2K工业相机稳定采集
- [ ] 15类检测正常工作
- [ ] 网络结果上传成功
- [ ] 系统连续运行无崩溃

## 🎯 **预期验证时间**

```
硬件环境搭建: 0.5天
网络性能验证: 0.5天  
NPU推理验证: 1天
完整集成测试: 1天
总计: 3天完成全部验证
```

验证完成后，您将拥有完整的实测数据证明项目完全满足所有技术指标！
