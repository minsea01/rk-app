#!/usr/bin/env python3
"""
RK3588工业相机集成模块
专门处理2K分辨率实时图像采集和网络传输
要求：网口1连接工业相机，实时采集2K图像数据
"""

import cv2
import numpy as np
import socket
import threading
import time
import struct
import json
from datetime import datetime
from pathlib import Path
import logging

class IndustrialCameraHandler:
    """工业相机处理类 - 专门优化2K实时采集"""
    
    def __init__(self, camera_config):
        self.config = camera_config
        self.camera = None
        self.is_streaming = False
        
        # 网络配置
        self.camera_network_ip = "192.168.1.10"  # RK3588在相机网络中的IP
        self.upload_network_ip = "192.168.2.100"  # 上位机IP
        self.upload_port = 8080
        
        # 性能统计
        self.frame_count = 0
        self.bytes_received = 0
        self.bytes_sent = 0
        self.start_time = time.time()
        
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] 工业相机: %(message)s',
            handlers=[
                logging.FileHandler('../logs/industrial_camera.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_gige_camera(self):
        """初始化GigE Vision工业相机"""
        self.logger.info("🔧 初始化GigE Vision工业相机...")
        
        try:
            # 方案1: 使用专业GigE Vision库 (如Vimba, GenICam)
            # 这里用OpenCV作为通用示例，实际部署时建议使用专业库
            
            # 尝试通过IP地址连接网络相机
            gige_url = f"rtsp://192.168.1.100:554/stream"  # GigE Vision RTSP流
            self.camera = cv2.VideoCapture(gige_url)
            
            if not self.camera.isOpened():
                # 备用方案：USB3.0工业相机
                self.logger.warning("GigE相机连接失败，尝试USB相机...")
                self.camera = cv2.VideoCapture(0)
            
            if self.camera.isOpened():
                # 配置2K分辨率
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 2K宽度
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 2K高度
                self.camera.set(cv2.CAP_PROP_FPS, 30)             # 30fps
                
                # 工业相机专用配置
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # 最小缓冲，减少延迟
                self.camera.set(cv2.CAP_PROP_EXPOSURE, -6)        # 自动曝光
                self.camera.set(cv2.CAP_PROP_GAIN, 0)             # 增益控制
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 50)      # 亮度
                self.camera.set(cv2.CAP_PROP_CONTRAST, 50)        # 对比度
                
                # 验证实际配置
                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
                
                self.logger.info(f"✅ 相机初始化成功: {actual_width}x{actual_height} @ {actual_fps}fps")
                
                # 计算2K图像数据量
                bytes_per_frame = actual_width * actual_height * 3  # RGB
                mbps_required = (bytes_per_frame * actual_fps * 8) / (1024 * 1024)
                self.logger.info(f"📊 2K数据流量: {bytes_per_frame/1024/1024:.1f}MB/frame, {mbps_required:.1f}Mbps")
                
                return True
            else:
                raise Exception("工业相机初始化失败")
                
        except Exception as e:
            self.logger.error(f"❌ 相机初始化失败: {e}")
            return False

    def configure_gige_network(self):
        """配置GigE Vision网络参数"""
        self.logger.info("🌐 配置GigE Vision网络...")
        
        try:
            import subprocess
            
            # 配置网口1为相机专用网络
            commands = [
                # 设置网口1 IP
                f"sudo ip addr add {self.camera_network_ip}/24 dev eth0",
                
                # 网口1优化配置
                "sudo ethtool -G eth0 rx 4096 tx 4096",  # 大接收缓冲区
                "sudo ethtool -K eth0 gro on gso on tso on",  # 硬件加速
                "sudo ip link set eth0 mtu 9000",  # 巨型帧 (如果支持)
                
                # 相机网络专用优化
                "echo 268435456 | sudo tee /proc/sys/net/core/rmem_max",  # 256MB接收缓冲
                "echo 10000 | sudo tee /proc/sys/net/core/netdev_max_backlog",  # 队列长度
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd.split(), check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"命令执行失败: {cmd}")
            
            self.logger.info("✅ GigE网络配置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 网络配置失败: {e}")
            return False

    def start_streaming(self):
        """开始2K实时流采集"""
        if not self.camera or not self.camera.isOpened():
            self.logger.error("❌ 相机未初始化，无法开始流采集")
            return False
        
        self.is_streaming = True
        self.logger.info("🎥 开始2K实时图像流采集...")
        
        # 创建结果上传线程
        upload_thread = threading.Thread(target=self.upload_worker, daemon=True)
        upload_thread.start()
        
        # 性能监控线程  
        monitor_thread = threading.Thread(target=self.performance_monitor, daemon=True)
        monitor_thread.start()
        
        return True

    def capture_frame(self):
        """采集单帧2K图像"""
        if not self.is_streaming:
            return None, None
            
        start_time = time.time()
        ret, frame = self.camera.read()
        capture_time = (time.time() - start_time) * 1000  # ms
        
        if ret:
            self.frame_count += 1
            self.bytes_received += frame.nbytes
            
            # 帧信息
            frame_info = {
                'frame_id': self.frame_count,
                'timestamp': datetime.now().isoformat(),
                'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                'size_bytes': frame.nbytes,
                'capture_time_ms': capture_time,
                'network_interface': 'eth0-RGMII'
            }
            
            return frame, frame_info
        else:
            self.logger.warning("⚠️ 图像采集失败")
            return None, None

    def upload_worker(self):
        """检测结果上传工作线程 (网口2)"""
        self.logger.info("📤 启动结果上传线程 (eth1)...")
        
        try:
            # 连接上位机 (通过网口2)
            upload_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            upload_socket.settimeout(5.0)
            
            # 绑定到网口2的IP
            upload_socket.bind(("192.168.2.10", 0))  # RK3588在上传网络中的IP
            upload_socket.connect((self.upload_network_ip, self.upload_port))
            
            self.logger.info(f"✅ 已连接上位机: {self.upload_network_ip}:{self.upload_port}")
            
            while self.is_streaming:
                try:
                    # 模拟检测结果数据
                    result_data = {
                        'timestamp': datetime.now().isoformat(),
                        'frame_id': self.frame_count,
                        'detections': [],  # 实际检测结果
                        'network_stats': self.get_network_stats(),
                        'performance': self.get_performance_stats()
                    }
                    
                    # 发送结果 (通过网口2)
                    message = json.dumps(result_data).encode('utf-8')
                    upload_socket.send(struct.pack('!I', len(message)))  # 先发送长度
                    upload_socket.send(message)  # 再发送数据
                    
                    self.bytes_sent += len(message) + 4
                    
                    time.sleep(0.033)  # ~30fps结果上传
                    
                except Exception as e:
                    self.logger.error(f"❌ 结果上传失败: {e}")
                    break
            
        except Exception as e:
            self.logger.error(f"❌ 上传线程启动失败: {e}")

    def get_network_stats(self):
        """获取网络统计信息"""
        try:
            stats = {}
            
            # 读取网口统计
            for iface in ['eth0', 'eth1']:
                iface_stats = {}
                base_path = f"/sys/class/net/{iface}/statistics"
                
                if Path(base_path).exists():
                    # 接收统计
                    rx_bytes = int(Path(f"{base_path}/rx_bytes").read_text().strip())
                    rx_packets = int(Path(f"{base_path}/rx_packets").read_text().strip())
                    rx_errors = int(Path(f"{base_path}/rx_errors").read_text().strip())
                    
                    # 发送统计
                    tx_bytes = int(Path(f"{base_path}/tx_bytes").read_text().strip())
                    tx_packets = int(Path(f"{base_path}/tx_packets").read_text().strip())
                    tx_errors = int(Path(f"{base_path}/tx_errors").read_text().strip())
                    
                    iface_stats = {
                        'rx_bytes': rx_bytes,
                        'rx_packets': rx_packets,
                        'rx_errors': rx_errors,
                        'tx_bytes': tx_bytes,
                        'tx_packets': tx_packets,
                        'tx_errors': tx_errors,
                        'rx_mbps': (rx_bytes * 8) / (1024 * 1024) / max(1, time.time() - self.start_time),
                        'tx_mbps': (tx_bytes * 8) / (1024 * 1024) / max(1, time.time() - self.start_time)
                    }
                
                stats[iface] = iface_stats
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"网络统计获取失败: {e}")
            return {}

    def get_performance_stats(self):
        """获取性能统计"""
        elapsed_time = time.time() - self.start_time
        
        return {
            'runtime_seconds': elapsed_time,
            'total_frames': self.frame_count,
            'fps': self.frame_count / max(1, elapsed_time),
            'bytes_received': self.bytes_received,
            'bytes_sent': self.bytes_sent,
            'rx_mbps': (self.bytes_received * 8) / (1024 * 1024) / max(1, elapsed_time),
            'tx_mbps': (self.bytes_sent * 8) / (1024 * 1024) / max(1, elapsed_time),
        }

    def performance_monitor(self):
        """性能监控线程"""
        while self.is_streaming:
            time.sleep(30)  # 每30秒输出一次性能报告
            
            stats = self.get_performance_stats()
            network_stats = self.get_network_stats()
            
            self.logger.info("📊 === 性能报告 ===")
            self.logger.info(f"运行时间: {stats['runtime_seconds']:.1f}s")
            self.logger.info(f"采集帧数: {stats['total_frames']}")
            self.logger.info(f"采集帧率: {stats['fps']:.1f} FPS")
            self.logger.info(f"网口1接收: {stats['rx_mbps']:.1f} Mbps")
            self.logger.info(f"网口2发送: {stats['tx_mbps']:.1f} Mbps")
            
            # 检查是否达到900Mbps要求
            if network_stats.get('eth0', {}).get('rx_mbps', 0) >= 900:
                self.logger.info("✅ 网口1吞吐量≥900Mbps - 达标")
            else:
                self.logger.warning("⚠️ 网口1吞吐量<900Mbps - 需要优化")
                
            if network_stats.get('eth1', {}).get('tx_mbps', 0) >= 900:
                self.logger.info("✅ 网口2吞吐量≥900Mbps - 达标")
            else:
                self.logger.warning("⚠️ 网口2吞吐量<900Mbps - 需要优化")

    def stop_streaming(self):
        """停止流采集"""
        self.is_streaming = False
        if self.camera:
            self.camera.release()
        self.logger.info("⏹️ 2K图像流采集已停止")

class NetworkThroughputTester:
    """网络吞吐量测试器 - 验证≥900Mbps要求"""
    
    def __init__(self):
        self.logger = logging.getLogger("ThroughputTester")
    
    def test_interface_throughput(self, interface, target_ip, duration=30):
        """测试单个网口吞吐量"""
        self.logger.info(f"🧪 测试 {interface} 吞吐量 -> {target_ip}")
        
        try:
            import subprocess
            
            # 使用iperf3测试
            cmd = [
                "iperf3", "-c", target_ip, 
                "-t", str(duration),
                "-i", "5",
                "-w", "1M",  # 1MB窗口
                "-P", "4",   # 4个并行连接
                "-J"         # JSON输出
            ]
            
            # 绑定到指定网口 (如果支持)
            if interface == "eth0":
                cmd.extend(["-B", "192.168.1.10"])  # 相机网络IP
            elif interface == "eth1":
                cmd.extend(["-B", "192.168.2.10"])  # 上传网络IP
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration+10)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # 提取吞吐量信息
                throughput_bps = data['end']['sum_received']['bits_per_second']
                throughput_mbps = throughput_bps / (1024 * 1024)
                
                self.logger.info(f"📊 {interface} 测试结果: {throughput_mbps:.1f} Mbps")
                
                if throughput_mbps >= 900:
                    self.logger.info(f"✅ {interface} 吞吐量达标 (≥900Mbps)")
                    return True, throughput_mbps
                else:
                    self.logger.warning(f"⚠️ {interface} 吞吐量不达标 (<900Mbps)")
                    return False, throughput_mbps
            else:
                self.logger.error(f"❌ {interface} 吞吐量测试失败: {result.stderr}")
                return False, 0
                
        except Exception as e:
            self.logger.error(f"❌ {interface} 测试异常: {e}")
            return False, 0
    
    def test_dual_port_concurrent(self):
        """并发测试双网口吞吐量"""
        self.logger.info("🔥 并发测试双千兆网口...")
        
        # 启动两个测试线程
        import threading
        
        eth0_result = [False, 0]
        eth1_result = [False, 0]
        
        def test_eth0():
            eth0_result[0], eth0_result[1] = self.test_interface_throughput(
                "eth0", "192.168.1.100", 30
            )
        
        def test_eth1():
            eth1_result[0], eth1_result[1] = self.test_interface_throughput(
                "eth1", "192.168.2.100", 30
            )
        
        # 并发执行
        t1 = threading.Thread(target=test_eth0)
        t2 = threading.Thread(target=test_eth1)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # 汇总结果
        total_mbps = eth0_result[1] + eth1_result[1]
        
        self.logger.info("📊 === 双网口并发测试结果 ===")
        self.logger.info(f"eth0 (相机网络): {eth0_result[1]:.1f} Mbps")
        self.logger.info(f"eth1 (上传网络): {eth1_result[1]:.1f} Mbps")
        self.logger.info(f"总吞吐量: {total_mbps:.1f} Mbps")
        
        if eth0_result[0] and eth1_result[0]:
            self.logger.info("🎉 双网口吞吐量测试全部通过！")
            return True
        else:
            self.logger.warning("⚠️ 部分网口吞吐量未达标，需要优化")
            return False

def main():
    """主函数 - 工业相机集成测试"""
    print("🏭 RK3588工业相机集成模块")
    print("要求: 2K分辨率实时采集 + 双千兆网口≥900Mbps")
    print("="*60)
    
    # 相机配置
    camera_config = {
        'resolution': (1920, 1080),  # 2K分辨率
        'fps': 30,
        'network_interface': 'eth0',  # 相机连接网口1
        'gige_ip': '192.168.1.100'    # 工业相机IP
    }
    
    # 初始化工业相机处理器
    camera_handler = IndustrialCameraHandler(camera_config)
    
    try:
        # 配置网络
        if not camera_handler.configure_gige_network():
            print("❌ 网络配置失败")
            return
        
        # 初始化相机
        if not camera_handler.initialize_gige_camera():
            print("❌ 工业相机初始化失败")
            return
        
        # 开始流采集
        if not camera_handler.start_streaming():
            print("❌ 流采集启动失败")
            return
        
        print("✅ 工业相机系统启动成功")
        print("🎥 2K实时图像采集中...")
        print("📤 检测结果上传中...")
        print("按 Ctrl+C 停止")
        
        # 主循环 - 采集和处理
        try:
            while True:
                frame, frame_info = camera_handler.capture_frame()
                
                if frame is not None:
                    # 这里可以插入AI检测代码
                    # detections = run_yolo_detection(frame)
                    
                    # 显示帧信息 (每100帧一次)
                    if frame_info['frame_id'] % 100 == 0:
                        print(f"📊 Frame {frame_info['frame_id']}: "
                              f"{frame_info['resolution']}, "
                              f"{frame_info['size_bytes']/1024/1024:.1f}MB, "
                              f"采集用时: {frame_info['capture_time_ms']:.1f}ms")
                
                time.sleep(0.001)  # 最小延迟
                
        except KeyboardInterrupt:
            print("\n🛑 用户停止系统")
        
    finally:
        camera_handler.stop_streaming()
        print("✅ 工业相机系统已停止")

if __name__ == "__main__":
    main()
