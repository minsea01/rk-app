#!/usr/bin/env python3
"""
RTP H.264 视频流接收器
适用于RK3588等嵌入式平台的低延迟视频处理
"""

import gi
import numpy as np
import cv2
import queue
import threading
import time
import argparse

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

class RTPVideoReceiver:
    def __init__(self, port=5000, latency=50, max_buffers=4):
        """
        初始化RTP H.264视频接收器
        
        Args:
            port: UDP端口号
            latency: 抖动缓冲区延迟(ms)
            max_buffers: 最大缓冲帧数
        """
        Gst.init(None)
        
        self.port = port
        self.frame_queue = queue.Queue(maxsize=8)
        self.running = False
        
        # 性能统计
        self.stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'start_time': time.time(),
            'last_fps_update': time.time()
        }
        
        # 构建GStreamer管道
        pipeline_str = (
            f"udpsrc port={port} "
            f"caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
            f"rtpjitterbuffer latency={latency} drop-on-late=true do-lost=true ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=true sync=false "
            f"max-buffers={max_buffers} drop=true"
        )
        
        print(f"[INFO] 创建视频管道: {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name("sink")
        
        # 连接回调函数
        self.sink.connect("new-sample", self._on_new_sample)
    
    def _on_new_sample(self, sink):
        """处理新的视频帧"""
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
            
        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        # 获取帧尺寸
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        
        # 映射缓冲区数据
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        
        # 转换为numpy数组
        frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
        frame = frame_data.reshape((height, width, 3)).copy()
        buf.unmap(map_info)
        
        # 放入队列（非阻塞，丢弃旧帧保证实时性）
        try:
            self.frame_queue.put_nowait(frame)
            self.stats['frames_received'] += 1
        except queue.Full:
            # 队列满时丢弃当前帧
            pass
            
        return Gst.FlowReturn.OK
    
    def start(self):
        """启动视频接收"""
        print(f"[INFO] 启动视频接收器，监听端口 {self.port}")
        self.running = True
        self.pipeline.set_state(Gst.State.PLAYING)
    
    def stop(self):
        """停止视频接收"""
        print("[INFO] 停止视频接收器")
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
    
    def get_frame(self, timeout=0.1):
        """
        获取最新帧
        
        Args:
            timeout: 超时时间(秒)
            
        Returns:
            numpy.ndarray: BGR格式的图像帧，如果无帧则返回None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def process_video(self, processor_func=None):
        """
        视频处理主循环
        
        Args:
            processor_func: 自定义处理函数，接收frame参数
        """
        print("[INFO] 开始视频处理循环")
        
        while self.running:
            frame = self.get_frame()
            if frame is None:
                continue
                
            # 默认处理：边缘检测
            if processor_func:
                processed_frame = processor_func(frame)
            else:
                processed_frame = self._default_process(frame)
            
            self.stats['frames_processed'] += 1
            self._update_fps_stats()
    
    def _default_process(self, frame):
        """默认图像处理：边缘检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return edges
    
    def _update_fps_stats(self):
        """更新FPS统计"""
        now = time.time()
        if now - self.stats['last_fps_update'] >= 2.0:  # 每2秒更新一次
            elapsed = now - self.stats['start_time']
            recv_fps = self.stats['frames_received'] / elapsed
            proc_fps = self.stats['frames_processed'] / elapsed
            
            print(f"[STATS] 接收FPS: {recv_fps:.1f}, 处理FPS: {proc_fps:.1f}")
            self.stats['last_fps_update'] = now


def vision_control_demo(frame):
    """
    机器视觉控制演示
    可以在这里添加目标检测、跟踪等算法
    """
    height, width = frame.shape[:2]
    
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在原图上绘制轮廓和中心点
    result = frame.copy()
    
    for contour in contours:
        # 过滤小轮廓
        area = cv2.contourArea(contour)
        if area > 1000:
            # 绘制轮廓
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            
            # 计算中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 绘制中心点
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                
                # 计算相对于图像中心的偏移（可用于控制系统）
                center_x, center_y = width // 2, height // 2
                offset_x = cx - center_x
                offset_y = cy - center_y
                
                # 在这里可以调用PID控制器来计算控制量
                # pid_output_x = pid_x.compute(0, offset_x, dt)
                # pid_output_y = pid_y.compute(0, offset_y, dt)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='RTP H.264 视频流接收器')
    parser.add_argument('--port', type=int, default=5000, help='UDP端口号')
    parser.add_argument('--latency', type=int, default=50, help='抖动缓冲区延迟(ms)')
    parser.add_argument('--demo', action='store_true', help='运行视觉控制演示')
    
    args = parser.parse_args()
    
    # 创建接收器
    receiver = RTPVideoReceiver(port=args.port, latency=args.latency)
    
    try:
        # 启动接收
        receiver.start()
        
        # 启动GLib主循环（在后台线程）
        loop = GObject.MainLoop()
        loop_thread = threading.Thread(target=loop.run, daemon=True)
        loop_thread.start()
        
        # 选择处理函数
        processor = vision_control_demo if args.demo else None
        
        # 开始处理
        receiver.process_video(processor)
        
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，正在退出...")
    finally:
        receiver.stop()
        

if __name__ == "__main__":
    main()
