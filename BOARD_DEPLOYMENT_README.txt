================================================================
RK3588板子到手后快速部署指南
================================================================

部署包: rk-deploy-complete.tar.gz (2.5MB)
预计时间: 20-40分钟（首次部署）

================================================================
Step 1: 连接板子（2分钟）
================================================================

ssh radxa@<板子IP>
# 默认密码通常是: radxa 或 rock

# 验证系统
uname -a
# 应该看到: Linux ... aarch64

================================================================
Step 2: 传输部署包（5分钟）
================================================================

# 在PC (WSL) 上执行:
scp rk-deploy-complete.tar.gz radxa@<板子IP>:/home/radxa/

# 验证传输
ssh radxa@<板子IP> "ls -lh /home/radxa/rk-deploy-complete.tar.gz"

================================================================
Step 3: 解压（1分钟）
================================================================

ssh radxa@<板子IP>
cd /home/radxa
tar xzf rk-deploy-complete.tar.gz

# 验证文件
ls -la apps/ config/ scripts/deploy/

================================================================
Step 4: 健康检查（5分钟）
================================================================

bash scripts/deploy/board_health_check.sh

预期输出:
  [Python3安装] ... PASS
  [Pip3安装] ... PASS
  [NumPy安装] ... PASS/FAIL
  [OpenCV安装] ... PASS/FAIL
  [RKNNLite导入] ... PASS/FAIL
  [NPU设备文件] ... PASS
  ...
  总计: X PASS, Y FAIL

如果全部PASS:
  ✅ 直接跳到Step 6

如果有FAIL:
  ⚠️  继续Step 5

================================================================
Step 5: 安装依赖（10-20分钟，仅在Step 4有FAIL时执行）
================================================================

bash scripts/deploy/install_dependencies.sh

这个脚本会自动:
  - 配置清华pip镜像（加速下载）
  - 安装numpy, opencv, pillow
  - 尝试安装rknn-toolkit-lite2

注意: 如果rknn-toolkit-lite2自动安装失败，脚本会提示手动下载链接

手动安装rknn-toolkit-lite2:
  1. 访问: https://github.com/rockchip-linux/rknn-toolkit2/releases
  2. 下载对应Python版本的wheel文件，例如:
     rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl
  3. 安装: pip3 install <wheel文件>

================================================================
Step 6: 首次推理测试（5分钟）
================================================================

# 下载测试图片
wget -O /home/radxa/test.jpg https://ultralytics.com/images/zidane.jpg

# 设置Python路径
export PYTHONPATH=/home/radxa

# 运行推理
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/best.rknn \
  --source test.jpg \
  --save result.jpg \
  --imgsz 640 \
  --conf 0.25

预期输出:
  [INFO] Loading RKNN: artifacts/models/best.rknn
  [INFO] Initializing runtime, core_mask=0x7
  [INFO] Inference time: XX.XX ms
  [INFO] Detections: X
  [INFO] Saved: result.jpg

如果看到以上输出:
  🎉 恭喜！部署成功！

================================================================
常见问题处理
================================================================

问题1: RKNNLite导入失败
  错误: ImportError: No module named 'rknnlite'
  解决: bash scripts/deploy/install_dependencies.sh

问题2: NPU设备不存在
  错误: ls: cannot access '/dev/rknpu*'
  解决: sudo modprobe rknpu

问题3: OpenCV导入错误
  错误: ImportError: libGL.so.1: cannot open shared object file
  解决: sudo apt install -y libgl1-mesa-glx libglib2.0-0

问题4: 内存不足
  错误: RuntimeError: Cannot allocate memory
  解决:
    - 检查: free -h
    - 清理: sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    - 或减小图片尺寸: --imgsz 416

================================================================
性能验证（可选）
================================================================

# FPS基准测试
python3 scripts/profiling/performance_profiler.py \
  --model artifacts/models/best.rknn \
  --model-type rknn \
  --images-dir <测试图片目录> \
  --limit 100 \
  --output performance_report.json

预期性能:
  - 推理延迟: 20-40ms @ 640×640
  - FPS: 25-35 (INT8量化)
  - 内存峰值: ~300MB

================================================================
网络配置（毕设要求）
================================================================

# RGMII驱动验证
sudo bash scripts/network/rgmii_driver_config.sh

# 配置双千兆网口
sudo ip addr add 192.168.1.10/24 dev eth0  # 相机网络
sudo ip addr add 192.168.2.10/24 dev eth1  # 上传网络

# 吞吐量测试（≥900Mbps验证）
bash scripts/network/network_throughput_validator.sh

================================================================
故障排查检查清单
================================================================

□ Python版本 >= 3.8
  python3 --version

□ NPU设备存在
  ls /dev/rknpu*

□ NPU驱动加载
  lsmod | grep rknpu

□ RKNNLite可导入
  python3 -c "from rknnlite.api import RKNNLite"

□ 模型文件完整
  ls -lh artifacts/models/best.rknn  # 应该是4.7MB

□ 可用内存充足
  free -h  # 至少有1GB可用

================================================================
紧急联系
================================================================

如果遇到无法解决的问题:
  1. 查看错误日志: dmesg | tail -50
  2. 查看Python错误: journalctl -xe
  3. 检查资源使用: top, free -h

Rockchip官方资源:
  - GitHub: https://github.com/rockchip-linux/rknn-toolkit2
  - 文档: https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc
  - 示例: https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknpu2/examples

================================================================
部署成功后的下一步
================================================================

✅ 基础推理成功后，可以继续:
  1. 行人检测mAP验证 (scripts/evaluation/pedestrian_map_evaluator.py)
  2. 网络吞吐量验证 (≥900Mbps)
  3. 长时间稳定性测试
  4. 实时视频流处理

================================================================
