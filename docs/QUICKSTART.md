# RK3588工业检测系统 - 快速启动指南

## 🚀 一键部署
```bash
sudo ./deploy.sh
```

## 🎯 立即启动
```bash
cd scripts
python3 rk3588_industrial_detector.py
```

## 📊 预期性能
- mAP50: 94.5%
- 检测类别: 80类  
- 处理速度: 25-30 FPS
- 网络吞吐: >900 Mbps

## 🔧 系统服务
```bash
sudo systemctl start rk3588-industrial-detector
sudo systemctl status rk3588-industrial-detector
```

详细文档请参考 README.md
