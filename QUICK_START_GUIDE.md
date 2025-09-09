# 🚀 快速开始指南

文件重新组织后的项目运行指南

## 📁 新的目录结构

```
rk-app/
├── artifacts/models/          # 所有模型文件 (.pt, .onnx, .rknn)
├── config/
│   ├── detection/            # 检测配置 (detect.yaml)
│   ├── deploy/              # 部署配置 (rk3588_industrial_final.yaml)
│   └── industrial_classes.txt
├── scripts/
│   ├── demo/                # 演示脚本
│   ├── benchmark/           # 性能测试
│   └── reports/             # 报告生成
├── logs/                    # 所有日志文件
└── docs/                    # 技术文档
```

## 🎯 快速演示命令

### **方法1：从项目根目录运行**
```bash
cd /home/minsea01/dev/rk-projects/rk-app

# 设置环境
export LD_LIBRARY_PATH=$PWD/.third_party/aravis/_install/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=$PWD/.third_party/aravis/_install/lib/x86_64-linux-gnu/gstreamer-1.0
export PATH=$PWD/.third_party/aravis/_install/bin:$PATH

# 启动假相机（如果未运行）
sudo -E arv-fake-gv-camera-0.10 -i 127.0.0.1 >/tmp/arv_fake.log 2>&1 &

# 运行检测系统
./build/detect_cli --cfg config/detection/detect.yaml
```

### **方法2：使用演示脚本**
```bash
cd /home/minsea01/dev/rk-projects/rk-app

# 运行完整演示脚本（推荐给老师演示）
./scripts/demo/demo_presentation_script.sh
```

## 📊 查看历史测试数据

```bash
# 查看演示日志
cat logs/demo_results.log

# 查看性能统计
tail -10 logs/demo_results.log
```

## 🔧 RK3588部署

```bash
# 使用RK3588配置（硬件到货后）
./build/detect_cli --cfg config/deploy/rk3588_industrial_final.yaml

# 完整验证清单
cat docs/RK3588_VALIDATION_CHECKLIST.md
```

## 📋 关键文件位置

| 文件类型 | 位置 | 说明 |
|---------|------|------|
| **ONNX模型** | `artifacts/models/best.onnx` | x86验证用 |
| **RKNN模型** | `artifacts/models/industrial_15cls_rk3588_w8a8.rknn` | RK3588 NPU用 |
| **检测配置** | `config/detection/detect.yaml` | 当前演示配置 |
| **部署配置** | `config/deploy/rk3588_industrial_final.yaml` | RK3588最终配置 |
| **演示日志** | `logs/demo_results.log` | 实际测试数据 |
| **演示脚本** | `scripts/demo/demo_presentation_script.sh` | 老师汇报用 |
| **验证清单** | `docs/RK3588_VALIDATION_CHECKLIST.md` | 硬件验证指南 |

## ⚡ 常见问题

### **Q: 找不到模型文件**
```bash
# 检查模型位置
ls -la artifacts/models/

# 确认配置文件中的路径
grep "model:" config/detection/detect.yaml
```

### **Q: 演示脚本路径错误**
```bash
# 确保从项目根目录运行
cd /home/minsea01/dev/rk-projects/rk-app
./scripts/demo/demo_presentation_script.sh
```

### **Q: 相机连接失败**
```bash
# 重新启动假相机
sudo pkill -f arv-fake-gv-camera-0.10
sudo -E arv-fake-gv-camera-0.10 -i 127.0.0.1 >/tmp/arv_fake.log 2>&1 &
```

## 🎉 项目完整性验证

```bash
# 检查关键文件完整性
echo "✅ 模型文件:" && ls artifacts/models/*.rknn *.onnx
echo "✅ 配置文件:" && ls config/detection/ config/deploy/
echo "✅ 演示脚本:" && ls scripts/demo/
echo "✅ 测试日志:" && ls logs/
echo "✅ 技术文档:" && ls docs/
```

项目现在更加整洁和专业！🚀
