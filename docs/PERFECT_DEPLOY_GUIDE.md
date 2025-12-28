# RK3588 上板完美方案

## 核心原则：先获取版本，再转换模型

---

## 步骤 1：板子到手先跑版本检测

```bash
# 1. 把检测脚本传到板子
scp scripts/deploy/check_board_env.sh user@板子IP:/tmp/

# 2. 在板子上运行
ssh user@板子IP
chmod +x /tmp/check_board_env.sh
/tmp/check_board_env.sh > board_env.txt

# 3. 把结果传回来
scp user@板子IP:/tmp/board_env.txt ./
```

**这一步输出示例：**

```
=== RK3588 环境检测 ===
OS: Ubuntu 22.04.3 LTS
Kernel: 5.10.110-rockchip-rk3588
RKNPU Driver: 1.0.0
Python: 3.10.12
rknnlite: 1.6.0
```

---

## 步骤 2：根据版本重新转换模型

```bash
# 更新 PC 上的 rknn-toolkit2 到对应版本
# 假设板子是 1.6.0
pip install rknn-toolkit2==1.6.0

# 重新转换模型
python tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/yolo11n_416.onnx \
  --out artifacts/models/yolo11n_416_v160.rknn \
  --target rk3588
```

---

## 步骤 3：最小验证 (不用Docker)

```bash
# 1. 传输必要文件 (不是整个项目)
scp artifacts/models/yolo11n_416_v160.rknn user@板子:/tmp/
scp apps/yolov8_rknn_infer.py user@板子:/tmp/
scp -r apps/utils user@板子:/tmp/
scp assets/test.jpg user@板子:/tmp/

# 2. 板子上安装最小依赖
ssh user@板子
pip3 install numpy opencv-python-headless

# 3. 直接运行
cd /tmp
PYTHONPATH=. python3 yolov8_rknn_infer.py \
  --model yolo11n_416_v160.rknn \
  --source test.jpg \
  --save result.jpg

# 4. 看输出
# [INFO] Inference time: 22.5ms  ← 说明成功了！
```

---

## 步骤 4：完整部署

最小验证通过后，再考虑完整部署：

```bash
# 现在可以放心传整个项目了
bash scripts/deploy/pack_for_board.sh
scp rk-app-board-deploy.tar.gz user@板子:/home/user/

# 板子上
tar xzf rk-app-board-deploy.tar.gz
cd rk-app
bash scripts/deploy/install_dependencies.sh
```

---

## 版本对应表

| 板子 RKNN 驱动 | PC rknn-toolkit2 | rknn-toolkit-lite2 |
|---------------|------------------|-------------------|
| 0.9.x | 1.5.x | 1.5.x |
| 1.0.x | 1.6.x | 1.6.x |
| 2.0.x | 2.0.x | 2.0.x |

**关键：PC 转换版本 = 板子运行版本**

---

## 常见问题速查

### 问题1：rknn.inference() 返回 None

```
原因：模型版本与运行时不匹配
解决：重新用正确版本转换模型
```

### 问题2：init_runtime() 返回 -1

```
原因：NPU 设备无权限
解决：sudo chmod 666 /dev/rknpu0
      或者 sudo 运行脚本
```

### 问题3：Segmentation fault

```
原因：librknpu.so 版本不匹配
解决：确认板子系统镜像和 RKNN SDK 版本一致
```

### 问题4：推理速度很慢 (>100ms)

```
原因：模型中有算子回退到 CPU
解决：用 416x416 尺寸，避免动态 shape
      检查 rknn.build() 时的日志
```

---

## 时间线预估

| 阶段 | 理想 | 现实 |
|------|------|------|
| 版本检测 | 5分钟 | 5分钟 |
| 模型重转 | 10分钟 | 30分钟 |
| 最小验证 | 10分钟 | 1小时 |
| 完整部署 | 30分钟 | 2小时 |

**总计：最快 1 小时，正常 4 小时内搞定**
