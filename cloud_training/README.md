# AutoDL 4090 云端训练指南

## 目标: 90% mAP 行人检测

### 当前进度
- ✅ 80% mAP 已达成 (COCO Person)
- 🎯 目标: 90% mAP
- ⚠️ 已知问题: `cache=ram` 导致 RAM 溢出

---

## ⚡ 快速开始 (推荐流程)

```bash
# SSH连接AutoDL后
cd ~/cloud_training
chmod +x *.sh

# 1. 配置环境
./setup_autodl.sh

# 2. 筛选COCO Person (如未准备)
python3 filter_coco_person.py --coco-root /root/autodl-tmp/coco

# 3. 合并CrowdHuman + COCO (可选，提高精度)
./merge_datasets.sh

# 4. 开始训练 (使用优化版脚本，避免RAM溢出)
./train_90map_optimized.sh

# 5. 导出ONNX
./export_onnx.sh
```

---

## 可用脚本

| 脚本 | 用途 | 预计时间 | RAM安全 |
|------|------|----------|---------|
| `train_90map_optimized.sh` | **推荐** - 90% mAP训练 | 6-10小时 | ✅ |
| `train_90map.sh` | 高精度训练 (旧版) | 4-8小时 | ⚠️ cache=ram |
| `train_coco_extreme.sh` | 极限训练 | 8-12小时 | ⚠️ cache=ram |
| `train.sh` | 基础训练 (80% mAP) | 2-4小时 | ✅ |
| `merge_datasets.sh` | 合并COCO+CrowdHuman | 10分钟 | - |
| `filter_coco_person.py` | 筛选COCO Person类 | 5分钟 | - |

---

## RAM溢出问题解决

### AutoDL 配置
- **CPU**: 25 vCPU Xeon Platinum 8470Q
- **GPU**: RTX 5090 (32GB 显存) - batch=160 可用
- **RAM**: 90GB
- **数据盘**: 50GB (可扩容)
- **价格**: ¥2.78/时

### RAM爆的根本原因
**COCO + CrowdHuman 联合训练 `cache=ram` 需要 ~108GB，超出 90GB RAM！**

```
数据集: 79,000 张 × 1.17MB = 90GB 基础
+ 预处理缓冲: 18GB
= 总需求: ~108GB > 90GB ❌
```

### 解决方案
`train_90map_optimized.sh` 使用：
```bash
cache=disk     # ✅ 使用磁盘缓存，避免RAM溢出
workers=8      # ✅ 减少worker降低峰值内存
batch=128      # ✅ 26GB显存支持batch=128
```

### 如果仅用 COCO Person (不合并CrowdHuman)
```bash
# COCO Person 64k 图片 ≈ 77GB，90GB RAM 刚好够用
CACHE_MODE=ram ./train_90map_optimized.sh
```

---

## 数据集配置

### 已有资源
- **完整COCO**: `/root/autodl-tmp/coco`
- **CrowdHuman**: 已下载

### 数据集准备

#### 方案1: 仅COCO Person (简单)
```bash
python3 filter_coco_person.py --coco-root /root/autodl-tmp/coco
# 输出: datasets/coco_person/ (~64k训练图)
```

#### 方案2: COCO + CrowdHuman (推荐，更高精度)
```bash
# 1. 准备COCO Person
python3 filter_coco_person.py --coco-root /root/autodl-tmp/coco

# 2. 确保CrowdHuman已准备 (YOLO格式)
# datasets/crowdhuman/train/images/
# datasets/crowdhuman/train/labels/

# 3. 合并数据集
./merge_datasets.sh
# 输出: datasets/merged/ (~80k训练图)
```

### 预期精度

| 数据集 | 训练图片 | 预期mAP50 |
|--------|----------|-----------|
| COCO Person | ~64k | 85-88% |
| COCO + CrowdHuman | ~80k | **90-92%** ✅ |
| + WiderPerson | ~100k | 92-95% |

---

## 训练参数说明

| 参数 | 优化版 | 说明 |
|------|--------|------|
| epochs | 300 | 更长训练 |
| batch | 160 | 32GB显存可用160-192 |
| cache | disk | COCO+CrowdHuman需要disk |
| workers | 10 | 25 vCPU Xeon |
| lr0 | 0.0005 | 微调学习率 |
| patience | 80 | 更大耐心值 |
| mosaic | 1.0 | 马赛克增强 |
| mixup | 0.15 | 混合增强 |
| copy_paste | 0.1 | 复制粘贴增强 |

### 环境变量覆盖
```bash
# 自定义配置
EPOCHS=200 BATCH=64 ./train_90map_optimized.sh

# 仅COCO Person + RAM缓存 (更快)
CACHE_MODE=ram ./train_90map_optimized.sh
```

---

## 训练监控

```bash
# 查看训练进度
tail -f outputs/yolov8n_pedestrian_90/results.csv

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看内存使用
watch -n 5 free -h
```

---

## 训练中断恢复

```bash
# 从last.pt继续训练
yolo detect train \
    model=outputs/yolov8n_pedestrian_90/weights/last.pt \
    resume=True
```

---

## 下载训练结果

```bash
# 本地执行
scp root@<autodl_ip>:/root/autodl-tmp/pedestrian_training/outputs/yolov8n_pedestrian_90/weights/best.pt ./artifacts/models/
scp root@<autodl_ip>:/root/autodl-tmp/pedestrian_training/outputs/yolov8n_pedestrian_90/weights/best.onnx ./artifacts/models/
```

---

## 本地RKNN转换

```bash
source ~/yolo_env/bin/activate
python3 tools/convert_onnx_to_rknn.py \
    --onnx artifacts/models/best.onnx \
    --out artifacts/models/yolov8n_pedestrian_int8.rknn \
    --calib datasets/coco/calib_images/calib.txt \
    --target rk3588
```

---

## 费用估算

- RTX 4090: ¥2.5-3/小时
- 达到90% mAP预计: **¥20-30** (6-10小时)

---

## 常见问题

**Q: RAM溢出 / 进程被kill**
A: 使用 `train_90map_optimized.sh` 替代旧脚本

**Q: CUDA out of memory**
A: 减小batch size: `BATCH=32 ./train_90map_optimized.sh`

**Q: 数据集下载慢**
A: 使用AutoDL的数据盘或对象存储预先上传

**Q: 训练中断**
A: 使用 `resume=True` 从 last.pt 继续

**Q: mAP不到90%**
A:
1. 合并更多数据 (CrowdHuman + WiderPerson)
2. 增加epochs到500
3. 尝试更大模型 (YOLOv8s)
