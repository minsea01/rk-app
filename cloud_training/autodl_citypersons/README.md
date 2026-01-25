# AutoDL CityPersons 训练包

## 目标
- YOLOv8n 行人检测
- mAP@0.5 >= 90%
- 模型大小 < 5MB (RKNN INT8)

## 快速开始 (推荐)

```bash
# 1. 上传此文件夹到 AutoDL
scp -r -P <端口> autodl_citypersons root@<地址>:/root/

# 2. SSH 连接后执行
cd /root/autodl_citypersons
chmod +x *.sh
./one_click_train.sh
```

## 脚本说明

| 脚本 | 说明 | 时间 |
|------|------|------|
| `one_click_train.sh` | **推荐** - 一键训练 (COCO Person) | 2-4h |
| `START_HERE.sh` | 使用 WiderPerson 数据集 | 3-5h |
| `train_citypersons.sh` | 核心训练逻辑 | - |
| `export_model.sh` | 导出 ONNX | 1min |

## 数据集选择

| 数据集 | 图片数 | 预期 mAP | 推荐度 |
|--------|--------|----------|--------|
| COCO Person | ~64k | 85-88% | ⭐⭐⭐ |
| CrowdHuman | ~15k | 88-92% | ⭐⭐⭐⭐ |
| WiderPerson | ~8k | 85-90% | ⭐⭐⭐ |
| COCO + CrowdHuman | ~80k | **90-95%** | ⭐⭐⭐⭐⭐ |

## 如果 mAP 不到 90%

1. **增加训练轮数**: `EPOCHS=200 ./one_click_train.sh`
2. **使用 CrowdHuman**: 下载并合并数据集
3. **调整学习率**: 在脚本中修改 `lr0=0.0005`

## 训练完成后

```bash
# 下载模型到本地
scp -P <端口> root@<地址>:/root/autodl-tmp/person_detection/outputs/yolov8n_person/weights/best.pt ./artifacts/models/
scp -P <端口> root@<地址>:/root/autodl-tmp/person_detection/outputs/yolov8n_person/weights/best.onnx ./artifacts/models/

# 本地 RKNN 转换
source ~/yolo_env/bin/activate
python3 tools/convert_onnx_to_rknn.py \
    --onnx artifacts/models/best.onnx \
    --out artifacts/models/yolov8n_person_int8.rknn \
    --calib datasets/coco/calib_images/calib.txt \
    --target rk3588
```

## 费用估算

- RTX 4090: ¥2.5-3/小时
- 预计训练时间: 2-4 小时
- 总费用: **约 ¥10-15**

## 常见问题

**Q: CUDA out of memory**
A: 减小 batch size: `BATCH=32 ./one_click_train.sh`

**Q: 训练中断**
A: 脚本支持断点续训，重新运行即可

**Q: 数据集下载慢**
A: AutoDL 通常已有 COCO，脚本会自动检测
