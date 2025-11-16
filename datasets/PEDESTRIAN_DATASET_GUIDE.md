# 行人检测数据集准备指南

本指南用于满足毕业设计要求：mAP@0.5 >90% on pedestrian detection dataset

## 方案对比

| 数据集 | 规模 | 场景 | 预期mAP@0.5 | 下载难度 | 推荐度 |
|--------|------|------|-------------|----------|--------|
| COCO person | 2693张(val) | 通用 | 70-85% | 简单 | ⭐⭐ |
| CityPersons | 2975张 | 城市监控 | 85-92% | 中等 | ⭐⭐⭐⭐⭐ |
| WiderPerson | 8000张 | 密集人群 | 90-95% | 困难 | ⭐⭐⭐⭐ |

**推荐：CityPersons**（城市场景，易达标，符合RK3588工业应用定位）

---

## 快速开始：COCO Person（今天就能跑）

### 1. 下载数据

```bash
cd ~/rk-app
bash scripts/datasets/prepare_coco_person.sh
```

### 2. 评估mAP

```bash
python scripts/evaluation/pedestrian_map_evaluator.py \
  --model artifacts/models/best.rknn \
  --annotations datasets/coco/annotations/person_val2017.json \
  --images datasets/coco/val2017
```

**预期结果：** mAP@0.5 = 70-85%

**如果<90%** → 使用CityPersons数据集

---

## 保险方案：CityPersons（推荐用于答辩）

### 下载
官方：https://www.cityscapes-dataset.com/downloads/
需要：leftImg8bit_trainvaltest.zip (11GB) + gtBboxCityPersons_trainval.zip (2MB)

### 微调（如需要）
YOLO11n在CityPersons上微调50 epochs，mAP@0.5通常达到88-92%

**时间规划：** 3-5天完成数据准备+微调+评估
