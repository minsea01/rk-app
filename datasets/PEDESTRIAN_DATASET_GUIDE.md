# 行人检测数据集准备指南

本指南说明如何准备用于RK3588行人检测模型训练的数据集。

## 方案选择

### 方案A：COCO Person（推荐用于快速验证）

**优点**：
- 数据集成熟，标注质量高
- 场景多样（室内/室外/不同光照）
- 下载方便，社区支持好

**数据规模**：
- 训练集：约64k张包含person的图像
- 验证集：约5k张包含person的图像
- 平均每张图2-3个行人

**预期精度**：
- YOLOv8n: mAP@0.5 约85-88%
- YOLOv8s: mAP@0.5 约90-93% ✅

### 方案B：CrowdHuman（适合密集场景）

**优点**：
- 专注行人检测
- 密集场景（平均22.6人/张）
- 遮挡处理能力强

**数据规模**：
- 训练集：15k张
- 验证集：4k张

**适用场景**：人群密集的商场、车站等

## 快速开始：COCO Person数据集

### 步骤1：下载COCO 2017数据集

```bash
# 下载完整COCO数据集（约20GB，需要时间）
bash scripts/download_coco.sh datasets/coco_raw

# 或手动下载（可断点续传）
mkdir -p datasets/coco_raw && cd datasets/coco_raw
wget http://images.cocodataset.org/zips/train2017.zip      # 18GB
wget http://images.cocodataset.org/zips/val2017.zip        # 1GB
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip  # 241MB

# 解压
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

### 步骤2：筛选Person类别并转换格式

```bash
# 激活环境
source ~/yolo_env/bin/activate

# 处理数据集（仅生成标注，不复制图像 - 节省磁盘空间）
python tools/prepare_coco_person.py \
  --coco-dir datasets/coco_raw \
  --output-dir datasets/coco_person

# 或者复制图像到输出目录（需要额外约20GB空间）
python tools/prepare_coco_person.py \
  --coco-dir datasets/coco_raw \
  --output-dir datasets/coco_person \
  --copy-images
```

**输出目录结构**：
```
datasets/coco_person/
├── data.yaml                # YOLO配置文件
├── images/
│   ├── train/              # 训练集图像（如果使用--copy-images）
│   └── val/                # 验证集图像
├── labels/
│   ├── train/              # YOLO格式标注（0 x_center y_center w h）
│   └── val/
├── train_images.txt        # 训练集图像列表
└── val_images.txt          # 验证集图像列表
```

### 步骤3：验证数据集

```bash
# 检查data.yaml
cat datasets/coco_person/data.yaml

# 统计数据集规模
echo "训练集图像数量: $(wc -l < datasets/coco_person/train_images.txt)"
echo "验证集图像数量: $(wc -l < datasets/coco_person/val_images.txt)"

# 查看标注示例
head -5 datasets/coco_person/labels/train/*.txt | head -20
```

### 步骤4：可视化验证（可选）

```bash
# 使用工具可视化标注
python tools/visualize_yolo_annotations.py \
  --data datasets/coco_person/data.yaml \
  --num-samples 10 \
  --output artifacts/vis/coco_person_samples
```

## 高级选项

### 使用部分数据快速验证

如果不想下载完整数据集，可以使用现有的300张COCO校准图像：

```bash
# 从现有校准集提取person标注
python tools/prepare_coco_person.py \
  --coco-dir datasets/coco \
  --output-dir datasets/coco_person_mini \
  --subset calib_images
```

### 自定义校准集

```bash
# 从训练集中抽取300张图像作为RKNN量化校准集
python tools/make_calib_set.py \
  --data datasets/coco_person/data.yaml \
  --output datasets/coco_person_calib \
  --num 300
```

## 数据集质量检查

### 运行数据健康检查

```bash
# 检查数据集完整性
python tools/dataset_health_check.py \
  --data datasets/coco_person/data.yaml

# 预期输出
# ✓ 图像与标注文件匹配
# ✓ 标注格式正确
# ✓ 边界框在图像范围内
# ✓ 类别ID有效
```

## 常见问题

### Q1: 下载速度太慢怎么办？

**A**: 使用国内镜像或云盘下载：
- 清华镜像：https://mirrors.tuna.tsinghua.edu.cn/
- 百度云盘：搜索"COCO 2017"

### Q2: 磁盘空间不足？

**A**: 不使用`--copy-images`选项，只生成标注文件：
```bash
python tools/prepare_coco_person.py \
  --coco-dir datasets/coco_raw \
  --output-dir datasets/coco_person
  # 不加 --copy-images
```

然后在`data.yaml`中修改图像路径指向原始COCO目录。

### Q3: 如何只使用一部分数据训练？

**A**: 修改生成的图像列表文件：
```bash
# 只保留前5000张训练图像
head -5000 datasets/coco_person/train_images.txt > datasets/coco_person/train_images_mini.txt
```

然后在训练时使用自定义列表。

## 下一步

数据集准备完成后，参考 [训练指南](../scripts/train/README.md) 开始模型训练：

```bash
# 使用COCO person数据集训练YOLOv8s
bash scripts/train/train_pedestrian.sh
```

## 参考链接

- COCO数据集官网：https://cocodataset.org/
- CrowdHuman数据集：https://www.crowdhuman.org/
- YOLO格式说明：https://docs.ultralytics.com/datasets/
