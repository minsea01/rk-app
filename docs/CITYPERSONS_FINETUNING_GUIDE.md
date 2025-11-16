## CityPersons Fine-tuning Guide

**Goal:** Achieve ≥90% mAP@0.5 on COCO person validation set

**Current Baseline:** 61.57% mAP@0.5 (YOLO11n pretrained)

**Expected After Fine-tuning:** 85-92% mAP@0.5

---

## Quick Start (TL;DR)

```bash
# 1. Download dataset (manual, requires registration)
bash scripts/datasets/download_citypersons.sh

# 2. Convert to YOLO format
python scripts/datasets/prepare_citypersons.py

# 3. Train (2-4 hours on RTX 3060)
bash scripts/train/train_citypersons.sh

# 4. Validate
python scripts/evaluation/official_yolo_map.py \
  --model runs/citypersons_finetune/yolo11n_citypersons/weights/best.pt \
  --annotations datasets/coco/annotations/person_val2017.json \
  --images-dir datasets/coco/val2017 \
  --output artifacts/yolo11n_finetuned_map.json
```

---

## Step 1: Download CityPersons Dataset

### Manual Download (Required)

CityPersons is built on top of CityScapes, which requires registration:

1. **Register** at https://www.cityscapes-dataset.com/register/
2. **Login** and go to Downloads
3. **Download these files:**
   - `leftImg8bit_trainvaltest.zip` (11GB) - Images
   - Download CityPersons annotations from https://github.com/cvgroup-njust/CityPersons

4. **Place files in:**
   ```
   datasets/citypersons/raw/
   ├── leftImg8bit_trainvaltest.zip
   └── gtBboxCityPersons.zip
   ```

5. **Run extraction script:**
   ```bash
   bash scripts/datasets/download_citypersons.sh
   ```

### Expected Directory Structure

```
datasets/citypersons/
├── raw/
│   ├── leftImg8bit/
│   │   ├── train/  (2975 images)
│   │   └── val/    (500 images)
│   └── gtBboxCityPersons/
│       ├── train/  (annotations)
│       └── val/    (annotations)
└── yolo/          (created in step 2)
```

---

## Step 2: Convert Annotations to YOLO Format

### Run Conversion

```bash
python scripts/datasets/prepare_citypersons.py
```

### What This Does

1. **Reads CityPersons JSON annotations**
   - Filters for "person" class only
   - Uses `bboxVis` (visible part) to handle occlusions

2. **Converts to YOLO format**
   - Format: `<class_id> <x_center> <y_center> <width> <height>`
   - Normalized to [0, 1]
   - class_id=0 for person

3. **Splits dataset**
   - Train: ~2,975 images, ~19,654 persons
   - Val: ~500 images, ~3,157 persons

4. **Creates dataset YAML**
   - Path: `datasets/citypersons/yolo/citypersons.yaml`

### Expected Output

```
datasets/citypersons/yolo/
├── train/
│   ├── images/  (2975 .png files)
│   └── labels/  (2975 .txt files)
├── val/
│   ├── images/  (500 .png files)
│   └── labels/  (500 .txt files)
└── citypersons.yaml
```

### Verify Dataset

```bash
# Check a random sample
python -c "
from pathlib import Path
import random

train_imgs = list(Path('datasets/citypersons/yolo/train/images').glob('*.png'))
sample = random.choice(train_imgs)
label_file = sample.parent.parent / 'labels' / f'{sample.stem}.txt'

print(f'Image: {sample}')
print(f'Label: {label_file}')
print(f'Exists: {label_file.exists()}')

if label_file.exists():
    print(f'\\nAnnotations:')
    print(label_file.read_text())
"
```

---

## Step 3: Fine-tune YOLO11n

### Training Configuration

```bash
# File: scripts/train/train_citypersons.sh

Model:    yolo11n.pt (starting point)
Dataset:  citypersons.yaml
Epochs:   50 (with early stopping patience=10)
Batch:    16
ImgSize:  640x640
Device:   GPU 0 (or cpu)
```

### Run Training

```bash
bash scripts/train/train_citypersons.sh
```

### Expected Training Time

- **RTX 3060 (12GB):** 2-4 hours
- **RTX 4090:** 1-2 hours
- **CPU (not recommended):** 12-24 hours

### Monitor Training

```bash
# Real-time monitoring
tail -f runs/citypersons_finetune/yolo11n_citypersons/train.log

# View training plots
ls runs/citypersons_finetune/yolo11n_citypersons/
# Plots: results.png, confusion_matrix.png, PR_curve.png, F1_curve.png
```

### Expected Training Metrics

| Epoch | mAP@0.5 | Precision | Recall |
|-------|---------|-----------|--------|
| 1     | ~50%    | ~70%      | ~60%   |
| 10    | ~75%    | ~85%      | ~75%   |
| 20    | ~85%    | ~90%      | ~82%   |
| 30    | ~88%    | ~92%      | ~85%   |
| Best  | **90%+**| **93%+**  | **87%+**|

---

## Step 4: Validate Fine-tuned Model

### On CityPersons Val Set

Training automatically validates on CityPersons val set. Check final results:

```bash
cat runs/citypersons_finetune/yolo11n_citypersons/results.csv | tail -1
```

### On COCO Person Test Set (Official Evaluation)

```bash
python scripts/evaluation/official_yolo_map.py \
  --model runs/citypersons_finetune/yolo11n_citypersons/weights/best.pt \
  --annotations datasets/coco/annotations/person_val2017.json \
  --images-dir datasets/coco/val2017 \
  --imgsz 640 \
  --output artifacts/yolo11n_citypersons_finetuned_map.json
```

### Expected Results

**Before Fine-tuning (baseline):**
- mAP@0.5: 61.57%
- Precision: 84.18%
- Recall: 65.03%

**After Fine-tuning (target):**
- mAP@0.5: **≥90%** ✅
- Precision: **≥92%**
- Recall: **≥85%**

### Compare Results

```bash
# View comparison
cat << 'EOF'
| Metric      | Baseline | Fine-tuned | Improvement |
|-------------|----------|------------|-------------|
| mAP@0.5     | 61.57%   | ≥90%       | +28.43%     |
| Precision   | 84.18%   | ≥92%       | +7.82%      |
| Recall      | 65.03%   | ≥85%       | +19.97%     |
EOF

cat artifacts/yolo11n_official_full_map.json | jq '.metrics'
cat artifacts/yolo11n_citypersons_finetuned_map.json | jq '.metrics'
```

---

## Step 5: Export to ONNX → RKNN

### Export to ONNX

```bash
yolo export \
  model=runs/citypersons_finetune/yolo11n_citypersons/weights/best.pt \
  format=onnx \
  opset=12 \
  simplify=True \
  imgsz=640

# Output: best.onnx
```

### Convert to RKNN

```bash
python tools/convert_onnx_to_rknn.py \
  --onnx runs/citypersons_finetune/yolo11n_citypersons/weights/best.onnx \
  --out artifacts/models/yolo11n_citypersons_int8.rknn \
  --calib datasets/coco/calib_images/calib.txt \
  --target rk3588 \
  --do-quant
```

### Validate ONNX vs RKNN Accuracy

```bash
# Should match within 5%
python scripts/compare_onnx_rknn.py \
  --onnx runs/citypersons_finetune/yolo11n_citypersons/weights/best.onnx \
  --rknn artifacts/models/yolo11n_citypersons_int8.rknn \
  --image datasets/coco/val2017/000000000139.jpg
```

---

## Troubleshooting

### Issue 1: Out of Memory (OOM) during training

**Solution:** Reduce batch size

```bash
# Edit scripts/train/train_citypersons.sh
BATCH=8  # or 4 for low-memory GPUs
```

### Issue 2: Training not improving after epoch 20

**Solution:** Adjust learning rate

```bash
# Lower learning rate for better convergence
LR0=0.005  # instead of 0.01
```

### Issue 3: mAP still <90% after fine-tuning

**Possible causes:**
1. **Dataset too small** → Add COCO person training images
2. **Overfitting** → Increase data augmentation
3. **Underfitting** → Train more epochs or use YOLO11s/m

**Solution A: Combined dataset**
```bash
# Merge CityPersons + COCO person training
python scripts/datasets/merge_datasets.py \
  --dataset1 datasets/citypersons/yolo \
  --dataset2 datasets/coco_person/yolo \
  --output datasets/citypersons_coco/yolo
```

**Solution B: Use larger model**
```bash
# Use YOLO11s instead of YOLO11n
MODEL="yolo11s.pt"
# Expected: 92-95% mAP@0.5 (but larger size: ~9MB)
```

### Issue 4: CityPersons download failed

**Alternative: Use COCO Person subset for quick testing**

```bash
# Prepare COCO person-only dataset
python scripts/datasets/prepare_coco_person.py

# Train (expect 72-78% mAP, not 90%)
bash scripts/train/train_coco_person.sh
```

---

## Alternative: Quick Test with COCO Person Subset

If CityPersons download is blocked or too slow, you can start with COCO person subset:

```bash
# 1. Prepare COCO person training set
python scripts/datasets/prepare_coco_person.py

# 2. Train
yolo train \
  model=yolo11n.pt \
  data=datasets/coco_person/coco_person.yaml \
  epochs=30 \
  imgsz=640 \
  batch=16

# 3. Expected: 72-78% mAP@0.5 (still below 90%)
```

**Note:** COCO-only training won't reach 90% due to:
- Same domain as validation set (risk of overfitting)
- Limited diversity compared to CityPersons
- Smaller dataset size

---

## Timeline Estimate

| Task | Duration | Status |
|------|----------|--------|
| Download CityPersons | 1-2 hours | Manual |
| Convert annotations | 10 minutes | Automated |
| Fine-tune training | 2-4 hours | Automated |
| Validation | 5 minutes | Automated |
| ONNX → RKNN | 10 minutes | Automated |
| **Total** | **4-7 hours** | - |

---

## Success Criteria

✅ **Phase 1: Dataset Preparation**
- [ ] CityPersons downloaded (11GB)
- [ ] Annotations converted to YOLO format
- [ ] Train: ~2975 images, Val: ~500 images
- [ ] Dataset YAML created

✅ **Phase 2: Fine-tuning**
- [ ] Training completed (50 epochs)
- [ ] Best weights saved
- [ ] Training mAP curve shows convergence

✅ **Phase 3: Validation** (CRITICAL)
- [ ] mAP@0.5 ≥ 90% on COCO person val set ✅
- [ ] Precision ≥ 92%
- [ ] Recall ≥ 85%

✅ **Phase 4: Deployment**
- [ ] ONNX export successful
- [ ] RKNN conversion successful
- [ ] ONNX vs RKNN accuracy gap < 5%

---

## Next Steps After Reaching 90% mAP

1. **Update graduation thesis** with fine-tuning results
2. **Run full acceptance tests** (FPS, dual-NIC, etc.)
3. **Deploy to RK3588 board** (when hardware arrives)
4. **Document final model** (weights, metrics, deployment guide)

---

## Files Created

- `scripts/datasets/download_citypersons.sh` - Download helper
- `scripts/datasets/prepare_citypersons.py` - Annotation converter
- `scripts/train/train_citypersons.sh` - Training script
- `docs/CITYPERSONS_FINETUNING_GUIDE.md` - This guide

**Start here:** `bash scripts/datasets/download_citypersons.sh`
