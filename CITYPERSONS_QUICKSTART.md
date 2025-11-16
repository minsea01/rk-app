# CityPersons Fine-tuning Quick Start

**Goal:** Achieve ≥90% mAP@0.5 (currently 61.57%)

**Time Required:** 4-7 hours total

---

## 📋 Prerequisites

✅ YOLO11n baseline completed (61.57% mAP)
✅ GPU available (RTX 3060 or better recommended)
⏸️ CityPersons dataset (~11GB download)

---

## 🚀 Quick Start (3 Commands)

### Step 1: Download Dataset (Manual)

1. Register at https://www.cityscapes-dataset.com/register/
2. Download these files:
   - `leftImg8bit_trainvaltest.zip` (11GB)
   - `gtBboxCityPersons.zip` from https://github.com/cvgroup-njust/CityPersons
3. Place in `datasets/citypersons/raw/`
4. Run extraction:

```bash
bash scripts/datasets/download_citypersons.sh
```

### Step 2: Convert Annotations (~10 mins)

```bash
python scripts/datasets/prepare_citypersons.py
```

**Expected output:**
- Train: ~2975 images, ~19654 persons
- Val: ~500 images, ~3157 persons

### Step 3: Fine-tune (~2-4 hours)

```bash
bash scripts/train/train_citypersons.sh
```

**Monitor training:**
```bash
tail -f runs/citypersons_finetune/yolo11n_citypersons/train.log
```

### Step 4: Validate

```bash
python scripts/evaluation/official_yolo_map.py \
  --model runs/citypersons_finetune/yolo11n_citypersons/weights/best.pt \
  --annotations datasets/coco/annotations/person_val2017.json \
  --images-dir datasets/coco/val2017 \
  --output artifacts/yolo11n_finetuned_map.json

cat artifacts/yolo11n_finetuned_map.json | jq '.metrics."mAP@0.5"'
# Expected: >= 0.90 (90%)
```

---

## 📊 Expected Results

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| mAP@0.5 | 61.57% | **90%+** ✅ | ≥90% |
| Precision | 84.18% | **92%+** | ≥92% |
| Recall | 65.03% | **85%+** | ≥85% |

---

## 🔧 Troubleshooting

**GPU Out of Memory?**
```bash
# Edit scripts/train/train_citypersons.sh
BATCH=8  # Reduce from 16
```

**No GPU?**
```bash
# Edit scripts/train/train_citypersons.sh
DEVICE="cpu"  # Warning: 12-24 hours training time
```

**Can't download CityPersons?**
```bash
# Alternative: Use COCO person subset (72-78% mAP only)
python scripts/datasets/prepare_coco_person.py
bash scripts/train/train_coco_person.sh
```

---

## 📁 Files Created

- `scripts/datasets/download_citypersons.sh` - Extraction helper
- `scripts/datasets/prepare_citypersons.py` - YOLO converter
- `scripts/train/train_citypersons.sh` - Training pipeline
- `docs/CITYPERSONS_FINETUNING_GUIDE.md` - Full documentation

---

## ⏭️ Next Steps

After reaching 90% mAP:

1. Export to ONNX:
   ```bash
   yolo export model=runs/.../best.pt format=onnx
   ```

2. Convert to RKNN:
   ```bash
   python tools/convert_onnx_to_rknn.py --onnx best.onnx --out yolo11n_citypersons.rknn
   ```

3. Update graduation thesis with results

---

**Full Guide:** `docs/CITYPERSONS_FINETUNING_GUIDE.md`

**Start Now:** `bash scripts/datasets/download_citypersons.sh`
