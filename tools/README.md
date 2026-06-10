# Tools

本目录只保留可复用的训练、导出、转换、评估和数据处理工具。一次性实验脚本、
旧工业数据集脚本和生成结果不放在源码树里。

## Model Export And Conversion

```bash
python tools/export_yolov8_to_onnx.py --weights yolo11n.pt --imgsz 416

python tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/yolo11n_coco80_416.onnx \
  --out artifacts/models/yolo11n_coco80_416_int8.rknn \
  --calib datasets/calib
```

## Evaluation

```bash
python tools/model_evaluation.py \
  --model runs/train/person/weights/best.pt \
  --data datasets/coco_person/data.yaml

python tools/dataset_health_check.py --data datasets/coco_person/data.yaml
python tools/compare.py --help
```

## Runtime Utilities

```bash
python tools/http_receiver.py --port 9000
python tools/http_post.py --url http://127.0.0.1:9000/ingest --file artifacts/bench_summary.json
python tools/make_fake_camera_video.py --help
```

## Notes

- 大文件和运行输出放到忽略目录：`artifacts/`、`runs/`、`datasets/`、`output/`。
- 板端部署和演示入口在 `scripts/deploy/` 与 `scripts/demo/`。
- 训练入口统一为 `scripts/train.sh`，旧的重复 wrapper 已移除。
