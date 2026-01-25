# 为什么Python比C++快？深度分析 (2026-01-12)

## 问题：C++ 30.85 FPS vs Python 40.2 FPS

**意外结果：** Python比C++快了**30%**！这违反了常识，需要深入分析。

---

## 关键发现：测试条件不同 ⚠️

### 测试差异对比

| 维度 | C++ (video_infer_rga) | Python (test_video_multiclass.py) | 影响 |
|------|----------------------|----------------------------------|------|
| **处理帧数** | 647帧 (全视频) | **100帧 (限制)** | 🔴 **关键差异** |
| **输入尺寸** | 640×640 | **416×416** | 🔴 **关键差异** |
| **预处理** | RGA硬件加速 | OpenCV CPU letterbox | 不同 |
| **推理API** | rknn_api (C) | rknnlite (Python绑定) | 不同 |
| **后处理** | 无（仅计时） | 包含postprocess | 不同 |

---

## 差异一：输入尺寸 🔴 **最大影响**

### C++ 使用 640×640
```bash
./video_infer_rga artifacts/models/yolov8n_person_80map_int8.rknn \
  assets/traffic_video.mp4 640  # ← 640×640

结果: 32.42ms/frame (30.85 FPS)
```

### Python 使用 416×416
```python
img_input, ratio, (dw, dh) = letterbox(frame, 416)  # ← 416×416

结果: 24.90ms/frame (40.2 FPS)
```

### 计算量对比

**输入像素数：**
- 640×640 = **409,600** 像素
- 416×416 = **173,056** 像素
- 差异：409,600 / 173,056 = **2.37倍**

**理论推理时间比例：**
```
假设线性缩放（实际更复杂）:
24.90ms × 2.37 ≈ 59ms （640×640理论时间）

实际C++: 32.42ms (RGA + 其他优化)
```

**结论：** 输入尺寸是主要因素！416×416计算量只有640×640的42%

---

## 差异二：处理帧数 🟡 **次要影响**

### C++ 处理全部647帧
```
Frames 0-50:    27.46 ms
Frames 50-100:  27.81 ms
Frames 100-150: 27.63 ms
...
Frames 600-647: 31.91 ms  ← 后期变慢（热节流）
```

**观察：** 后期帧延迟增加 (27ms → 32ms)，可能是：
- 热节流 (thermal throttling)
- 内存压力
- 缓存失效

### Python 仅处理100帧
```python
while frame_count < max_frames:  # max_frames=100
    ...

平均: 24.90ms (全程冷启动状态)
```

**结论：** Python测试未进入热节流阶段

---

## 差异三：预处理开销 ⚠️ **RGA未必更快？**

### C++ 使用 RGA 硬件加速
```cpp
cv::Mat letterbox_rga(const cv::Mat& src, int target_size, double& preprocess_time) {
    // RGA resize
    rga_buffer_t src_buf, dst_buf;
    ret = imresize(src_buf, dst_buf);  // 硬件加速

    // OpenCV padding (仍然CPU)
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, ...);
}

平均预处理: 2.61ms
```

**问题：**
1. RGA resize确实快
2. 但padding仍用OpenCV CPU
3. RGA初始化/同步有开销

### Python 使用 OpenCV CPU
```python
def letterbox(img, new_size=416):
    # CPU resize + padding
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, ...)

预处理时间: 包含在24.90ms总时间中（未单独统计）
```

**416×416预处理更快：**
- 输入768×432 → 416×416 (缩小到54%)
- 输入768×432 → 640×640 (缩小到83%)

---

## 差异四：API层级差异 🟢 **轻微影响**

### C++ 使用底层 rknn_api
```cpp
// 每帧都需要：
rknn_inputs_set(ctx, 1, inputs);  // 设置输入
rknn_run(ctx, nullptr);            // 运行推理
// 未获取输出（测试中省略）

开销: 函数调用 + 内存拷贝
```

### Python 使用封装的 rknnlite
```python
outputs = rknn.inference(inputs=[img_input])

# 内部封装了:
# - rknn_inputs_set
# - rknn_run
# - rknn_outputs_get

开销: Python函数调用 + C扩展
```

**观察：** Python绑定可能有批量优化

---

## 实验验证：公平对比

### 方案1：C++ 使用 416×416

```bash
ssh root@192.168.137.226
cd /root/rk-app

# 使用416×416重新测试
./video_infer_rga artifacts/models/yolov8n_person_80map_int8.rknn \
  assets/traffic_video.mp4 416  # ← 改为416
```

**预期结果：** C++ FPS应该提升到 ~38-42 FPS

### 方案2：Python 使用 640×640

```bash
# 修改Python代码
img_input, ratio, (dw, dh) = letterbox(frame, 640)  # ← 改为640
```

**预期结果：** Python FPS应该下降到 ~30-35 FPS

---

## 让我运行实验验证

让我实际运行这两个实验：
