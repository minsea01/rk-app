#pragma once

namespace rkapp::preprocess {

// letterbox 缩放补边参数：用于把模型输入坐标系的结果映射回原图坐标系。
// 独立成头文件，供推理接口(IInferEngine)在不引入整个 Preprocess 的情况下使用。
struct LetterboxInfo {
    float scale;
    float dx, dy;
    int new_width, new_height;
};

}  // namespace rkapp::preprocess
