#pragma once

#include <vector>

namespace rkapp::infer {

// DFL 头解码时使用的 anchor 布局缓存。
struct AnchorLayout {
    std::vector<float> stride_map;
    std::vector<float> anchor_cx;
    std::vector<float> anchor_cy;
    bool valid = false;
};

// 根据输入尺寸和 anchor 数量推断 stride 组合。
bool resolve_stride_set(int img_size, int num_anchors, std::vector<int>& out_strides);
// 根据 stride 组合构建每个 anchor 的中心点与步长映射。
AnchorLayout build_anchor_layout(int img_size, int num_anchors, const std::vector<int>& strides);

} // namespace rkapp::infer
