#pragma once

#include <vector>

namespace rkapp::infer {

struct AnchorLayout {
    std::vector<float> stride_map;
    std::vector<float> anchor_cx;
    std::vector<float> anchor_cy;
    bool valid = false;
};

bool resolve_stride_set(int img_size, int num_anchors, std::vector<int>& out_strides);
AnchorLayout build_anchor_layout(int img_size, int num_anchors, const std::vector<int>& strides);

} // namespace rkapp::infer

