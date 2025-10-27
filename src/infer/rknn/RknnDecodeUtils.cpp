#include "rkapp/infer/RknnDecodeUtils.hpp"

#include <algorithm>
#include <cmath>

namespace rkapp::infer {

bool resolve_stride_set(int img_size, int num_anchors, std::vector<int>& out_strides) {
  const std::vector<std::vector<int>> candidates = {
      {8, 16, 32},
      {4, 8, 16, 32},
      {8, 16, 32, 64},
      {16, 32, 64},
      {8, 16},
      {8, 16, 32, 64, 128}
  };

  for (const auto& strides : candidates) {
    int total = 0;
    bool valid = true;
    for (int s : strides) {
      if (s <= 0 || img_size % s != 0) {
        valid = false;
        break;
      }
      const int fm = img_size / s;
      total += fm * fm;
    }
    if (valid && total == num_anchors) {
      out_strides = strides;
      return true;
    }
  }
  return false;
}

AnchorLayout build_anchor_layout(int img_size, int num_anchors, const std::vector<int>& strides) {
  AnchorLayout layout;
  layout.stride_map.resize(num_anchors, 0.0f);
  layout.anchor_cx.resize(num_anchors, 0.0f);
  layout.anchor_cy.resize(num_anchors, 0.0f);

  auto fill_with = [&](const std::vector<int>& order) -> bool {
    int idx = 0;
    for (int s : order) {
      if (s <= 0 || img_size % s != 0) {
        return false;
      }
      const int fm = img_size / s;
      for (int iy = 0; iy < fm; ++iy) {
        for (int ix = 0; ix < fm; ++ix) {
          if (idx >= num_anchors) {
            return false;
          }
          layout.stride_map[idx] = static_cast<float>(s);
          layout.anchor_cx[idx] = (ix + 0.5f) * s;
          layout.anchor_cy[idx] = (iy + 0.5f) * s;
          ++idx;
        }
      }
    }
    return idx == num_anchors;
  };

  if (fill_with(strides)) {
    layout.valid = true;
    return layout;
  }

  std::vector<int> reversed(strides.rbegin(), strides.rend());
  if (fill_with(reversed)) {
    layout.valid = true;
    return layout;
  }

  // Fallback: approximate uniform grid
  const int grid = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_anchors)))));
  const float stride = static_cast<float>(img_size) / std::max(1, grid);
  int idx = 0;
  for (int iy = 0; iy < grid && idx < num_anchors; ++iy) {
    for (int ix = 0; ix < grid && idx < num_anchors; ++ix) {
      layout.stride_map[idx] = stride;
      layout.anchor_cx[idx] = (ix + 0.5f) * stride;
      layout.anchor_cy[idx] = (iy + 0.5f) * stride;
      ++idx;
    }
  }
  layout.valid = false;
  return layout;
}

} // namespace rkapp::infer
