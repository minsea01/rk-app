#pragma once

#include <string>
#include <vector>

namespace rkapp::infer {

struct ModelMeta {
  int reg_max = -1;
  std::vector<int> strides;
  std::string head;             // "dfl" / "raw" / ""
  int output_index = -1;        // 多输出模型时，指定用于检测的输出分支
  int num_classes = -1;         // 类别数
  int has_objectness = -1;      // -1 未知，0 无，1 有
  int score_is_probability = -1;  // -1 未知，0 需要 sigmoid，1 已是 [0,1] 概率
  int coords_are_normalized = -1;  // -1 未知，0 像素坐标，1 相对输入尺寸归一化
  std::string task{"detect"};   // "detect" | "pose" | "segment"
  int num_keypoints = 0;        // 关键点数量（纯检测时为 0）

  bool hasAny() const;
};

struct ModelMetaLoadResult {
  ModelMeta meta;
  std::string source_path;
};

bool modelMetaHasAny(const ModelMeta& meta);
void mergeModelMetaMissingFields(ModelMeta& dst, const ModelMeta& src);
ModelMeta parseModelMetaText(const std::string& content);
ModelMetaLoadResult loadModelMetaFromPath(const std::string& model_path);

}  // namespace rkapp::infer
