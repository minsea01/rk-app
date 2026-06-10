#include "rkapp/infer/ModelMeta.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>

#include <yaml-cpp/yaml.h>

#include "rkapp/common/StringUtils.hpp"
#include "rkapp/common/log.hpp"

namespace rkapp::infer {
namespace {

std::string stripComments(const std::string& content) {
  std::string out;
  out.reserve(content.size());
  bool in_string = false;
  bool escape = false;
  for (size_t i = 0; i < content.size(); ++i) {
    const char c = content[i];
    if (in_string) {
      out.push_back(c);
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      out.push_back(c);
      continue;
    }

    if (c == '/' && i + 1 < content.size()) {
      const char next = content[i + 1];
      if (next == '/') {
        ++i;
        while (i + 1 < content.size() && content[i + 1] != '\n') {
          ++i;
        }
        continue;
      }
      if (next == '*') {
        ++i;
        while (i + 1 < content.size()) {
          if (content[i] == '*' && content[i + 1] == '/') {
            ++i;
            break;
          }
          ++i;
        }
        continue;
      }
    }

    if (c == '#') {
      while (i + 1 < content.size() && content[i + 1] != '\n') {
        ++i;
      }
      continue;
    }

    out.push_back(c);
  }
  return out;
}

std::string trimCopy(const std::string& input) {
  size_t begin = 0;
  size_t end = input.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(input[begin]))) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return input.substr(begin, end - begin);
}

std::string unquoteCopy(const std::string& input) {
  if (input.size() >= 2 &&
      ((input.front() == '"' && input.back() == '"') ||
       (input.front() == '\'' && input.back() == '\''))) {
    return input.substr(1, input.size() - 2);
  }
  return input;
}

// 键统一转小写存放；YAML::Node 内部共享底层存储，离开 root 作用域后仍然有效。
using MetaEntries = std::map<std::string, YAML::Node>;

MetaEntries collectMetaEntries(const std::string& sanitized) {
  MetaEntries entries;

  // 1) 整文档解析：覆盖 JSON sidecar 与 "key: value" 风格的 YAML。
  try {
    YAML::Node root = YAML::Load(sanitized);
    if (root && root.IsMap()) {
      for (const auto& kv : root) {
        try {
          entries[rkapp::common::toLowerCopy(kv.first.as<std::string>())] = kv.second;
        } catch (const YAML::Exception&) {
        }
      }
      return entries;
    }
  } catch (const YAML::Exception&) {
  }

  // 2) 行级回退：兼容 "key=value" 的松散 .meta 格式与轻度损坏的 JSON。
  std::istringstream lines(sanitized);
  std::string line;
  while (std::getline(lines, line)) {
    const size_t sep = line.find_first_of(":=");
    if (sep == std::string::npos) {
      continue;
    }
    const std::string key = unquoteCopy(trimCopy(line.substr(0, sep)));
    std::string value = trimCopy(line.substr(sep + 1));
    if (!value.empty() && value.back() == ',') {
      value.pop_back();
      value = trimCopy(value);
    }
    if (key.empty() || value.empty()) {
      continue;
    }
    try {
      entries[rkapp::common::toLowerCopy(key)] = YAML::Load(value);
    } catch (const YAML::Exception&) {
    }
  }
  return entries;
}

int parseIntField(const MetaEntries& entries, const std::vector<std::string>& keys) {
  for (const auto& key : keys) {
    const auto it = entries.find(key);
    if (it == entries.end()) {
      continue;
    }
    try {
      return it->second.as<int>();
    } catch (const YAML::Exception&) {
    }
  }
  return -1;
}

int parseBoolField(const MetaEntries& entries, const std::vector<std::string>& keys) {
  for (const auto& key : keys) {
    const auto it = entries.find(key);
    if (it == entries.end()) {
      continue;
    }
    try {
      const int numeric = it->second.as<int>();
      if (numeric == 0 || numeric == 1) {
        return numeric;
      }
      continue;
    } catch (const YAML::Exception&) {
    }
    try {
      return it->second.as<bool>() ? 1 : 0;
    } catch (const YAML::Exception&) {
    }
  }
  return -1;
}

std::vector<int> parseIntListField(const MetaEntries& entries, const std::string& key) {
  std::vector<int> out;
  const auto it = entries.find(key);
  if (it == entries.end() || !it->second.IsSequence()) {
    return out;
  }
  for (const auto& element : it->second) {
    try {
      const int parsed = element.as<int>();
      if (parsed > 0) {
        out.push_back(parsed);
      }
    } catch (const YAML::Exception&) {
    }
  }
  return out;
}

std::string parseEnumStringField(const MetaEntries& entries,
                                 const std::string& key,
                                 const std::vector<std::string>& allowed) {
  const auto it = entries.find(key);
  if (it == entries.end()) {
    return {};
  }
  try {
    const std::string lowered = rkapp::common::toLowerCopy(it->second.as<std::string>());
    if (std::find(allowed.begin(), allowed.end(), lowered) != allowed.end()) {
      return lowered;
    }
  } catch (const YAML::Exception&) {
  }
  return {};
}

std::vector<std::filesystem::path> modelSidecarCandidates(const std::filesystem::path& model_path) {
  return {
      std::filesystem::path(model_path.string() + ".json"),
      std::filesystem::path(model_path.string() + ".meta"),
  };
}

bool loadCandidateFile(const std::filesystem::path& path, ModelMeta* meta) {
  if (meta == nullptr) {
    return false;
  }

  std::ifstream handle(path);
  if (!handle.is_open()) {
    return false;
  }

  std::string content((std::istreambuf_iterator<char>(handle)), std::istreambuf_iterator<char>());
  if (content.empty()) {
    return false;
  }

  ModelMeta parsed = parseModelMetaText(content);
  if (!modelMetaHasAny(parsed)) {
    return false;
  }

  mergeModelMetaMissingFields(*meta, parsed);
  return true;
}

}  // namespace

bool ModelMeta::hasAny() const { return modelMetaHasAny(*this); }

bool modelMetaHasAny(const ModelMeta& meta) {
  return meta.reg_max > 0 || !meta.strides.empty() || !meta.head.empty() || meta.output_index >= 0 ||
         meta.num_classes > 0 || meta.has_objectness >= 0 || meta.score_is_probability >= 0 ||
         meta.coords_are_normalized >= 0 ||
         meta.task != "detect" || meta.num_keypoints > 0;
}

void mergeModelMetaMissingFields(ModelMeta& dst, const ModelMeta& src) {
  if (dst.reg_max <= 0 && src.reg_max > 0) {
    dst.reg_max = src.reg_max;
  }
  if (dst.strides.empty() && !src.strides.empty()) {
    dst.strides = src.strides;
  }
  if (dst.head.empty() && !src.head.empty()) {
    dst.head = src.head;
  }
  if (dst.output_index < 0 && src.output_index >= 0) {
    dst.output_index = src.output_index;
  }
  if (dst.num_classes <= 0 && src.num_classes > 0) {
    dst.num_classes = src.num_classes;
  }
  if (dst.has_objectness < 0 && src.has_objectness >= 0) {
    dst.has_objectness = src.has_objectness;
  }
  if (dst.score_is_probability < 0 && src.score_is_probability >= 0) {
    dst.score_is_probability = src.score_is_probability;
  }
  if (dst.coords_are_normalized < 0 && src.coords_are_normalized >= 0) {
    dst.coords_are_normalized = src.coords_are_normalized;
  }
  if (dst.task == "detect" && src.task != "detect") {
    dst.task = src.task;
  }
  if (dst.num_keypoints <= 0 && src.num_keypoints > 0) {
    dst.num_keypoints = src.num_keypoints;
  }
}

ModelMeta parseModelMetaText(const std::string& content) {
  const MetaEntries entries = collectMetaEntries(stripComments(content));

  ModelMeta meta;
  const int reg_max = parseIntField(entries, {"reg_max"});
  if (reg_max > 0) {
    meta.reg_max = reg_max;
  }

  auto strides = parseIntListField(entries, "strides");
  if (!strides.empty()) {
    meta.strides = std::move(strides);
  }

  meta.head = parseEnumStringField(entries, "head", {"raw", "dfl"});

  int output_index = parseIntField(entries, {"output_index", "output_idx"});
  if (output_index >= 0) {
    meta.output_index = output_index;
  }

  const int num_classes = parseIntField(entries, {"num_classes", "classes", "nc"});
  if (num_classes > 0) {
    meta.num_classes = num_classes;
  }

  const int has_objectness = parseBoolField(entries, {"has_objectness", "objectness", "has_obj"});
  if (has_objectness >= 0) {
    meta.has_objectness = has_objectness;
  }

  const int score_is_probability =
      parseBoolField(entries,
                     {"score_is_probability", "scores_are_probabilities", "score_is_prob"});
  if (score_is_probability >= 0) {
    meta.score_is_probability = score_is_probability;
  }

  const int coords_are_normalized = parseBoolField(
      entries, {"coords_are_normalized", "normalized_coords", "coords_normalized"});
  if (coords_are_normalized >= 0) {
    meta.coords_are_normalized = coords_are_normalized;
  }

  const std::string task = parseEnumStringField(entries, "task", {"detect", "pose", "segment"});
  if (!task.empty()) {
    meta.task = task;
  }

  const int num_keypoints = parseIntField(entries, {"num_keypoints", "kpt_shape"});
  if (num_keypoints > 0) {
    meta.num_keypoints = num_keypoints;
  }
  return meta;
}

ModelMetaLoadResult loadModelMetaFromPath(const std::string& model_path) {
  ModelMetaLoadResult result;
  const std::filesystem::path absolute_model = std::filesystem::absolute(model_path);
  const auto candidates = modelSidecarCandidates(absolute_model);

  std::vector<std::string> loaded_sources;
  for (const auto& candidate : candidates) {
    if (loadCandidateFile(candidate, &result.meta)) {
      loaded_sources.push_back(candidate.string());
    }
  }

  if (!loaded_sources.empty()) {
    std::ostringstream joined;
    for (size_t i = 0; i < loaded_sources.size(); ++i) {
      if (i > 0) {
        joined << ';';
      }
      joined << loaded_sources[i];
    }
    result.source_path = joined.str();
  }
  return result;
}

}  // namespace rkapp::infer
