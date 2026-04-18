#include "rkapp/infer/ModelMeta.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <utility>

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

int parseIntField(const std::string& content, const std::vector<std::string>& keys) {
  for (const auto& key : keys) {
    try {
      std::smatch match;
      const std::string pattern =
          std::string("(?:^|[^A-Za-z0-9_])\\\"?") + key + "\\\"?\\s*[:=]\\s*(-?\\d+)";
      std::regex re(pattern, std::regex::icase);
      if (std::regex_search(content, match, re) && match.size() > 1) {
        return std::stoi(match[1].str());
      }
    } catch (const std::exception&) {
    }
  }
  return -1;
}

int parseBoolField(const std::string& content, const std::vector<std::string>& keys) {
  for (const auto& key : keys) {
    try {
      std::smatch match;
      const std::string pattern =
          std::string("(?:^|[^A-Za-z0-9_])\\\"?") + key +
          "\\\"?\\s*[:=]\\s*(true|false|0|1)";
      std::regex re(pattern, std::regex::icase);
      if (std::regex_search(content, match, re) && match.size() > 1) {
        const std::string value = rkapp::common::toLowerCopy(match[1].str());
        if (value == "1" || value == "true") {
          return 1;
        }
        if (value == "0" || value == "false") {
          return 0;
        }
      }
    } catch (const std::exception&) {
    }
  }
  return -1;
}

std::vector<int> parseIntListField(const std::string& content, const std::string& key) {
  std::vector<int> out;
  try {
    std::smatch match;
    const std::string pattern =
        std::string("(?:^|[^A-Za-z0-9_])\\\"?") + key + "\\\"?\\s*[:=]\\s*\\[([^\\]]+)\\]";
    std::regex re(pattern, std::regex::icase);
    if (!std::regex_search(content, match, re) || match.size() < 2) {
      return out;
    }
    std::stringstream ss(match[1].str());
    std::string token;
    while (std::getline(ss, token, ',')) {
      try {
        const int parsed = std::stoi(token);
        if (parsed > 0) {
          out.push_back(parsed);
        }
      } catch (const std::exception&) {
      }
    }
  } catch (const std::exception&) {
  }
  return out;
}

std::string parseEnumStringField(const std::string& content,
                                 const std::string& key,
                                 const std::vector<std::string>& allowed) {
  try {
    std::smatch match;
    const std::string pattern =
        std::string("(?:^|[^A-Za-z0-9_])\\\"?") + key +
        "\\\"?\\s*[:=]\\s*\\\"?([A-Za-z_]+)\\\"?";
    std::regex re_quoted(pattern, std::regex::icase);
    if (std::regex_search(content, match, re_quoted) && match.size() > 1) {
      const std::string lowered = rkapp::common::toLowerCopy(match[1].str());
      if (std::find(allowed.begin(), allowed.end(), lowered) != allowed.end()) {
        return lowered;
      }
    }
  } catch (const std::exception&) {
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
         meta.num_classes > 0 || meta.has_objectness >= 0 || meta.task != "detect" ||
         meta.num_keypoints > 0;
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
  if (dst.task == "detect" && src.task != "detect") {
    dst.task = src.task;
  }
  if (dst.num_keypoints <= 0 && src.num_keypoints > 0) {
    dst.num_keypoints = src.num_keypoints;
  }
}

ModelMeta parseModelMetaText(const std::string& content) {
  const std::string sanitized = stripComments(content);

  ModelMeta meta;
  const int reg_max = parseIntField(sanitized, {"reg_max"});
  if (reg_max > 0) {
    meta.reg_max = reg_max;
  }

  auto strides = parseIntListField(sanitized, "strides");
  if (!strides.empty()) {
    meta.strides = std::move(strides);
  }

  meta.head = parseEnumStringField(sanitized, "head", {"raw", "dfl"});

  int output_index = parseIntField(sanitized, {"output_index", "output_idx"});
  if (output_index >= 0) {
    meta.output_index = output_index;
  }

  const int num_classes = parseIntField(sanitized, {"num_classes", "classes", "nc"});
  if (num_classes > 0) {
    meta.num_classes = num_classes;
  }

  const int has_objectness = parseBoolField(sanitized, {"has_objectness", "objectness", "has_obj"});
  if (has_objectness >= 0) {
    meta.has_objectness = has_objectness;
  }

  const std::string task = parseEnumStringField(sanitized, "task", {"detect", "pose", "segment"});
  if (!task.empty()) {
    meta.task = task;
  }

  const int num_keypoints = parseIntField(sanitized, {"num_keypoints", "kpt_shape"});
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
