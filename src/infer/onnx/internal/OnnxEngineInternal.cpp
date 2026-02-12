#include "OnnxEngineInternal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>

#include "rkapp/common/StringUtils.hpp"
#include "rkapp/common/log.hpp"
#include "rkapp/infer/RknnDecodeUtils.hpp"

namespace rkapp::infer::onnx_internal {

namespace {

std::string stripComments(const std::string& content) {
  std::string out;
  out.reserve(content.size());
  bool in_string = false;
  bool escape = false;
  for (size_t i = 0; i < content.size(); ++i) {
    char c = content[i];
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
      char next = content[i + 1];
      if (next == '/') {
        i += 1;
        while (i + 1 < content.size() && content[i + 1] != '\n') {
          ++i;
        }
        continue;
      }
      if (next == '*') {
        i += 1;
        while (i + 1 < content.size()) {
          if (content[i] == '*' && content[i + 1] == '/') {
            i += 1;
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

std::string trim(const std::string& s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
    ++start;
  }
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
    --end;
  }
  return s.substr(start, end - start);
}

std::string sanitizeKey(const std::string& key) {
  std::string out;
  out.reserve(key.size());
  for (char c : key) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out;
}

std::string escapeRegex(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\':
      case '.':
      case '^':
      case '$':
      case '|':
      case '(':
      case ')':
      case '[':
      case ']':
      case '{':
      case '}':
      case '*':
      case '+':
      case '?':
        out.push_back('\\');
        break;
      default:
        break;
    }
    out.push_back(c);
  }
  return out;
}

bool parseFloatValue(const std::string& value, float& out) {
  try {
    const std::string trimmed = trim(value);
    size_t idx = 0;
    float parsed = std::stof(trimmed, &idx);
    if (idx == 0 || !std::isfinite(parsed) || parsed <= 0.0f) {
      return false;
    }
    out = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool parseIntValue(const std::string& value, int32_t& out) {
  try {
    const std::string trimmed = trim(value);
    size_t idx = 0;
    int32_t parsed = static_cast<int32_t>(std::stoi(trimmed, &idx));
    if (idx == 0) {
      return false;
    }
    out = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool parseBoolValue(const std::string& value, int& out) {
  const std::string lowered = rkapp::common::toLowerCopy(trim(value));
  if (lowered == "1" || lowered == "true") {
    out = 1;
    return true;
  }
  if (lowered == "0" || lowered == "false") {
    out = 0;
    return true;
  }
  return false;
}

bool parseIntListValue(const std::string& value, std::vector<int>& out) {
  std::string normalized = value;
  for (char& c : normalized) {
    if (c == '[' || c == ']' || c == ',' || c == ';') {
      c = ' ';
    }
  }
  std::stringstream ss(normalized);
  std::string token;
  bool any = false;
  while (ss >> token) {
    int32_t parsed = 0;
    if (parseIntValue(token, parsed)) {
      out.push_back(static_cast<int>(parsed));
      any = true;
    }
  }
  return any;
}

bool parseFloatField(const std::string& content, const std::string& key, float& out) {
  try {
    const std::string escaped = escapeRegex(key);
    std::regex re("(^|[^A-Za-z0-9_])\\\"?" + escaped +
                      "\\\"?\\s*[:=]\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)",
                  std::regex::icase);
    std::smatch m;
    if (std::regex_search(content, m, re) && m.size() > 2) {
      return parseFloatValue(m[2].str(), out);
    }
  } catch (const std::regex_error&) {
  }
  return false;
}

bool parseBoolField(const std::string& content, const std::string& key, int& out) {
  try {
    const std::string escaped = escapeRegex(key);
    std::regex re("(^|[^A-Za-z0-9_])\\\"?" + escaped +
                      "\\\"?\\s*[:=]\\s*(true|false|0|1)",
                  std::regex::icase);
    std::smatch m;
    if (std::regex_search(content, m, re) && m.size() > 2) {
      return parseBoolValue(m[2].str(), out);
    }
  } catch (const std::regex_error&) {
  }
  return false;
}

bool parseStringField(const std::string& content, const std::string& key, std::string& out) {
  try {
    const std::string escaped = escapeRegex(key);
    std::regex re("(^|[^A-Za-z0-9_])\\\"?" + escaped +
                      "\\\"?\\s*[:=]\\s*\\\"?([A-Za-z_]+)\\\"?",
                  std::regex::icase);
    std::smatch m;
    if (std::regex_search(content, m, re) && m.size() > 2) {
      out = rkapp::common::toLowerCopy(m[2].str());
      return true;
    }
  } catch (const std::regex_error&) {
  }
  return false;
}

bool parseIntListField(const std::string& content, const std::string& key, std::vector<int>& out) {
  try {
    const std::string escaped = escapeRegex(key);
    std::regex re("(^|[^A-Za-z0-9_])\\\"?" + escaped + "\\\"?\\s*[:=]\\s*\\[([^\\]]+)\\]",
                  std::regex::icase);
    std::smatch m;
    if (std::regex_search(content, m, re) && m.size() > 2) {
      return parseIntListValue(m[2].str(), out);
    }
  } catch (const std::regex_error&) {
  }
  return false;
}

bool parseIntField(const std::string& content, const std::string& key, int32_t& out) {
  try {
    const std::string escaped = escapeRegex(key);
    std::regex re("(^|[^A-Za-z0-9_])\\\"?" + escaped + "\\\"?\\s*[:=]\\s*([-+]?[0-9]+)",
                  std::regex::icase);
    std::smatch m;
    if (std::regex_search(content, m, re) && m.size() > 2) {
      return parseIntValue(m[2].str(), out);
    }
  } catch (const std::regex_error&) {
  }
  return false;
}

QuantParams parseQuantFromText(const std::string& content,
                               const std::vector<std::string>& scale_keys,
                               const std::vector<std::string>& zero_keys) {
  QuantParams params;
  for (const auto& key : scale_keys) {
    float value = 0.0f;
    if (parseFloatField(content, key, value)) {
      params.scale = value;
      params.has_scale = true;
      break;
    }
  }
  for (const auto& key : zero_keys) {
    int32_t value = 0;
    if (parseIntField(content, key, value)) {
      params.zero_point = value;
      params.has_zero_point = true;
      break;
    }
  }
  return params;
}

QuantParams parseQuantFromMetadata(const Ort::Session& session,
                                   Ort::AllocatorWithDefaultOptions& allocator,
                                   const std::vector<std::string>& scale_keys,
                                   const std::vector<std::string>& zero_keys) {
  QuantParams params;
  try {
    auto metadata = session.GetModelMetadata();
    auto keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
    if (keys.empty()) {
      return params;
    }
    std::unordered_map<std::string, std::string> meta_map;
    meta_map.reserve(keys.size());
    for (const auto& key_alloc : keys) {
      std::string key = key_alloc.get();
      auto val_alloc = metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
      if (val_alloc) {
        meta_map.emplace(std::move(key), val_alloc.get());
      }
    }

    auto lookupValue = [&](const std::vector<std::string>& candidates, std::string& out) -> bool {
      for (const auto& key : candidates) {
        auto it = meta_map.find(key);
        if (it != meta_map.end()) {
          out = it->second;
          return true;
        }
      }
      return false;
    };

    std::string value;
    if (lookupValue(scale_keys, value)) {
      float parsed = 0.0f;
      if (parseFloatValue(value, parsed)) {
        params.scale = parsed;
        params.has_scale = true;
      }
    }
    if (lookupValue(zero_keys, value)) {
      int32_t parsed = 0;
      if (parseIntValue(value, parsed)) {
        params.zero_point = parsed;
        params.has_zero_point = true;
      }
    }
  } catch (const Ort::Exception&) {
  }
  return params;
}

QuantParams parseQuantFromSidecar(const std::string& model_path,
                                  const std::vector<std::string>& scale_keys,
                                  const std::vector<std::string>& zero_keys) {
  const std::vector<std::string> candidates = {
      model_path + ".json", model_path + ".meta", "artifacts/models/decode_meta.json"};
  for (const auto& path : candidates) {
    std::ifstream f(path);
    if (!f.is_open()) {
      continue;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (content.empty()) {
      continue;
    }
    QuantParams params = parseQuantFromText(stripComments(content), scale_keys, zero_keys);
    if (params.has_scale || params.has_zero_point) {
      return params;
    }
  }
  return QuantParams{};
}

DecodeMeta parseDecodeMetaFromText(const std::string& content) {
  DecodeMeta meta;

  int32_t iv = 0;
  int bv = -1;
  std::string sv;

  if (parseStringField(content, "head", sv) && (sv == "dfl" || sv == "raw")) {
    meta.head = sv;
  }
  if (parseIntField(content, "reg_max", iv) && iv > 0) {
    meta.reg_max = static_cast<int>(iv);
  }

  if (parseIntField(content, "num_classes", iv) && iv > 0) {
    meta.num_classes = static_cast<int>(iv);
  } else if (parseIntField(content, "classes", iv) && iv > 0) {
    meta.num_classes = static_cast<int>(iv);
  } else if (parseIntField(content, "nc", iv) && iv > 0) {
    meta.num_classes = static_cast<int>(iv);
  }

  if (parseBoolField(content, "has_objectness", bv)) {
    meta.has_objectness = bv;
  } else if (parseBoolField(content, "objectness", bv)) {
    meta.has_objectness = bv;
  } else if (parseBoolField(content, "has_obj", bv)) {
    meta.has_objectness = bv;
  }

  std::vector<int> strides;
  if (parseIntListField(content, "strides", strides) && !strides.empty()) {
    meta.strides = std::move(strides);
  }

  return meta;
}

void mergeDecodeMetaMissingFields(DecodeMeta& dst, const DecodeMeta& src) {
  if (dst.reg_max <= 0 && src.reg_max > 0) {
    dst.reg_max = src.reg_max;
  }
  if (dst.strides.empty() && !src.strides.empty()) {
    dst.strides = src.strides;
  }
  if (dst.head.empty() && !src.head.empty()) {
    dst.head = src.head;
  }
  if (dst.num_classes <= 0 && src.num_classes > 0) {
    dst.num_classes = src.num_classes;
  }
  if (dst.has_objectness < 0 && src.has_objectness >= 0) {
    dst.has_objectness = src.has_objectness;
  }
}

bool resolveLayout(const std::vector<int64_t>& shape, int& N, int& C) {
  if (shape.size() == 3) {
    const int64_t d1 = shape[1];
    const int64_t d2 = shape[2];
    if (d1 >= 4 && d1 <= 4096 && d2 > d1) {
      C = static_cast<int>(d1);
      N = static_cast<int>(d2);
      return true;
    }
    if (d2 >= 4 && d2 <= 4096 && d1 > d2) {
      C = static_cast<int>(d2);
      N = static_cast<int>(d1);
      return true;
    }
    if (d1 >= 4 && d2 >= 4) {
      C = static_cast<int>(std::min(d1, d2));
      N = static_cast<int>(std::max(d1, d2));
      return true;
    }
    return false;
  }
  if (shape.size() == 2) {
    const int64_t d1 = shape[0];
    const int64_t d2 = shape[1];
    if (d2 >= 4) {
      N = static_cast<int>(d1);
      C = static_cast<int>(d2);
      return true;
    }
    if (d1 >= 4) {
      N = static_cast<int>(d2);
      C = static_cast<int>(d1);
      return true;
    }
  }
  return false;
}

}  // namespace

QuantParams resolveOutputQuantParams(const std::string& model_path,
                                     const std::string& output_name,
                                     const Ort::Session& session,
                                     Ort::AllocatorWithDefaultOptions& allocator) {
  std::string base_name = output_name;
  auto pos = base_name.find(':');
  if (pos != std::string::npos) {
    base_name = base_name.substr(0, pos);
  }
  const std::string sanitized = sanitizeKey(output_name);
  const std::string sanitized_base = sanitizeKey(base_name);

  const std::vector<std::string> scale_keys = {
      output_name + "_scale",      base_name + "_scale",       sanitized + "_scale",
      sanitized_base + "_scale",   "output_scale",             "quant_scale",
      "quantization_scale"};
  const std::vector<std::string> zero_keys = {
      output_name + "_zero_point",    output_name + "_zero",     base_name + "_zero_point",
      base_name + "_zero",            sanitized + "_zero_point",  sanitized_base + "_zero_point",
      "output_zero_point",            "quant_zero_point",         "quantization_zero_point"};

  QuantParams params = parseQuantFromMetadata(session, allocator, scale_keys, zero_keys);
  QuantParams sidecar = parseQuantFromSidecar(model_path, scale_keys, zero_keys);
  if (!params.has_scale && sidecar.has_scale) {
    params.scale = sidecar.scale;
    params.has_scale = true;
  }
  if (!params.has_zero_point && sidecar.has_zero_point) {
    params.zero_point = sidecar.zero_point;
    params.has_zero_point = true;
  }
  return params;
}

DecodeMeta parseDecodeMetaFromSidecar(const std::string& model_path) {
  const std::vector<std::string> candidates = {
      model_path + ".json", model_path + ".meta", "artifacts/models/decode_meta.json"};

  DecodeMeta meta;
  for (const auto& path : candidates) {
    std::ifstream f(path);
    if (!f.is_open()) {
      continue;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (content.empty()) {
      continue;
    }
    DecodeMeta parsed = parseDecodeMetaFromText(stripComments(content));
    if (parsed.hasAny()) {
      mergeDecodeMetaMissingFields(meta, parsed);
      break;
    }
  }
  return meta;
}

std::vector<Detection> parseOutput(Ort::Value& output,
                                   const rkapp::preprocess::LetterboxInfo& letterbox_info,
                                   cv::Size original_size,
                                   int input_size,
                                   const DecodeParams& params,
                                   const DecodeMeta& decode_meta,
                                   const QuantParams& quant_params,
                                   bool& unsupported_model) {
  std::vector<Detection> detections;
  auto type_info = output.GetTensorTypeAndShapeInfo();
  auto shape = type_info.GetShape();
  ONNXTensorElementDataType data_type = type_info.GetElementType();

  float* data = nullptr;
  std::vector<float> converted_data;

  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      data = output.GetTensorMutableData<float>();
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      auto fp16_data = output.GetTensorData<Ort::Float16_t>();
      size_t total = type_info.GetElementCount();
      converted_data.resize(total);
      for (size_t i = 0; i < total; ++i) {
        converted_data[i] = static_cast<float>(fp16_data[i]);
      }
      data = converted_data.data();
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      auto int8_data = output.GetTensorData<int8_t>();
      size_t total = type_info.GetElementCount();
      converted_data.resize(total);
      const float default_scale = 1.0f / 127.0f;
      const float scale = (quant_params.has_scale && quant_params.scale > 0.0f)
                              ? quant_params.scale
                              : default_scale;
      const float zero = quant_params.has_zero_point ? static_cast<float>(quant_params.zero_point)
                                                     : 0.0f;
      for (size_t i = 0; i < total; ++i) {
        converted_data[i] = (static_cast<float>(int8_data[i]) - zero) * scale;
      }
      data = converted_data.data();
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      auto uint8_data = output.GetTensorData<uint8_t>();
      size_t total = type_info.GetElementCount();
      converted_data.resize(total);
      const float default_scale = 1.0f / 127.0f;
      const float scale = (quant_params.has_scale && quant_params.scale > 0.0f)
                              ? quant_params.scale
                              : default_scale;
      const float zero = quant_params.has_zero_point ? static_cast<float>(quant_params.zero_point)
                                                     : 128.0f;
      for (size_t i = 0; i < total; ++i) {
        converted_data[i] = (static_cast<float>(uint8_data[i]) - zero) * scale;
      }
      data = converted_data.data();
      break;
    }
    default: {
      LOGE("OnnxEngine: Unsupported output data type: ", static_cast<int>(data_type));
      unsupported_model = true;
      return detections;
    }
  }

  int N = 0, C = 0;
  if (!resolveLayout(shape, N, C)) {
    LOGW("OnnxEngine: unsupported output shape");
    unsupported_model = true;
    return detections;
  }

  bool channels_first = false;
  if (shape.size() == 3) {
    const int64_t d1 = shape[1], d2 = shape[2];
    channels_first = (d1 == C && d2 == N);
  } else if (shape.size() == 2) {
    channels_first = false;
  }

  auto at = [&](int c, int i) -> float {
    if (channels_first) return data[c * N + i];
    return data[i * C + c];
  };

  auto sigmoid = [](float x) {
    if (x >= 0.0f) {
      return 1.0f / (1.0f + std::exp(-x));
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
  };

  enum class DecodeHead { kUnknown, kDfl, kRaw };
  DecodeHead decode_head = DecodeHead::kUnknown;
  if (decode_meta.head == "dfl") {
    decode_head = DecodeHead::kDfl;
  } else if (decode_meta.head == "raw") {
    decode_head = DecodeHead::kRaw;
  } else {
    bool dfl_candidate = false;
    bool raw_candidate = false;

    if (decode_meta.reg_max > 0) {
      const int cls_ch = C - 4 * decode_meta.reg_max;
      dfl_candidate = cls_ch > 0;
      if (dfl_candidate && decode_meta.num_classes > 0 && cls_ch != decode_meta.num_classes) {
        dfl_candidate = false;
      }
    }

    if (decode_meta.has_objectness >= 0) {
      const int cls_offset = decode_meta.has_objectness == 1 ? 5 : 4;
      const int cls_ch = C - cls_offset;
      raw_candidate = cls_ch >= 0;
      if (raw_candidate && decode_meta.num_classes > 0 && cls_ch != decode_meta.num_classes) {
        raw_candidate = false;
      }
    } else if (decode_meta.num_classes > 0) {
      raw_candidate = (C == (4 + decode_meta.num_classes) || C == (5 + decode_meta.num_classes));
    } else {
      raw_candidate = (C >= 5 && C < 64);
    }

    if (dfl_candidate == raw_candidate) {
      LOGE("OnnxEngine: decode head ambiguous/unknown (C=", C, ", N=", N,
           "). Provide decode metadata (head/reg_max/num_classes/has_objectness).");
      unsupported_model = true;
      return detections;
    }
    decode_head = dfl_candidate ? DecodeHead::kDfl : DecodeHead::kRaw;
  }

  if (decode_head == DecodeHead::kDfl) {
    int reg_max = decode_meta.reg_max;
    if (reg_max <= 0 && decode_meta.num_classes > 0) {
      const int remain = C - decode_meta.num_classes;
      if (remain > 0 && remain % 4 == 0) {
        reg_max = remain / 4;
      }
    }
    if (reg_max <= 0) {
      LOGE("OnnxEngine: DFL decode requires reg_max metadata");
      unsupported_model = true;
      return detections;
    }
    constexpr int kMaxRegMax = 32;
    if (reg_max > kMaxRegMax) {
      LOGE("OnnxEngine: reg_max=", reg_max, " exceeds max supported=", kMaxRegMax);
      unsupported_model = true;
      return detections;
    }

    const int cls_ch = C - 4 * reg_max;
    if (cls_ch <= 0) {
      LOGE("OnnxEngine: invalid DFL channel layout (C=", C, ", reg_max=", reg_max, ")");
      unsupported_model = true;
      return detections;
    }
    if (decode_meta.num_classes > 0 && cls_ch != decode_meta.num_classes) {
      LOGE("OnnxEngine: DFL class channels mismatch (tensor=", cls_ch,
           ", meta=", decode_meta.num_classes, ")");
      unsupported_model = true;
      return detections;
    }

    std::vector<int> strides = decode_meta.strides;
    if (strides.empty() && !resolve_stride_set(input_size, N, strides)) {
      LOGE("OnnxEngine: DFL decode cannot resolve strides (N=", N, ", input=", input_size, ")");
      unsupported_model = true;
      return detections;
    }
    AnchorLayout layout = build_anchor_layout(input_size, N, strides);
    if (!layout.valid) {
      LOGE("OnnxEngine: DFL layout invalid for anchors=", N);
      unsupported_model = true;
      return detections;
    }

    std::vector<float> probs(static_cast<size_t>(reg_max), 0.0f);
    auto dfl_softmax_project = [&](int i) -> std::array<float, 4> {
      std::array<float, 4> out{};
      for (int side = 0; side < 4; ++side) {
        const int ch0 = side * reg_max;
        float maxv = -1e30f;
        for (int k = 0; k < reg_max; ++k) {
          maxv = std::max(maxv, at(ch0 + k, i));
        }
        float denom = 0.f;
        for (int k = 0; k < reg_max; ++k) {
          probs[k] = std::exp(at(ch0 + k, i) - maxv);
          denom += probs[k];
        }
        float proj = 0.f;
        if (denom > 0.f) {
          for (int k = 0; k < reg_max; ++k) {
            proj += probs[k] * static_cast<float>(k);
          }
          proj /= denom;
        }
        out[side] = proj;
      }
      return out;
    };

    int limit = N;
    if (params.max_boxes > 0) {
      limit = std::min(limit, params.max_boxes);
    }

    for (int i = 0; i < limit; ++i) {
      auto dfl = dfl_softmax_project(i);
      const float stride = layout.stride_map[i];
      const float l = dfl[0] * stride;
      const float t = dfl[1] * stride;
      const float r = dfl[2] * stride;
      const float b = dfl[3] * stride;
      const float x1 = layout.anchor_cx[i] - l;
      const float y1 = layout.anchor_cy[i] - t;
      const float x2 = layout.anchor_cx[i] + r;
      const float y2 = layout.anchor_cy[i] + b;

      float best_conf = 0.f;
      int best_cls = 0;
      for (int c = 0; c < cls_ch; ++c) {
        float conf = sigmoid(at(4 * reg_max + c, i));
        if (conf > best_conf) {
          best_conf = conf;
          best_cls = c;
        }
      }
      if (best_conf < params.conf_thres) {
        continue;
      }

      Detection det;
      const float scale = letterbox_info.scale;
      const float dx = letterbox_info.dx;
      const float dy = letterbox_info.dy;
      det.x = (x1 - dx) / scale;
      det.y = (y1 - dy) / scale;
      det.w = (x2 - x1) / scale;
      det.h = (y2 - y1) / scale;
      det.x = std::max(0.0f, std::min(det.x, static_cast<float>(original_size.width)));
      det.y = std::max(0.0f, std::min(det.y, static_cast<float>(original_size.height)));
      det.w =
          std::max(0.0f, std::min(det.w, static_cast<float>(original_size.width) - det.x));
      det.h =
          std::max(0.0f, std::min(det.h, static_cast<float>(original_size.height) - det.y));
      det.confidence = best_conf;
      det.class_id = best_cls;
      det.class_name = "class_" + std::to_string(best_cls);
      detections.push_back(det);
    }
  } else {
    bool has_objness = false;
    if (decode_meta.has_objectness >= 0) {
      has_objness = decode_meta.has_objectness == 1;
    } else if (decode_meta.num_classes > 0) {
      if (C == 5 + decode_meta.num_classes) {
        has_objness = true;
      } else if (C == 4 + decode_meta.num_classes) {
        has_objness = false;
      } else {
        LOGE("OnnxEngine: RAW channel layout mismatch with num_classes metadata");
        unsupported_model = true;
        return detections;
      }
    } else {
      has_objness = (C - 5) >= 0;
    }

    const int cls_offset = has_objness ? 5 : 4;
    int num_classes = decode_meta.num_classes > 0 ? decode_meta.num_classes : (C - cls_offset);
    if (num_classes < 0 || cls_offset + num_classes != C) {
      LOGE("OnnxEngine: invalid RAW class channel layout (C=", C, ")");
      unsupported_model = true;
      return detections;
    }
    if (num_classes == 0 && !has_objness) {
      LOGE("OnnxEngine: invalid RAW output without classes/objectness (C=", C, ")");
      unsupported_model = true;
      return detections;
    }

    int limit = N;
    if (params.max_boxes > 0) {
      limit = std::min(limit, params.max_boxes);
    }

    for (int i = 0; i < limit; ++i) {
      const float cx = at(0, i);
      const float cy = at(1, i);
      const float w = at(2, i);
      const float h = at(3, i);

      float obj = 1.0f;
      if (has_objness) {
        obj = sigmoid(at(4, i));
      }

      float best_conf = 1.0f;
      int best_cls = 0;
      if (num_classes > 0) {
        best_conf = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
          float conf = sigmoid(at(cls_offset + c, i));
          if (conf > best_conf) {
            best_conf = conf;
            best_cls = c;
          }
        }
      }

      const float combined = obj * best_conf;
      if (combined < params.conf_thres) {
        continue;
      }

      Detection det;
      const float scale = letterbox_info.scale;
      const float dx = letterbox_info.dx;
      const float dy = letterbox_info.dy;
      det.x = (cx - w / 2.0f - dx) / scale;
      det.y = (cy - h / 2.0f - dy) / scale;
      det.w = w / scale;
      det.h = h / scale;
      det.x = std::max(0.0f, std::min(det.x, static_cast<float>(original_size.width)));
      det.y = std::max(0.0f, std::min(det.y, static_cast<float>(original_size.height)));
      det.w =
          std::max(0.0f, std::min(det.w, static_cast<float>(original_size.width) - det.x));
      det.h =
          std::max(0.0f, std::min(det.h, static_cast<float>(original_size.height) - det.y));
      det.confidence = combined;
      det.class_id = best_cls;
      det.class_name = "class_" + std::to_string(best_cls);
      detections.push_back(det);
    }
  }

  unsupported_model = false;
  return detections;
}

}  // namespace rkapp::infer::onnx_internal
