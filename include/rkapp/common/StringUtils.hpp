#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace rkapp::common {

inline std::string toLowerCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

}  // namespace rkapp::common
