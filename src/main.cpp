#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

#include "log.hpp"

#ifdef RKAPP_HAVE_YAMLCPP
#include <yaml-cpp/yaml.h>
#endif

namespace {

struct AppConfig {
  std::string log_level = "INFO";
};

std::string to_upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return s;
}

rklog::Level level_from_string(const std::string& level_name) {
  const std::string upper = to_upper(level_name);
  if (upper == "TRACE") return rklog::TRACE;
  if (upper == "DEBUG") return rklog::DEBUG;
  if (upper == "INFO") return rklog::INFO;
  if (upper == "WARN") return rklog::WARN;
  if (upper == "ERROR") return rklog::ERROR;
  std::cerr << "[WARN] Unknown log level '" << level_name
            << "', fallback to INFO" << std::endl;
  return rklog::INFO;
}

AppConfig load_config(const std::string& path) {
  AppConfig config;

#ifdef RKAPP_HAVE_YAMLCPP
  try {
    const YAML::Node node = YAML::LoadFile(path);
    if (node["log_level"]) {
      config.log_level = node["log_level"].as<std::string>();
    }
  } catch (const YAML::BadFile&) {
    std::cerr << "[WARN] Config file not found: " << path
              << ". Using defaults." << std::endl;
  } catch (const YAML::Exception& ex) {
    std::cerr << "[WARN] Failed to parse config '" << path
              << "': " << ex.what() << ". Using defaults." << std::endl;
  }
#else
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "[WARN] Config file not found: " << path
              << ". Using defaults." << std::endl;
    return config;
  }

  std::string key;
  while (f >> key) {
    if (key == "log_level:") {
      std::string value;
      if (f >> value) {
        config.log_level = value;
      }
    }
  }
#endif

  return config;
}

}  // namespace

int main(int argc, char** argv) {
  std::string cfg = "config/app.yaml";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "-c" || a == "--config") && i + 1 < argc) cfg = argv[++i];
  }

  const AppConfig config = load_config(cfg);
  const rklog::Level level = level_from_string(config.log_level);
  rklog::g_level.store(level, std::memory_order_relaxed);

  LOGI("Hello RK3588 (x86/QEMU template). cfg=", cfg,
       ", log_level=", config.log_level);
  return 0;
}
