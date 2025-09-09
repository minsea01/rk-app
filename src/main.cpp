#include <fstream>
#include <iostream>
#include <string>
#include "log.hpp"

static void set_log_level(const std::string& s){
  using L = rklog::Level;
  if (s == "TRACE") rklog::g_level = L::TRACE;
  else if (s == "DEBUG") rklog::g_level = L::DEBUG;
  else if (s == "INFO") rklog::g_level = L::INFO;
  else if (s == "WARN") rklog::g_level = L::WARN;
  else rklog::g_level = L::ERROR;
}

static std::string read_level_from_yaml(const std::string& path){
  std::ifstream f(path);
  std::string word;
  while (f >> word){
    if (word.find("log_level:") != std::string::npos){
      std::string val; if (f >> val) return val; else break;
    }
  }
  return "INFO";
}

int main(int argc, char** argv){
  std::string cfg = "config/app.yaml";
  for (int i = 1; i < argc; ++i){
    std::string a = argv[i];
    if ((a == "-c" || a == "--config") && i + 1 < argc) cfg = argv[++i];
  }
  set_log_level(read_level_from_yaml(cfg));
  LOGI("Hello RK3588 (x86/QEMU template). cfg=", cfg);
  return 0;
}
