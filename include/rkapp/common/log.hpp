#pragma once

#include <atomic>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace rklog {
enum Level { TRACE = 0, DEBUG, INFO, WARN, ERROR };
inline std::atomic<Level> g_level{INFO};
inline std::mutex g_mu;

inline const char* lvlstr(Level l) {
  switch (l) {
    case TRACE: return "TRACE";
    case DEBUG: return "DEBUG";
    case INFO: return "INFO";
    case WARN: return "WARN";
    default: return "ERROR";
  }
}

template <typename... A>
inline void write(Level l, const char* file, int line, const A&... a) {
  const Level current = g_level.load(std::memory_order_relaxed);
  if (l < current) return;
  std::ostringstream os;
  (void)std::initializer_list<int>{(os << a, 0)...};
  auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

  struct tm tm_buf;
#if defined(_WIN32) || defined(_WIN64)
  localtime_s(&tm_buf, &t);
#else
  localtime_r(&t, &tm_buf);
#endif

  std::lock_guard<std::mutex> lk(g_mu);
  std::cerr << "[" << lvlstr(l) << "] " << std::put_time(&tm_buf, "%F %T") << " " << file
            << ":" << line << " | " << os.str() << "\n";
}
}  // namespace rklog

#define LOGT(...) rklog::write(rklog::TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define LOGD(...) rklog::write(rklog::DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOGI(...) rklog::write(rklog::INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOGW(...) rklog::write(rklog::WARN, __FILE__, __LINE__, __VA_ARGS__)
#define LOGE(...) rklog::write(rklog::ERROR, __FILE__, __LINE__, __VA_ARGS__)
