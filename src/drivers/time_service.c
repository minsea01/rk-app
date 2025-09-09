#include "drivers/time_service.h"
#include <time.h>

uint32_t micros(void) {
  static uint64_t base_us = 0;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  uint64_t now_us = (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
  if (base_us == 0) base_us = now_us;
  uint64_t rel = now_us - base_us;
  return (uint32_t)(rel & 0xFFFFFFFFu); // 32 位回绕兼容
}

