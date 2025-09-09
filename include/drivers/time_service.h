#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

// 返回单调时钟的微秒计数（32位回绕）
uint32_t micros(void);

#ifdef __cplusplus
}
#endif

