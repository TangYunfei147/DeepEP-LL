#pragma once
#include "ll_core_config.h"
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_SPLIT_INLINE __aicore__ inline
#else
#define LL_SPLIT_INLINE inline
#endif
LL_SPLIT_INLINE constexpr bool LlSmallBsCoreSplit(uint32_t cores, uint32_t bs) {
    return bs > 0U && cores > 1U && bs < cores - 1U;
}
LL_SPLIT_INLINE constexpr uint32_t LlDispatchSendCores(uint32_t cores, uint32_t bs) {
    return LlSmallBsCoreSplit(cores, bs) ? bs : cores / 2U;
}
LL_SPLIT_INLINE constexpr uint32_t LlDispatchRecvCores(uint32_t cores, uint32_t bs) {
    return cores - LlDispatchSendCores(cores, bs);
}
#undef LL_SPLIT_INLINE
}
