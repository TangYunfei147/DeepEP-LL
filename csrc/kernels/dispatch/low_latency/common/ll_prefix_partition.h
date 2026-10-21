#pragma once
#include "ll_core_config.h"
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PREFIX_INLINE __aicore__ inline
#else
#define LL_PREFIX_INLINE inline
#endif
// One producer per 32-byte GM block; no partial-block sharing between AIVs.
struct LlPrefixPartition { uint32_t begin=0, count=0, active=0; };
LL_PREFIX_INLINE LlPrefixPartition LlMakePrefixPartition(uint32_t experts, uint32_t worker) {
    const uint32_t blocks=(experts+7U)/8U;
    const uint32_t workers=kLlPrefixR0 ? 1U:kLlRecvCoreNum;
    const uint32_t active=blocks<workers ? blocks:workers;
    if (worker>=active) return {0U,0U,active};
    const uint32_t base=blocks/workers, rem=blocks%workers;
    const uint32_t begin=(worker*base+(worker<rem ? worker:rem))*8U;
    const uint32_t limit=begin+(base+(worker<rem ? 1U:0U))*8U;
    const uint32_t end=limit<experts ? limit:experts;
    return {begin,end-begin,active};
}
LL_PREFIX_INLINE uint32_t LlPrefixMaxRecords(uint32_t experts) {
    const uint32_t workers=kLlPrefixR0 ? 1U:kLlRecvCoreNum;
    const uint32_t bound=(((experts+7U)/8U+workers-1U)/workers)*8U;
    return bound<experts ? bound:experts;
}
#undef LL_PREFIX_INLINE
}
