#pragma once
#include <cstdint>
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PARALLEL_PREFIX_INLINE __aicore__ inline
#else
#define LL_PARALLEL_PREFIX_INLINE inline
#endif
struct LlPrefixSlice { uint32_t begin,count,active; };
LL_PARALLEL_PREFIX_INLINE LlPrefixSlice LlParallelPrefixSlice(uint32_t records,uint32_t receivers,uint32_t receiver) {
    const uint32_t base=records/receivers,extra=records%receivers;
    return {receiver*base+(receiver<extra?receiver:extra),base+(receiver<extra),records<receivers?records:receivers};
}
// Fixed unbanked records in each core's existing 512B magic slot. Offsets0/32
// remain the old magic/legacy-R0 epoch; new records do not depend on E/window.
LL_PARALLEL_PREFIX_INLINE uint64_t LlPrefixTotalControlByte(uint32_t core) {
    return uint64_t(core)*512U+64U;
}
LL_PARALLEL_PREFIX_INLINE uint64_t LlPrefixReadyControlByte(uint32_t core) {
    return uint64_t(core)*512U+96U;
}
LL_PARALLEL_PREFIX_INLINE uint32_t LlNextPrefixGeneration(uint32_t previous) {
    return previous==UINT32_MAX?1U:previous+1U;
}
#undef LL_PARALLEL_PREFIX_INLINE
}
