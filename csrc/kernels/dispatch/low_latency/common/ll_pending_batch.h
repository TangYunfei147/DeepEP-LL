#pragma once
#include <cstdint>
namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PENDING_INLINE __aicore__ inline
#else
#define LL_PENDING_INLINE inline
#endif
// Descending swap-with-last deletes a fully ready suffix by self-copying every
// removed item. The remaining prefix is unchanged, so those copies are optional.
// Do not extend this shortcut to overlapping interior ranges: their resulting
// pending permutation differs from a simple tail copy.
LL_PENDING_INLINE bool LlCanDropReadySuffix(uint32_t pending,uint32_t start,
                                            uint32_t count,uint32_t readyMask) {
    return count>0U && count<=16U && count<=pending && start==pending-count &&
           readyMask==((uint32_t(1)<<count)-1U);
}
#undef LL_PENDING_INLINE
}
